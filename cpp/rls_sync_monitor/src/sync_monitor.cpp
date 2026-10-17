// Live stream-health and synchronisation monitor for the capture graph (rclcpp).
//
// Subscribes to every recorded sensor stream and, while an episode is running,
// tracks per stream: received messages, frames missing (header-stamp gaps longer
// than 1.5 nominal periods), longest gap, and the age of each message on arrival
// (receive time minus header stamp; a large or negative age means the sensor
// clock is offset from the robot clock). It also tracks the skew between the
// front and wrist camera header stamps. Publishes diagnostic_msgs/DiagnosticArray
// on /diagnostics once per second and writes a JSON summary on shutdown.
//
// This is the operator-facing, online counterpart of the offline ingest checks:
// it cannot correct anything, but it flags a bad recording while it is happening.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "geometry_msgs/msg/wrench_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/string.hpp"

namespace rls {

struct StreamStats {
  double period_s = 0.0;
  long received = 0;      // inside episodes
  long missing = 0;       // estimated from gaps
  double max_gap_s = 0.0;
  double last_rx = -1.0;
  double last_stamp = -1.0;
  std::vector<double> ages;  // receive - header stamp, seconds (bounded)

  void reset_episode() { last_rx = -1.0; }

  double median_age() const {
    if (ages.empty()) return 0.0;
    std::vector<double> a(ages);
    std::nth_element(a.begin(), a.begin() + a.size() / 2, a.end());
    return a[a.size() / 2];
  }
  double drop_rate() const {
    const double expected = static_cast<double>(received + missing);
    return expected > 0 ? missing / expected : 0.0;
  }
};

class SyncMonitor : public rclcpp::Node {
 public:
  SyncMonitor() : Node("sync_monitor") {
    summary_path_ = declare_parameter<std::string>("summary_path", "");
    drop_warn_ = declare_parameter<double>("drop_warn", 0.01);
    drop_error_ = declare_parameter<double>("drop_error", 0.05);
    age_warn_s_ = declare_parameter<double>("age_warn_s", 0.1);
    skew_warn_s_ = declare_parameter<double>("skew_warn_s", 0.05);

    auto qos = rclcpp::QoS(50).reliable();
    add_stream<sensor_msgs::msg::JointState>("joint_states", "/joint_states", 120.0, qos);
    add_stream<geometry_msgs::msg::WrenchStamped>("wrist_ft", "/wrist_ft", 120.0, qos);
    add_stream<sensor_msgs::msg::CompressedImage>("cam_front", "/cam_front/color/image_raw/compressed", 15.0, qos);
    add_stream<sensor_msgs::msg::Image>("cam_front_depth", "/cam_front/depth/image_raw", 15.0, qos);
    add_stream<sensor_msgs::msg::CompressedImage>("cam_wrist", "/cam_wrist/color/image_raw/compressed", 15.0, qos);

    events_ = create_subscription<std_msgs::msg::String>(
        "/episode/events", rclcpp::QoS(10).reliable().transient_local(), [this](const std_msgs::msg::String& m) {
          if (m.data.find("\"event\": \"start\"") != std::string::npos) {
            in_episode_ = true;
            ++episodes_;
            for (auto& [_, s] : stats_) s.reset_episode();
          } else if (m.data.find("\"event\": \"end\"") != std::string::npos) {
            in_episode_ = false;
          }
        });
    diag_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/diagnostics", 10);
    timer_ = create_wall_timer(std::chrono::seconds(1), [this] { publish_diagnostics(); });
    RCLCPP_INFO(get_logger(), "sync_monitor watching %zu streams", stats_.size());
  }

  ~SyncMonitor() override { write_summary(); }

 private:
  template <typename MsgT>
  void add_stream(const std::string& key, const std::string& topic, double rate_hz, const rclcpp::QoS& qos) {
    stats_[key].period_s = 1.0 / rate_hz;
    subs_.push_back(create_subscription<MsgT>(topic, qos, [this, key](const typename MsgT::SharedPtr msg) {
      on_message(key, rclcpp::Time(msg->header.stamp).seconds());
    }));
  }

  void on_message(const std::string& key, double stamp) {
    if (!in_episode_) return;
    const double rx = now().seconds();
    StreamStats& s = stats_[key];
    if (s.last_rx >= 0.0) {
      // gaps from header stamps: receive times are quantised to the /clock rate under sim time,
      // and a sensor's own clock is regular even when it is offset from the robot clock
      const double gap = stamp - s.last_stamp;
      s.max_gap_s = std::max(s.max_gap_s, gap);
      if (gap > 1.5 * s.period_s) s.missing += std::lround(gap / s.period_s) - 1;
      if (gap > 0.3) RCLCPP_WARN(get_logger(), "%s: %.3f s without data (stamp %.3f, now %.3f)", key.c_str(), gap, stamp, rx);
    }
    s.last_rx = rx;
    s.last_stamp = stamp;
    ++s.received;
    if (s.ages.size() < 20000) s.ages.push_back(rx - stamp);
    if (key == "cam_wrist" || key == "cam_front") {
      const auto& f = stats_["cam_front"];
      const auto& w = stats_["cam_wrist"];
      if (f.last_stamp >= 0.0 && w.last_stamp >= 0.0 && skews_.size() < 20000) skews_.push_back(w.last_stamp - f.last_stamp);
    }
  }

  double median_skew() const {
    if (skews_.empty()) return 0.0;
    std::vector<double> a(skews_);
    std::nth_element(a.begin(), a.begin() + a.size() / 2, a.end());
    return a[a.size() / 2];
  }

  uint8_t level_for(const StreamStats& s, std::string& why) const {
    uint8_t lvl = diagnostic_msgs::msg::DiagnosticStatus::OK;
    if (s.drop_rate() > drop_error_) {
      lvl = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
      why = "drop rate above error budget";
    } else if (s.drop_rate() > drop_warn_) {
      lvl = diagnostic_msgs::msg::DiagnosticStatus::WARN;
      why = "drop rate above warning budget";
    }
    if (std::abs(s.median_age()) > age_warn_s_ && lvl == diagnostic_msgs::msg::DiagnosticStatus::OK) {
      lvl = diagnostic_msgs::msg::DiagnosticStatus::WARN;
      why = s.median_age() < 0 ? "header stamps ahead of the robot clock (sensor clock offset)" : "messages arrive late (latency or clock offset)";
    }
    return lvl;
  }

  static diagnostic_msgs::msg::KeyValue kv(const std::string& k, double v) {
    diagnostic_msgs::msg::KeyValue out;
    out.key = k;
    out.value = std::to_string(v);
    return out;
  }

  void publish_diagnostics() {
    diagnostic_msgs::msg::DiagnosticArray arr;
    arr.header.stamp = now();
    for (const auto& [key, s] : stats_) {
      diagnostic_msgs::msg::DiagnosticStatus st;
      st.name = "rls/capture/" + key;
      st.hardware_id = key;
      std::string why = "ok";
      st.level = level_for(s, why);
      st.message = why;
      st.values = {kv("received", s.received), kv("missing", s.missing), kv("drop_rate", s.drop_rate()), kv("max_gap_s", s.max_gap_s),
                   kv("median_age_s", s.median_age())};
      arr.status.push_back(st);
    }
    diagnostic_msgs::msg::DiagnosticStatus sk;
    sk.name = "rls/capture/camera_skew";
    const double skew = median_skew();
    sk.level = std::abs(skew) > skew_warn_s_ ? diagnostic_msgs::msg::DiagnosticStatus::WARN : diagnostic_msgs::msg::DiagnosticStatus::OK;
    sk.message = std::abs(skew) > skew_warn_s_ ? "wrist and front camera stamps disagree; check camera clocks" : "ok";
    sk.values = {kv("median_wrist_minus_front_s", skew)};
    arr.status.push_back(sk);
    diag_pub_->publish(arr);
  }

  void write_summary() {
    if (summary_path_.empty()) return;
    std::ofstream f(summary_path_);
    f << "{\n  \"episodes\": " << episodes_ << ",\n  \"median_wrist_minus_front_s\": " << median_skew() << ",\n  \"streams\": {";
    bool first = true;
    for (const auto& [key, s] : stats_) {
      std::string why;
      const int lvl = level_for(s, why);
      f << (first ? "\n" : ",\n") << "    \"" << key << "\": {\"received\": " << s.received << ", \"missing\": " << s.missing
        << ", \"drop_rate\": " << s.drop_rate() << ", \"max_gap_s\": " << s.max_gap_s << ", \"median_age_s\": " << s.median_age()
        << ", \"level\": " << lvl << "}";
      first = false;
    }
    f << "\n  }\n}\n";
  }

  std::map<std::string, StreamStats> stats_;
  std::vector<rclcpp::SubscriptionBase::SharedPtr> subs_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr events_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::vector<double> skews_;
  bool in_episode_ = false;
  long episodes_ = 0;
  std::string summary_path_;
  double drop_warn_, drop_error_, age_warn_s_, skew_warn_s_;
};

}  // namespace rls

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rls::SyncMonitor>();
  rclcpp::spin(node);
  node.reset();  // destructor writes the summary
  rclcpp::shutdown();
  return 0;
}
