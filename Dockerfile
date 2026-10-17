# ROS 2 Jazzy + MuJoCo (headless EGL/OSMesa) + the rls package.
#   docker build -t rls .
#   docker run --rm -it rls rls record --out /work/data/raw/s0 --episodes 4
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1 MUJOCO_GL=egl \
    ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-venv libegl1 libgl1 libosmesa6 ffmpeg build-essential cmake \
        ros-jazzy-rosbag2-storage-mcap ros-jazzy-sensor-msgs ros-jazzy-geometry-msgs ros-jazzy-tf2-msgs ros-jazzy-diagnostic-msgs \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
WORKDIR /opt/rls
COPY pyproject.toml README.md ./
COPY rls ./rls
RUN pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu ".[rl,tracking]"
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY cpp ./cpp
RUN source /opt/ros/jazzy/setup.bash && cmake -S cpp/rls_sync_monitor -B /tmp/b -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/rls/cpp/rls_sync_monitor/install \
    && cmake --build /tmp/b && cmake --install /tmp/b && rm -rf /tmp/b

# LeRobot lives in its own environment (pinned, different dependency set); `rls release` calls it.
RUN python3 -m venv /opt/lerobot-venv && /opt/lerobot-venv/bin/pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu "lerobot[dataset]==0.6.1" pyyaml h5py pandas pyarrow
ENV RLS_LEROBOT_PYTHON=/opt/lerobot-venv/bin/python \
    RLS_SYNC_MONITOR=/opt/rls/cpp/rls_sync_monitor/install/lib/rls_sync_monitor/sync_monitor

ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/jazzy/setup.bash && exec \"$@\"", "--"]
CMD ["rls", "--help"]
