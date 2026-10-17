from pathlib import Path

import numpy as np
import pytest
import torch

from rls.learn.export import export_onnx
from rls.learn.policies import Normalizer, Policy, PolicyBundle
from rls.learn.policies.layers import LoRALinear, add_lora, merge_lora
from rls.learn.runner import ChunkRunner
from rls.learn.text import INDEX, tokenize

SMALL = {"d_model": 32, "heads": 2, "enc_layers": 1, "dec_layers": 1, "chunk": 4, "cnn_width": 8, "front_hw": [24, 32], "wrist_hw": [16, 16]}


def _batch(b: int = 3) -> dict[str, torch.Tensor]:
    return {
        "front": torch.randint(0, 255, (b, 24, 32, 3), dtype=torch.uint8),
        "wrist": torch.randint(0, 255, (b, 16, 16, 3), dtype=torch.uint8),
        "state": torch.randn(b, 8),
        "tokens": torch.tensor([tokenize("put the red cube in the bin")] * b),
        "action": torch.randn(b, 4, 8),
        "action_valid": torch.ones(b, 4, dtype=torch.bool),
    }


def test_tokenizer_knows_unseen_colours() -> None:
    assert tokenize("stack the purple cube on the orange cube")[2] == INDEX["purple"]


def test_loss_decreases_on_a_fixed_batch() -> None:
    torch.manual_seed(0)
    m = Policy.build("chunk_transformer", {**SMALL, "dropout": 0.0, "augment_shift": 0})
    b = _batch()
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    first = float(m.compute_loss(b)["loss"])
    for _ in range(100):
        loss = m.compute_loss(b)["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.5 * first


def test_lora_is_a_no_op_at_init_and_merges_exactly() -> None:
    torch.manual_seed(0)
    m = Policy.build("chunk_transformer", {**SMALL, "dropout": 0.0, "augment_shift": 0}).eval()
    b = _batch()
    ref = m(b["front"], b["wrist"], b["state"], b["tokens"])
    n = add_lora(m.encoder) + add_lora(m.decoder)
    assert n > 0
    assert torch.allclose(m(b["front"], b["wrist"], b["state"], b["tokens"]), ref, atol=1e-6)
    for mod in m.modules():
        if isinstance(mod, LoRALinear):
            torch.nn.init.normal_(mod.lora_b, std=0.1)
    tuned = m(b["front"], b["wrist"], b["state"], b["tokens"])
    merge_lora(m)
    assert not any(isinstance(x, LoRALinear) for x in m.modules())
    assert torch.allclose(m(b["front"], b["wrist"], b["state"], b["tokens"]), tuned, atol=1e-5)


@pytest.mark.parametrize("name", ["chunk_transformer", "diffusion_transformer"])
def test_onnx_export_parity_and_runner(tmp_path: Path, name: str) -> None:
    m = Policy.build(name, {**SMALL, "infer_steps": 4})
    norm = Normalizer(np.zeros(8), np.ones(8), np.zeros(8), np.ones(8))
    PolicyBundle(m, norm, {"run_name": "t"}).save(tmp_path / "b")
    rep = export_onnx(tmp_path / "b", tmp_path / "p.onnx", check_samples=3)
    assert rep["max_abs_diff"] < 1e-4
    bundle = PolicyBundle.load(tmp_path / "b")
    r = ChunkRunner(bundle)
    r.reset("put the red cube in the bin")
    acts = [r.act(np.zeros((24, 32, 3), np.uint8), np.zeros((16, 16, 3), np.uint8), np.zeros(8, np.float32)) for _ in range(6)]
    assert all(a.shape == (8,) and 0 <= a[7] <= 1 for a in acts)


def test_diffusion_sampler_recovers_a_memorised_chunk() -> None:
    """Train the denoiser on one batch; DDIM sampling from the fixed latent must land near the target."""
    torch.manual_seed(0)
    m = Policy.build("diffusion_transformer", {**SMALL, "dropout": 0.0, "augment_shift": 0, "train_steps": 50, "infer_steps": 5})
    b = _batch(2)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    m.eval()
    before = float((m(b["front"], b["wrist"], b["state"], b["tokens"]) - b["action"]).abs().mean())
    m.train()
    for _ in range(300):
        loss = m.compute_loss(b)["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    after = float((m(b["front"], b["wrist"], b["state"], b["tokens"]) - b["action"]).abs().mean())
    assert after < 0.35 * before, (before, after)


def test_ee_action_space_round_trip() -> None:
    """Tool-pose actions recovered by FK map back to the same joint targets through IK."""
    from rls.learn.action_space import EEActionSpace
    from rls.sim.ik import down_rotation
    from rls.sim.scene import HOME_Q

    sp = EEActionSpace()
    rng = np.random.default_rng(0)
    q = np.array(HOME_Q, dtype=np.float64)
    joint = []
    for _ in range(20):  # a smooth gripper-down path, as the demonstrations are
        pos, _ = sp.ik.fk(q)
        q, _ = sp.ik.solve(q, pos + rng.normal(0, 0.01, 3), down_rotation(sp.ik.hand_yaw(q) + rng.normal(0, 0.05)))
        joint.append(np.r_[q, rng.integers(0, 2)])
    joint_arr = np.array(joint, np.float32)
    ee = sp.from_joint(joint_arr)
    assert ee.shape == (20, 6)
    for k in range(1, 20):
        back = sp.to_joint(ee[k], joint_arr[k - 1, :7])
        p_back, _ = sp.ik.fk(back[:7])
        p_true, _ = sp.ik.fk(joint_arr[k, :7])
        assert np.linalg.norm(p_back - p_true) < 1e-3 and back[7] == joint_arr[k, 7]


def test_lora_bundle_saves_merged_weights(tmp_path: Path) -> None:
    """A LoRA-fine-tuned policy is saved with the adapters merged and reloads to the same outputs."""
    torch.manual_seed(0)
    m = Policy.build("chunk_transformer", {**SMALL, "dropout": 0.0, "augment_shift": 0}).eval()
    add_lora(m.encoder) + add_lora(m.decoder)
    for mod in m.modules():
        if isinstance(mod, LoRALinear):
            torch.nn.init.normal_(mod.lora_b, std=0.1)
    b = _batch()
    ref = m(b["front"], b["wrist"], b["state"], b["tokens"])
    PolicyBundle(m, Normalizer(np.zeros(8), np.ones(8), np.zeros(8), np.ones(8)), {}).save(tmp_path / "b")
    assert not any(k.endswith("lora_a") for k in torch.load(tmp_path / "b" / "model.pt"))
    assert any(isinstance(x, LoRALinear) for x in m.modules())  # the live model keeps training with adapters
    back = PolicyBundle.load(tmp_path / "b").model
    assert torch.allclose(back(b["front"], b["wrist"], b["state"], b["tokens"]), ref, atol=1e-5)
