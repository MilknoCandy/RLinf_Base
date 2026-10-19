# Copyright 2026 The RLinf Authors.

from __future__ import annotations

import json
import threading
import time

import numpy as np
import torch

from rlinf.data.schema.embodied_trajectory_builder import (
    EmbodiedTrajectoryBuilder,
)
from rlinf.envs.libero.human_mailbox import (
    HumanCommand,
    HumanMailbox,
    apply_human_override,
    consume_human_skip,
    env0_action_chunk,
    first_env_obs,
    format_hitl_chunk,
    format_hitl_chunk_text,
    mailbox_applies_wasd,
    mailbox_expert_hold,
    normalize_hitl_controller,
    stack_chunk_intervene,
)


def test_mailbox_roundtrip(tmp_path):
    env_box = HumanMailbox(str(tmp_path), action_dim=7, block=True, timeout_s=1.0)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)
    seq = env_box.publish(frame, extra={"elapsed": 3})
    env_box.write_command(hold=True, action=[0.1, 0, 0, 0, 0, 0, -1], seq=seq)
    cmd = env_box.wait_command(seq)
    assert cmd.hold
    assert cmd.seq == seq
    np.testing.assert_allclose(cmd.action[0], 0.1, atol=1e-5)
    assert (tmp_path / "frame_main.jpg").is_file()


def test_mailbox_timeout_is_idle(tmp_path):
    env_box = HumanMailbox(str(tmp_path), action_dim=7, block=True, timeout_s=0.05)
    env_box.publish(np.zeros((8, 8, 3), dtype=np.uint8))
    cmd = env_box.wait_command()
    assert not cmd.hold
    assert cmd.action.shape == (7,)


def test_wait_command_accepts_stale_seq_if_file_is_fresh(tmp_path):
    box = HumanMailbox(str(tmp_path), action_dim=7, block=True, timeout_s=1.0)
    box.publish(np.zeros((8, 8, 3), dtype=np.uint8))
    box.write_command(hold=True, action=[0.0, 0.5, 0, 0, 0, 0, -1], seq=1)
    cmd = box.wait_command(seq=99)
    assert cmd.hold
    np.testing.assert_allclose(cmd.action[1], 0.5, atol=1e-5)


def test_stack_chunk_intervene_matches_update_last_actions():
    infos = []
    for step in range(5):
        infos.append(
            {
                "intervene_action": torch.full((1, 7), float(step)),
                "intervene_flag": torch.tensor([[True]]),
            }
        )
    stacked = stack_chunk_intervene(infos, num_envs=1)
    assert stacked is not None
    assert stacked["intervene_action"].shape == (1, 35)
    assert stacked["intervene_flag"].shape == (1, 5)

    builder = EmbodiedTrajectoryBuilder()
    builder.actions.append(torch.zeros(1, 35))
    builder.intervene_flags.append(torch.zeros(1, 35, dtype=torch.bool))
    builder.update_last_actions(stacked["intervene_action"], stacked["intervene_flag"])
    assert builder.actions[-1].shape == (1, 35)
    assert torch.allclose(builder.actions[-1][0, :7], torch.zeros(7) + 0.0)
    assert torch.allclose(builder.actions[-1][0, 7:14], torch.zeros(7) + 1.0)


def test_stack_chunk_intervene_skips_flag_only_expert_steps():
    infos = [
        {
            "intervene_action": torch.full((1, 7), 0.3),
            "intervene_flag": torch.tensor([[True]]),
        },
        {"intervene_flag": torch.tensor([[True]])},
        {
            "intervene_action": torch.full((1, 7), 0.5),
            "intervene_flag": torch.tensor([[True]]),
        },
        {"intervene_flag": torch.tensor([[True]])},
        {"intervene_flag": torch.tensor([[True]])},
    ]
    stacked = stack_chunk_intervene(infos, num_envs=1)
    assert stacked is not None
    assert stacked["intervene_action"].shape == (1, 35)
    assert stacked["intervene_flag"].shape == (1, 5)
    assert bool(stacked["intervene_flag"][0, 0])
    assert not bool(stacked["intervene_flag"][0, 1])
    builder = EmbodiedTrajectoryBuilder()
    builder.actions.append(torch.arange(35, dtype=torch.float32).unsqueeze(0))
    builder.intervene_flags.append(torch.zeros(1, 35, dtype=torch.bool))
    builder.update_last_actions(stacked["intervene_action"], stacked["intervene_flag"])
    assert builder.actions[-1].shape == (1, 35)


def test_publish_writes_numbered_frame(tmp_path):
    box = HumanMailbox(str(tmp_path), action_dim=7, timeout_s=0.1)
    box.publish(np.zeros((8, 8, 3), dtype=np.uint8), np.ones((8, 8, 3), dtype=np.uint8))
    assert (tmp_path / "frame_main.jpg").is_file()
    assert (tmp_path / "frame_main_1.jpg").is_file()
    assert (tmp_path / "frame_wrist_1.jpg").is_file()
    data = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert data["main"] == "frame_main_1.jpg"
    assert data["wrist"] == "frame_wrist_1.jpg"


def test_numbered_frame_cleanup_keeps_recent(tmp_path):
    box = HumanMailbox(str(tmp_path), action_dim=7, timeout_s=0.1)
    for _ in range(8):
        box.publish(np.zeros((8, 8, 3), dtype=np.uint8))
    assert not (tmp_path / "frame_main_1.jpg").is_file()
    assert (tmp_path / f"frame_main_{box.seq}.jpg").is_file()
    assert (tmp_path / "frame_main.jpg").is_file()


def test_idle_timeout_returns_without_cmd(tmp_path):
    box = HumanMailbox(str(tmp_path), action_dim=7, timeout_s=2.0, idle_timeout_s=0.05)
    box.publish(np.zeros((8, 8, 3), dtype=np.uint8))
    started = time.time()
    cmd = box.wait_command()
    assert time.time() - started < 1.0
    assert not cmd.hold


def test_format_hitl_chunk_marks_human_and_pending_rows():
    actor = np.array(
        [[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9], [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9]],
        dtype=np.float32,
    )
    ref = np.array(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9]],
        dtype=np.float32,
    )
    executed = np.array(
        [[0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9], [np.nan] * 7],
        dtype=np.float32,
    )
    flags = np.array([True, False])
    chunk = format_hitl_chunk(
        actor, executed, flags, chunk_id=3, step_i=1, ref=ref
    )
    assert chunk["id"] == 3
    assert chunk["len"] == 2
    assert chunk["step"] == 1
    assert chunk["human_steps"] == 1
    assert chunk["human"] == [True, False]
    assert chunk["executed"][1][0] is None
    assert chunk["ref"][0][0] == 0.0
    assert chunk["actor"][0][0] == 0.1
    assert chunk["l1_pi_a"][0] == 0.014
    assert chunk["l1_a_exec"][0] == 0.071
    text = format_hitl_chunk_text(chunk)
    assert "chunk #3" in text
    assert "π=VLA ref_chunk" in text
    assert " a " in text
    assert "L1(π,a)=0.014" in text


def test_format_hitl_chunk_text_shows_zap_not_na_cmd():
    actor = np.zeros((2, 7), dtype=np.float32)
    executed = np.array(
        [[0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9], [np.nan] * 7],
        dtype=np.float32,
    )
    pending = format_hitl_chunk(
        actor,
        executed,
        flags=np.zeros(2, dtype=bool),
        chunk_id=4,
        step_i=1,
        expert=np.array([True, False]),
    )
    pending_text = format_hitl_chunk_text(pending)
    assert "zap  (next chunk)" in pending_text
    assert all(not line.strip().startswith("cmd ") for line in pending_text.splitlines())

    active = format_hitl_chunk(
        actor,
        executed,
        flags=np.zeros(2, dtype=bool),
        chunk_id=5,
        step_i=1,
        cmd=executed,
        expert=np.array([True, False]),
    )
    active_text = format_hitl_chunk_text(active)
    assert "zap " in active_text
    assert "  0.40" in active_text
    assert "n/a" not in active_text.split("zap", 1)[1].split("\n", 1)[0]


def test_env0_action_chunk_reads_flat_ref():
    flat = np.arange(20, dtype=np.float32)
    rows = env0_action_chunk(flat.reshape(1, 20), chunk_len=2, action_dim=7)
    assert rows.shape == (2, 7)
    np.testing.assert_allclose(rows[0], np.arange(7, dtype=np.float32))
    stacked = env0_action_chunk(
        np.arange(14, dtype=np.float32).reshape(1, 2, 7),
        chunk_len=2,
        action_dim=7,
    )
    np.testing.assert_allclose(stacked[1], np.arange(7, 14, dtype=np.float32))


def test_apply_human_override_replaces_pose_and_freezes_gripper():
    policy = np.array([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.9]], dtype=np.float32)
    human = np.array([0.6, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    out, holding, latch = apply_human_override(
        policy, human, hold=True, freeze_gripper=True
    )
    assert holding
    np.testing.assert_allclose(out[0, :6], [0.6, 0.2, 0.3, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(out[0, 6], 0.9)
    np.testing.assert_allclose(latch, 0.9)


def test_apply_human_override_skips_idle_zeros():
    policy = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9]], dtype=np.float32)
    human = np.zeros(7, dtype=np.float32)
    out, holding, _latch = apply_human_override(policy, human, hold=True)
    assert out is None
    assert not holding


def test_apply_human_override_pressed_axis_replaces_idle_axes_inherit():
    policy = np.array([0.4, 0.4, 0.4, 0.1, 0.1, 0.1, 0.9], dtype=np.float32)
    human = np.array([0.6, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    out, holding, _latch = apply_human_override(
        policy, human, hold=True, freeze_gripper=True
    )
    assert holding
    np.testing.assert_allclose(out[:6], [0.6, 0.4, 0.4, 0.1, 0.1, 0.1])
    np.testing.assert_allclose(out[6], 0.9)


def test_inherit_gripper_roundtrip(tmp_path):
    box = HumanMailbox(str(tmp_path), action_dim=7, timeout_s=0.1)
    box.publish(np.zeros((8, 8, 3), dtype=np.uint8))
    box.write_command(
        hold=True,
        action=[0.2, 0, 0, 0, 0, 0, -1],
        seq=1,
        inherit_gripper=True,
    )
    cmd = box.wait_command(seq=1)
    assert cmd.hold
    assert cmd.inherit_gripper


def test_skip_command_roundtrip(tmp_path):
    box = HumanMailbox(str(tmp_path), action_dim=7, timeout_s=0.1)
    box.publish(np.zeros((8, 8, 3), dtype=np.uint8))
    box.write_command(
        hold=False,
        action=[0, 0, 0, 0, 0, 0, -1],
        seq=1,
        skip=True,
        skip_id=3,
    )
    cmd = box.wait_command(seq=1)
    assert cmd.skip
    assert cmd.skip_id == 3
    assert not cmd.hold


def test_missing_skip_defaults_false(tmp_path):
    box = HumanMailbox(str(tmp_path), action_dim=7, timeout_s=0.1)
    (tmp_path / "cmd.json").write_text(
        json.dumps({"seq": 1, "hold": False, "action": [0, 0, 0, 0, 0, 0, -1]}),
        encoding="utf-8",
    )
    cmd = box.read_command()
    assert cmd is not None
    assert cmd.skip is False
    assert cmd.skip_id == 0


def test_pause_command_roundtrip(tmp_path):
    box = HumanMailbox(str(tmp_path), action_dim=7, timeout_s=0.1)
    box.publish(np.zeros((8, 8, 3), dtype=np.uint8))
    box.write_command(
        hold=False,
        action=[0, 0, 0, 0, 0, 0, -1],
        seq=1,
        pause=True,
    )
    cmd = box.read_command()
    assert cmd is not None
    assert cmd.pause
    assert not cmd.hold
    assert not cmd.skip


def test_missing_pause_defaults_false(tmp_path):
    box = HumanMailbox(str(tmp_path), action_dim=7, timeout_s=0.1)
    (tmp_path / "cmd.json").write_text(
        json.dumps({"seq": 1, "hold": False, "action": [0, 0, 0, 0, 0, 0, -1]}),
        encoding="utf-8",
    )
    cmd = box.read_command()
    assert cmd is not None
    assert cmd.pause is False


def test_pause_blocks_until_cleared(tmp_path):
    box = HumanMailbox(
        str(tmp_path),
        action_dim=7,
        timeout_s=0.4,
        idle_timeout_s=0.05,
        poll_s=0.02,
    )
    box.publish(np.zeros((8, 8, 3), dtype=np.uint8))
    box.write_command(
        hold=False,
        action=[0, 0, 0, 0, 0, 0, -1],
        seq=1,
        pause=True,
    )
    done = []

    def _wait():
        cmd = box.wait_command(seq=1)
        done.append(cmd)

    thread = threading.Thread(target=_wait)
    thread.start()
    time.sleep(0.15)
    assert thread.is_alive()
    box.write_command(
        hold=False,
        action=[0, 0, 0, 0, 0, 0, -1],
        seq=1,
        pause=False,
    )
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert done and not done[0].pause


def test_skip_unblocks_pause(tmp_path):
    box = HumanMailbox(
        str(tmp_path),
        action_dim=7,
        timeout_s=0.4,
        idle_timeout_s=0.05,
        poll_s=0.02,
    )
    box.publish(np.zeros((8, 8, 3), dtype=np.uint8))
    box.write_command(
        hold=False,
        action=[0, 0, 0, 0, 0, 0, -1],
        seq=1,
        pause=True,
        skip=True,
        skip_id=4,
    )
    cmd = box.wait_command(seq=1)
    assert cmd.skip
    assert cmd.skip_id == 4


def test_consume_human_skip_is_edge_triggered():
    idle = HumanCommand.idle(7)
    applied, consumed = consume_human_skip(idle, 0)
    assert not applied
    assert consumed == 0

    first = HumanCommand(
        seq=1,
        hold=False,
        action=np.zeros((7,), dtype=np.float32),
        skip=True,
        skip_id=2,
    )
    applied, consumed = consume_human_skip(first, 0)
    assert applied
    assert consumed == 2
    applied, consumed = consume_human_skip(first, consumed)
    assert not applied
    assert consumed == 2

    sticky = HumanCommand(
        seq=2,
        hold=False,
        action=np.zeros((7,), dtype=np.float32),
        skip=True,
        skip_id=2,
    )
    applied, consumed = consume_human_skip(sticky, consumed)
    assert not applied


def test_missing_inherit_gripper_defaults_true(tmp_path):
    box = HumanMailbox(str(tmp_path), action_dim=7, timeout_s=0.1)
    (tmp_path / "cmd.json").write_text(
        json.dumps({"seq": 1, "hold": True, "action": [0, 0, 0, 0, 0, 0, -1]}),
        encoding="utf-8",
    )
    cmd = box.read_command()
    assert cmd is not None
    assert cmd.hold
    assert cmd.inherit_gripper is True


def test_first_env_obs_accepts_object_ndarray():
    obs = {"agentview_image": np.zeros((4, 4, 3), dtype=np.uint8)}
    assert first_env_obs([obs]) is obs
    assert first_env_obs(np.array([obs], dtype=object)) is obs
    assert first_env_obs(obs) is obs
    assert first_env_obs(None) is None
    assert first_env_obs(np.zeros((2, 3))) is None


def test_idle_timeout_does_not_drop_active_hold(tmp_path):
    box = HumanMailbox(str(tmp_path), action_dim=7, timeout_s=0.4, idle_timeout_s=0.05)
    box.publish(np.zeros((8, 8, 3), dtype=np.uint8))
    box.last_cmd = HumanCommand(
        seq=1, hold=True, action=np.zeros((7,), dtype=np.float32)
    )
    started = time.time()
    box.wait_command()
    assert time.time() - started >= 0.3
    assert time.time() - started < 1.0


def test_controller_defaults_human_when_missing(tmp_path):
    box = HumanMailbox(str(tmp_path), action_dim=7, timeout_s=0.1)
    (tmp_path / "cmd.json").write_text(
        json.dumps({"seq": 1, "hold": True, "action": [0.2, 0, 0, 0, 0, 0, -1]}),
        encoding="utf-8",
    )
    cmd = box.read_command()
    assert cmd is not None
    assert cmd.controller == "human"
    assert mailbox_applies_wasd(cmd)
    assert not mailbox_expert_hold(cmd)


def test_controller_expert_hold_roundtrip(tmp_path):
    box = HumanMailbox(str(tmp_path), action_dim=7, timeout_s=0.1)
    box.write_command(
        hold=True,
        action=[0, 0, 0, 0, 0, 0, -1],
        seq=3,
        controller="zap",
    )
    cmd = box.read_command()
    assert cmd is not None
    assert cmd.controller == "expert"
    assert cmd.hold
    assert mailbox_expert_hold(cmd)
    assert not mailbox_applies_wasd(cmd)
    idle = HumanCommand.idle(7)
    assert idle.controller == "human"
    assert not mailbox_expert_hold(idle)
    assert normalize_hitl_controller("teacher") == "expert"
    assert normalize_hitl_controller(None) == "human"


def test_append_episode_writes_jsonl(tmp_path):
    box = HumanMailbox(str(tmp_path), action_dim=7, timeout_s=0.1)
    box.append_episode(
        {
            "success": True,
            "intervened": False,
            "mode": "vla",
            "result": "success",
            "elapsed": 12,
        }
    )
    box.append_episode(
        {
            "success": False,
            "intervened": True,
            "mode": "hitl",
            "result": "fail",
            "elapsed": 240,
        }
    )
    lines = (tmp_path / "episodes.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["mode"] == "vla" and first["result"] == "success"
    assert second["mode"] == "hitl" and second["result"] == "fail"
