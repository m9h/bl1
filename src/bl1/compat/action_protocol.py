"""Action inference protocol: doom-neuron -> BL-1.

Sent on port 12352 so BL-1 can visualize what the decoder is inferring
from the neural spike responses.

Packet Format (80 bytes):
    [8B  timestamp]           uint64, microseconds
    [3 x 4B forward_probs]   float32[3], softmax [none, forward, backward]
    [3 x 4B strafe_probs]    float32[3], softmax [none, left, right]
    [3 x 4B camera_probs]    float32[3], softmax [none, turn_left, turn_right]
    [4B  attack_prob]         float32, sigmoid probability of attack
    [4B  forward_action]     uint32, selected action (0/1/2)
    [4B  strafe_action]      uint32, selected action (0/1/2)
    [4B  camera_action]      uint32, selected action (0/1/2)
    [4B  attack_action]      uint32, selected action (0/1)
    [4B  reward]             float32, current step reward
    [4B  episode_reward]     float32, cumulative episode reward
    [4B  kill_count]         float32, kills this episode

Usage (doom-neuron sender)::

    from action_protocol import pack_action_inference, ACTION_PORT
    sock.sendto(
        pack_action_inference(fwd_probs, strafe_probs, cam_probs, atk_prob,
                              fwd_act, str_act, cam_act, atk_act,
                              reward, ep_reward, kills),
        (bl1_host, ACTION_PORT),
    )

Usage (BL-1 receiver)::

    from bl1.compat.action_protocol import unpack_action_inference, ACTION_PORT
    data = unpack_action_inference(packet)
    # data['forward_probs'] = [0.1, 0.8, 0.1]  etc.
"""

from __future__ import annotations

import struct
import time

import numpy as np

ACTION_PORT = 12352

# 8 + 12 + 12 + 12 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 = 76... let me recount
# 8 (ts) + 3*4 (fwd) + 3*4 (strafe) + 3*4 (cam) + 4 (atk_prob)
# + 4*4 (actions) + 4 (reward) + 4 (ep_reward) + 4 (kills) = 80
ACTION_PACKET_SIZE = 80
ACTION_FORMAT = "<Q3f3f3fffffffffff"
# Unrolled: Q + 3f(fwd) + 3f(strafe) + 3f(cam) + f(atk) + 4f(actions) + 3f(metrics)
# = 8 + 12 + 12 + 12 + 4 + 16 + 12 = 76... hmm

# Let me be precise:
# Q=8, 3f=12, 3f=12, 3f=12, f=4, I=4, I=4, I=4, I=4, f=4, f=4, f=4 = 80
ACTION_FORMAT = "<Q9fffIIIIfff"

assert struct.calcsize(ACTION_FORMAT) == ACTION_PACKET_SIZE, (
    f"Format size {struct.calcsize(ACTION_FORMAT)} != expected {ACTION_PACKET_SIZE}"
)

# Action labels for visualization
FORWARD_LABELS = ["none", "forward", "backward"]
STRAFE_LABELS = ["none", "left", "right"]
CAMERA_LABELS = ["none", "turn_left", "turn_right"]


def pack_action_inference(
    forward_probs: np.ndarray,
    strafe_probs: np.ndarray,
    camera_probs: np.ndarray,
    attack_prob: float,
    forward_action: int,
    strafe_action: int,
    camera_action: int,
    attack_action: int,
    reward: float = 0.0,
    episode_reward: float = 0.0,
    kill_count: float = 0.0,
) -> bytes:
    """Pack decoder inference into an 80-byte UDP packet."""
    timestamp = int(time.time() * 1_000_000)
    fp = np.asarray(forward_probs, dtype=np.float32)
    sp = np.asarray(strafe_probs, dtype=np.float32)
    cp = np.asarray(camera_probs, dtype=np.float32)

    return struct.pack(
        ACTION_FORMAT,
        timestamp,
        fp[0],
        fp[1],
        fp[2],
        sp[0],
        sp[1],
        sp[2],
        cp[0],
        cp[1],
        cp[2],
        float(attack_prob),
        int(forward_action),
        int(strafe_action),
        int(camera_action),
        int(attack_action),
        float(reward),
        float(episode_reward),
        float(kill_count),
    )


def unpack_action_inference(packet: bytes) -> dict:
    """Unpack an 80-byte action inference packet."""
    if len(packet) != ACTION_PACKET_SIZE:
        raise ValueError(f"Expected {ACTION_PACKET_SIZE} bytes, got {len(packet)}")

    v = struct.unpack(ACTION_FORMAT, packet)
    return {
        "timestamp_us": v[0],
        "forward_probs": np.array(v[1:4], dtype=np.float32),
        "strafe_probs": np.array(v[4:7], dtype=np.float32),
        "camera_probs": np.array(v[7:10], dtype=np.float32),
        "attack_prob": v[10],
        "forward_action": v[11],
        "strafe_action": v[12],
        "camera_action": v[13],
        "attack_action": v[14],
        "reward": v[15],
        "episode_reward": v[16],
        "kill_count": v[17],
    }
