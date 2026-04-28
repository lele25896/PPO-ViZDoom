"""
Standalone script for recording ViZDoom model videos.
Run from notebook: !python record_video.py --scenario basic --model model_basic_multienv.pth
Must run in a separate process — ViZDoom crashes with access violation
when initialized directly in the Jupyter kernel process on Windows.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import numpy as np
import torch
import torch.nn as nn
import cv2
import vizdoom as vzd

# ── Constants (must match training) ──────────────────────────────────────────
FRAME_STACK = 4
IMG_H, IMG_W = 84, 84
HIDDEN_DIM  = 512
LSTM_HIDDEN = 256

BASIC_BUTTONS = [
    vzd.Button.MOVE_LEFT,
    vzd.Button.MOVE_RIGHT,
    vzd.Button.ATTACK,
]
FULL_BUTTONS = [
    vzd.Button.MOVE_FORWARD,
    vzd.Button.MOVE_BACKWARD,
    vzd.Button.MOVE_LEFT,
    vzd.Button.MOVE_RIGHT,
    vzd.Button.TURN_LEFT,
    vzd.Button.TURN_RIGHT,
    vzd.Button.ATTACK,
]

SCENARIOS = {
    "basic":           ("basic",                  BASIC_BUTTONS),
    "defend":          ("defend_the_center",      FULL_BUTTONS),
    "corridor_easy":   ("deadly_corridor_easy",   FULL_BUTTONS),
    "corridor_medium": ("deadly_corridor_medium", FULL_BUTTONS),
    "corridor":        ("deadly_corridor",        FULL_BUTTONS),
}


class ActorCriticLSTM(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, HIDDEN_DIM),
            nn.ReLU(),
        )
        self.lstm   = nn.LSTMCell(HIDDEN_DIM, LSTM_HIDDEN)
        self.actor  = nn.Linear(LSTM_HIDDEN, num_actions)
        self.critic = nn.Linear(LSTM_HIDDEN, 1)

    def forward(self, x, lstm_state):
        features = self.cnn(x)
        h, c = self.lstm(features, lstm_state)
        return self.actor(h), self.critic(h), (h, c)

    def init_lstm_state(self, device):
        return (
            torch.zeros(1, LSTM_HIDDEN, device=device),
            torch.zeros(1, LSTM_HIDDEN, device=device),
        )


def record(model_path, scenario_name, output_path, max_episodes, fps, width, height, cfg_path=None):
    scenario_cfg, buttons = SCENARIOS[scenario_name]
    num_actions = len(buttons)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    game = vzd.DoomGame()
    if cfg_path:
        game.load_config(cfg_path)
    else:
        game.load_config(f"{vzd.scenarios_path}/{scenario_cfg}.cfg")
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_window_visible(False)
    game.clear_available_buttons()
    for btn in buttons:
        game.add_available_button(btn)
    game.init()

    actions_onehot = np.eye(num_actions, dtype=np.uint8).tolist()

    policy = ActorCriticLSTM(num_actions).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    policy.eval()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_rewards = []
    for ep in range(max_episodes):
        game.new_episode()
        frame_buf  = np.zeros((FRAME_STACK, IMG_H, IMG_W), dtype=np.float32)
        lstm_state = policy.init_lstm_state(device)
        steps      = 0

        while not game.is_episode_finished():
            rgb = game.get_state().screen_buffer
            writer.write(cv2.cvtColor(cv2.resize(rgb, (width, height)), cv2.COLOR_RGB2BGR))

            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            frame_buf = np.roll(frame_buf, -1, axis=0)
            frame_buf[-1] = cv2.resize(gray, (IMG_W, IMG_H)).astype(np.float32) / 255.0

            state_t = torch.tensor(frame_buf, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.inference_mode():
                logits, _, lstm_state = policy(state_t, lstm_state)
            game.make_action(actions_onehot[logits.argmax(dim=-1).item()])
            steps += 1

        ep_reward = game.get_total_reward()
        total_rewards.append(ep_reward)
        print(f"  Ep {ep+1}/{max_episodes} | steps: {steps} | reward: {ep_reward:.1f}")

    writer.release()
    game.close()
    print(f"Saved: '{output_path}'  |  mean reward: {np.mean(total_rewards):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=list(SCENARIOS.keys()), metavar="{" + ",".join(SCENARIOS.keys()) + "}")
    parser.add_argument("--model",    required=True)
    parser.add_argument("--output",   default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--fps",      type=int, default=30)
    parser.add_argument("--width",    type=int, default=640)
    parser.add_argument("--height",   type=int, default=480)
    parser.add_argument("--cfg",      default=None)
    args = parser.parse_args()

    output = args.output or f"video_{args.scenario}.mp4"
    print(f"Recording '{args.scenario}' → '{output}'  ({args.episodes} episodes)")
    record(args.model, args.scenario, output, args.episodes, args.fps, args.width, args.height, args.cfg)
