"""
Worker process for multi-env PPO.
Must live in a .py file (not the notebook) so multiprocessing 'spawn' can import it on Windows.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import cv2
import vizdoom as vzd

FRAME_STACK = 4
IMG_H, IMG_W = 84, 84


def _make_game(scenario, buttons):
    game = vzd.DoomGame()
    game.load_config(f"{vzd.scenarios_path}/{scenario}.cfg")
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_window_visible(False)
    game.clear_available_buttons()
    for btn in buttons:
        game.add_available_button(btn)
    game.init()
    return game


def _preprocess(frame):
    frame = cv2.resize(frame, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return frame.astype(np.float32) / 255.0


def env_worker(conn, scenario, buttons, health_loss_coeff=0.0, kill_reward=0.0, item_bonus=0.0):
    """
    Worker process: owns one DoomEnv instance.
    Responds to commands from the main process via a multiprocessing Pipe.

    Commands (sent as (cmd, data) tuples):
        ('step',  action_idx) -> (next_state, reward, done)   auto-resets on done
        ('reset', None)       -> initial_state
        ('close', None)       -> (no response, process exits)

    health_loss_coeff : penalty per HP lost each step (requires HEALTH in game_variables).
    kill_reward       : bonus per enemy killed (requires KILLCOUNT in game_variables).
    item_bonus        : bonus per item picked up (requires ITEMCOUNT in game_variables).
                        The game's native distance reward is ignored for non-terminal steps;
                        only the death penalty (terminal step) is kept from the game engine.
        game_variables order must be { HEALTH KILLCOUNT ITEMCOUNT } in the cfg.
    """
    game        = _make_game(scenario, buttons)
    num_actions = len(buttons)
    actions     = np.eye(num_actions, dtype=np.uint8).tolist()
    frame_buf   = np.zeros((FRAME_STACK, IMG_H, IMG_W), dtype=np.float32)
    prev_health    = 100.0
    prev_killcount = 0.0
    prev_itemcount = 0.0

    def _read_vars(state):
        gv = state.game_variables
        health    = float(gv[0]) if gv is not None and len(gv) > 0 else 100.0
        killcount = float(gv[1]) if gv is not None and len(gv) > 1 else 0.0
        itemcount = float(gv[2]) if gv is not None and len(gv) > 2 else 0.0
        return health, killcount, itemcount

    def _reset():
        nonlocal prev_health, prev_killcount, prev_itemcount
        game.new_episode()
        state = game.get_state()
        frame_buf[:] = _preprocess(state.screen_buffer)
        prev_health, prev_killcount, prev_itemcount = _read_vars(state)
        return frame_buf.copy()

    def _step(action_idx):
        nonlocal prev_health, prev_killcount, prev_itemcount
        game_reward = game.make_action(actions[action_idx])
        done        = game.is_episode_finished()
        if not done:
            state      = game.get_state()
            next_frame = _preprocess(state.screen_buffer)
            cur_health, cur_kills, cur_items = _read_vars(state)

            # Keep game's distance reward as navigation signal; add kill/item shaping on top
            reward = game_reward
            if health_loss_coeff > 0:
                reward -= health_loss_coeff * max(0.0, prev_health - cur_health)
            if kill_reward > 0:
                reward += kill_reward * max(0.0, cur_kills - prev_killcount)
            if item_bonus > 0:
                reward += item_bonus * max(0.0, cur_items - prev_itemcount)

            prev_health    = cur_health
            prev_killcount = cur_kills
            prev_itemcount = cur_items
        else:
            next_frame = np.zeros((IMG_H, IMG_W), dtype=np.float32)
            reward = game_reward  # death penalty from cfg (negative on death)

        frame_buf[:-1] = frame_buf[1:]
        frame_buf[-1]  = next_frame
        return frame_buf.copy(), reward, done

    _reset()  # warm-up so first 'step' doesn't fail

    while True:
        try:
            cmd, data = conn.recv()
        except EOFError:
            break

        if cmd == 'step':
            next_state, reward, done = _step(data)
            if done:
                next_state = _reset()
            conn.send((next_state, reward, done))

        elif cmd == 'reset':
            conn.send(_reset())

        elif cmd == 'close':
            game.close()
            conn.close()
            return
