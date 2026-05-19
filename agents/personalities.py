"""
personalities.py
================
F1DriverEnv — a reward-shaping wrapper around CarRacing-v3 that gives
each AI driver a distinct personality.

Imported by train.py, evaluate.py, and the explore notebook.

Driver personalities
--------------------
  divebomber       — fast and reckless, doesn't mind the gravel
  tyre_whisperer   — smooth and consistent, hates leaving the tarmac
  sunday_specialist — balanced and reliable, quietly dangerous
"""

import numpy as np
import gymnasium as gym


class F1DriverEnv(gym.Wrapper):
    """
    Wraps CarRacing-v3 with personality-driven reward shaping.

    Each personality is defined by three levers:
      speed_bonus      — multiplier on positive (on-track) reward
      offtrack_penalty — extra penalty when reward spikes negative (grass/gravel)
      smooth_bonus     — reward for consistent, gentle inputs

    Parameters
    ----------
    personality : str
        One of 'divebomber', 'tyre_whisperer', 'sunday_specialist'.
    **kwargs
        Passed through to gym.make (e.g. render_mode='rgb_array').
    """

    PERSONALITIES = {
        "divebomber": {
            "speed_bonus":      0.5,
            "offtrack_penalty": -0.3,
            "smooth_bonus":     0.0,
            "emoji":            "💥",
            "description":      "Fast and reckless. Asks questions later.",
            "number":           "33",
            "colour":           "#e10600",   # Ferrari red
        },
        "tyre_whisperer": {
            "speed_bonus":      0.1,
            "offtrack_penalty": -1.5,
            "smooth_bonus":     0.4,
            "emoji":            "🧘",
            "description":      "Smooth and consistent. Tyres are a lifestyle.",
            "number":           "44",
            "colour":           "#00d2be",   # Mercedes teal
        },
        "sunday_specialist": {
            "speed_bonus":      0.2,
            "offtrack_penalty": -0.8,
            "smooth_bonus":     0.2,
            "emoji":            "🏆",
            "description":      "Balanced. Reliable. Quietly dangerous.",
            "number":           "1",
            "colour":           "#0600ef",   # Red Bull blue
        },
        "smooth_operator": {
            "speed_bonus":      0.15,
            "offtrack_penalty": -1.2,
            "smooth_bonus":     0.5,        # highest of any driver
            "emoji":            "🎸",       # Carlos is famously into guitar
            "description":      "Calculated, smooth, always in the mix. Never panics.",
            "number":           "55",       # Sainz's actual number
            "colour":           "#ff8000",  # McLaren papaya
        },
    }

    def __init__(self, personality: str = "sunday_specialist", **kwargs):
        if personality not in self.PERSONALITIES:
            raise ValueError(
                f"Unknown personality '{personality}'. "
                f"Choose from: {list(self.PERSONALITIES.keys())}"
            )
        env = gym.make("CarRacing-v3", continuous=True, **kwargs)
        super().__init__(env)
        self.cfg = self.PERSONALITIES[personality]
        self.personality = personality
        self.prev_action = np.zeros(3)

    def reset(self, **kwargs):
        self.prev_action = np.zeros(3)
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Speed bonus: amplify positive (on-track progress) reward
        if reward > 0:
            reward += self.cfg["speed_bonus"] * reward

        # Off-track penalty: negative reward spikes = grass or gravel
        if reward < -0.05:
            reward += self.cfg["offtrack_penalty"]

        # Smoothness bonus: reward gentle, consistent inputs
        input_delta = np.mean(np.abs(action - self.prev_action))
        reward += self.cfg["smooth_bonus"] * max(0.0, 1.0 - input_delta)

        self.prev_action = action.copy()
        return obs, reward, terminated, truncated, info


def list_drivers():
    """Print a summary of all available driver personalities."""
    print("\n🏎️  Box Box Bot — Driver Lineup\n")
    for name, cfg in F1DriverEnv.PERSONALITIES.items():
        print(f"  {cfg['emoji']}  #{cfg['number']}  {name.replace('_', ' ').title()}")
        print(f"       {cfg['description']}")
        print(f"       speed_bonus={cfg['speed_bonus']}  "
              f"offtrack_penalty={cfg['offtrack_penalty']}  "
              f"smooth_bonus={cfg['smooth_bonus']}\n")


if __name__ == "__main__":
    list_drivers()

    # Quick sanity check
    print("Running sanity check on all personalities...")
    for name in F1DriverEnv.PERSONALITIES:
        env = F1DriverEnv(personality=name, render_mode="rgb_array")
        obs, _ = env.reset()
        action = env.action_space.sample()
        obs, reward, _, _, _ = env.step(action)
        env.close()
        print(f"  ✅ {name} — step reward: {reward:.4f}")
