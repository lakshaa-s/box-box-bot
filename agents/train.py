"""
train.py
========
Full training script for all four Box Box Bot driver personalities.

Run from the project root:
    python agents/train.py

Trains each driver at checkpoints: 100k, 300k, 600k, 1M steps.
Saves models to:  models/<personality>_<steps>.zip
Saves results to: results/<personality>.csv

To train a single driver:
    python agents/train.py --driver divebomber

To do a quick test run (5k steps):
    python agents/train.py --test
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

# personalities.py lives in the same agents/ folder
import sys
sys.path.append(os.path.dirname(__file__))
from personalities import F1DriverEnv

# ── Config ────────────────────────────────────────────────────────────────────

DRIVERS = [
    "divebomber",
    "tyre_whisperer",
    "sunday_specialist",
    "smooth_operator",
]

CHECKPOINTS = [100_000, 300_000, 600_000, 1_000_000]

N_ENVS          = 4   # parallel environments
N_EVAL_EPISODES = 3   # episodes to average at each checkpoint

# ── Reward logger ─────────────────────────────────────────────────────────────

class RewardLogger(BaseCallback):
    """
    Tracks per-episode rewards during training, one accumulator per
    parallel env so numbers don't bleed across episodes.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self._ep_reward_buffer = None

    def _on_training_start(self) -> None:
        self._ep_reward_buffer = [0.0] * self.training_env.num_envs

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones   = self.locals["dones"]
        for i, (r, done) in enumerate(zip(rewards, dones)):
            self._ep_reward_buffer[i] += r
            if done:
                self.episode_rewards.append(self._ep_reward_buffer[i])
                self._ep_reward_buffer[i] = 0.0
        return True


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_driver(personality: str, model: PPO, n_episodes: int = N_EVAL_EPISODES) -> dict:
    """
    Run n_episodes deterministically and return averaged stats.
    Uses rgb_array so no display window is needed.
    """
    rewards, steers, step_counts = [], [], []

    for _ in range(n_episodes):
        env = F1DriverEnv(personality=personality, render_mode="rgb_array")
        obs, _ = env.reset()
        total_reward, ep_steers, steps = 0.0, [], 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            ep_steers.append(float(abs(action[0])))
            steps += 1
            done = terminated or truncated

        env.close()
        rewards.append(total_reward)
        steers.append(float(np.mean(ep_steers)))
        step_counts.append(steps)

    return {
        "avg_reward":   round(float(np.mean(rewards)), 2),
        "best_reward":  round(float(np.max(rewards)), 2),
        "worst_reward": round(float(np.min(rewards)), 2),
        "avg_steps":    round(float(np.mean(step_counts)), 1),
        "avg_steering": round(float(np.mean(steers)), 4),
    }


# ── Single driver training ────────────────────────────────────────────────────

def train_driver(personality: str, checkpoints: list) -> list:
    """
    Train one driver through all checkpoints.
    Returns a list of result dicts (one per checkpoint).
    """
    cfg = F1DriverEnv.PERSONALITIES[personality]

    print(f"\n{'='*60}")
    print(f"  {cfg['emoji']}  #{cfg['number']}  {personality.upper().replace('_', ' ')}")
    print(f"  {cfg['description']}")
    print(f"{'='*60}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    vec_env  = make_vec_env(
        lambda: F1DriverEnv(personality=personality),
        n_envs=N_ENVS,
    )
    callback = RewardLogger()

    model = PPO(
        "CnnPolicy",
        vec_env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        tensorboard_log=f"./logs/{personality}",
    )

    results_rows = []
    steps_so_far = 0

    for checkpoint in checkpoints:
        steps_this_round = checkpoint - steps_so_far
        t0 = time.time()

        print(f"\n  🚦 Training to {checkpoint:,} steps...", flush=True)

        model.learn(
            total_timesteps=steps_this_round,
            callback=callback,
            reset_num_timesteps=False,
            tb_log_name=personality,
        )
        steps_so_far = checkpoint
        elapsed = time.time() - t0

        # Save model checkpoint
        model_path = f"models/{personality}_{checkpoint}"
        model.save(model_path)

        # Evaluate
        print(f"  📊 Evaluating @ {checkpoint:,} steps...", flush=True)
        stats = evaluate_driver(personality, model)

        row = {
            "driver":       personality,
            "checkpoint":   checkpoint,
            "elapsed_mins": round(elapsed / 60, 1),
            **stats,
        }
        results_rows.append(row)

        # Terminal log
        print(
            f"  ✅ {checkpoint:>9,} steps | "
            f"avg: {stats['avg_reward']:>8.1f} | "
            f"best: {stats['best_reward']:>8.1f} | "
            f"steering: {stats['avg_steering']:.4f} | "
            f"⏱  {elapsed:.0f}s"
        )

    vec_env.close()

    # Save per-driver CSV
    df = pd.DataFrame(results_rows)
    csv_path = f"results/{personality}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  💾 Results saved → {csv_path}")

    return results_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Box Box Bot — championship training")
    parser.add_argument(
        "--driver",
        type=str,
        default=None,
        choices=DRIVERS,
        help="Train a single driver instead of all four.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test run — 5k steps only, to verify the pipeline.",
    )
    args = parser.parse_args()

    drivers_to_train = [args.driver] if args.driver else DRIVERS
    checkpoints      = [5_000] if args.test else CHECKPOINTS

    if args.test:
        print("\n⚠️  TEST MODE — 5,000 steps only. Not real training.")

    print("\n🏎️  BOX BOX BOT — Championship Training Session")
    print("   Four drivers. One track. Unlimited drama.\n")

    all_results = []
    total_start = time.time()

    for driver in drivers_to_train:
        rows = train_driver(driver, checkpoints)
        all_results.extend(rows)

    # Final championship standings
    if not args.test and len(drivers_to_train) > 1:
        print(f"\n\n{'='*60}")
        print("  🏆  FINAL CHAMPIONSHIP STANDINGS")
        print(f"{'='*60}\n")

        df_all = pd.DataFrame(all_results)
        final = (
            df_all[df_all["checkpoint"] == checkpoints[-1]]
            .sort_values("avg_reward", ascending=False)
            .reset_index(drop=True)
        )
        final.index += 1
        final.index.name = "POS"

        print(
            final[["driver", "avg_reward", "best_reward", "avg_steering"]]
            .to_string()
        )

    total_mins = (time.time() - total_start) / 60
    print(f"\n  ⏱  Total time: {total_mins:.0f} minutes")
    print("  Next: python agents/evaluate.py\n")


if __name__ == "__main__":
    main()
