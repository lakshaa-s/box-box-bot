"""
evaluate.py
===========
Load trained driver models and record replay GIFs + telemetry stats.

Run from the project root after training:
    python agents/evaluate.py

To evaluate a single driver at a specific checkpoint:
    python agents/evaluate.py --driver divebomber --checkpoint 1000000

To evaluate all drivers at their best checkpoint (1M steps):
    python agents/evaluate.py
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import imageio
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(__file__))
from personalities import F1DriverEnv

# ── Config ────────────────────────────────────────────────────────────────────

DRIVERS = [
    "divebomber",
    "tyre_whisperer",
    "sunday_specialist",
    "smooth_operator",
]

DEFAULT_CHECKPOINT = 1_000_000
MAX_STEPS          = 800   # max steps per recorded episode
GIF_FPS            = 30


# ── Single driver evaluation ──────────────────────────────────────────────────

def record_driver(personality: str, checkpoint: int) -> dict:
    """
    Load a saved model, run one deterministic episode, save a GIF,
    and return telemetry stats.
    """
    cfg        = F1DriverEnv.PERSONALITIES[personality]
    model_path = f"models/{personality}_{checkpoint}"

    # Check model exists
    if not os.path.exists(f"{model_path}.zip"):
        print(f"  ⚠️  No model found at {model_path}.zip — skipping.")
        return None

    print(f"\n  {cfg['emoji']}  #{cfg['number']}  {personality.replace('_', ' ').title()}")
    print(f"  Loading {model_path}.zip ...", flush=True)

    model = PPO.load(model_path)
    env   = F1DriverEnv(personality=personality, render_mode="rgb_array")
    obs, _ = env.reset()

    frames       = []
    total_reward = 0.0
    steers       = []
    throttles    = []
    brakes       = []
    done         = False
    steps        = 0

    while not done and steps < MAX_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        steers.append(float(abs(action[0])))
        throttles.append(float(action[1]))
        brakes.append(float(action[2]))
        frames.append(env.render())

        steps += 1
        done = terminated or truncated

    env.close()

    # Save GIF
    os.makedirs("replays", exist_ok=True)
    gif_path = f"replays/{personality}_{checkpoint}.gif"
    imageio.mimsave(gif_path, frames, fps=GIF_FPS)

    stats = {
        "driver":        personality,
        "number":        cfg["number"],
        "checkpoint":    checkpoint,
        "total_reward":  round(total_reward, 2),
        "steps":         steps,
        "avg_steering":  round(float(np.mean(steers)), 4),
        "avg_throttle":  round(float(np.mean(throttles)), 4),
        "avg_brake":     round(float(np.mean(brakes)), 4),
        "smoothness":    round(1.0 - float(np.mean(np.diff(steers) if len(steers) > 1 else [0])), 4),
        "gif":           gif_path,
    }

    print(f"  ✅ reward: {stats['total_reward']:>8.2f} | "
          f"steps: {steps} | "
          f"steering: {stats['avg_steering']:.4f} | "
          f"smoothness: {stats['smoothness']:.4f}")
    print(f"  📼 GIF saved → {gif_path}")

    return stats


# ── Standings table ───────────────────────────────────────────────────────────

def print_standings(results: list) -> None:
    """Print a formatted standings table from a list of result dicts."""
    valid = [r for r in results if r is not None]
    if not valid:
        print("\n  No results to display.")
        return

    df = (
        pd.DataFrame(valid)
        .drop(columns=["gif"])
        .sort_values("total_reward", ascending=False)
        .reset_index(drop=True)
    )
    df.index += 1
    df.index.name = "POS"

    print(f"\n\n{'='*60}")
    print("  🏆  POST-RACE STANDINGS")
    print(f"{'='*60}\n")
    print(df[["driver", "number", "total_reward", "steps", "avg_steering", "smoothness"]].to_string())
    print("\n  (smoothness: higher = more consistent inputs)")

    # Save combined results CSV
    os.makedirs("results", exist_ok=True)
    out_path = "results/evaluation_summary.csv"
    df.to_csv(out_path, index=True)
    print(f"\n  💾 Summary saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Box Box Bot — driver evaluation")
    parser.add_argument(
        "--driver",
        type=str,
        default=None,
        choices=DRIVERS,
        help="Evaluate a single driver.",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=DEFAULT_CHECKPOINT,
        help=f"Model checkpoint to load (default: {DEFAULT_CHECKPOINT:,}).",
    )
    args = parser.parse_args()

    drivers_to_eval = [args.driver] if args.driver else DRIVERS

    print("\n🏎️  BOX BOX BOT — Post-Race Evaluation")
    print(f"   Checkpoint: {args.checkpoint:,} steps\n")

    results = []
    for driver in drivers_to_eval:
        stats = record_driver(driver, args.checkpoint)
        results.append(stats)

    print_standings(results)
    print("\n  Done. Open replays/ to watch the GIFs.\n")


if __name__ == "__main__":
    main()
