"""
train.py — RL training loop for the EDGAR extraction system.

Phases:
  1. Bandit phase  (default): FieldBandit or ContextBandit — fast, interpretable
  2. PPO phase     (--ppo):   stable-baselines3 PPO on the gymnasium env

Usage:
  python train.py                          # bandit, auto-load registry
  python train.py --episodes 500           # run 500 episodes
  python train.py --ppo --timesteps 50000  # PPO phase
  python train.py --eval                   # evaluation only, no training
  python train.py --registry path/to/registry.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FILINGS_DIR   = Path(__file__).parent / "filings"
REGISTRY_PATH = FILINGS_DIR / "registry.json"
BANDIT_PATH   = Path(__file__).parent / "checkpoints" / "bandit.pkl"
PPO_PATH      = Path(__file__).parent / "checkpoints" / "ppo_model"
DB_PATH       = Path(__file__).parent / "edgar_rl.duckdb"

ACTION_NAMES = ["xbrl_direct", "html_table", "regex_txt", "llm_claude", "derived", "flag_missing"]


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------

def load_registry(path: Path = REGISTRY_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Registry not found at {path}.\n"
            f"Run: python download_filings.py --sample"
        )
    with open(path) as f:
        data = json.load(f)
    logger.info("Loaded registry: %d filings", len(data))
    return data


# ---------------------------------------------------------------------------
# Bandit training loop
# ---------------------------------------------------------------------------

def seed_db_from_registry(registry: list[dict], db_conn) -> None:
    """Upsert all registry entries into the filings table."""
    from db import upsert_filing
    for meta in registry:
        upsert_filing(db_conn, meta)
    logger.info("Seeded DB with %d filings", len(registry))


def run_bandit_training(
    registry: list[dict],
    n_episodes: int = 1000,
    bandit_type: str = "field",       # "field" | "context"
    max_steps: int = 3,
    log_every: int = 100,
    checkpoint_every: int = 100,
    db_conn=None,
) -> None:
    from bandit import FieldBandit, ContextBandit, save_bandit, load_bandit
    from env import EdgarExtractionEnv

    # Load or init bandit
    if BANDIT_PATH.exists():
        logger.info("Resuming bandit from %s", BANDIT_PATH)
        bandit = load_bandit(BANDIT_PATH)
    else:
        bandit = ContextBandit() if bandit_type == "context" else FieldBandit()
        logger.info("Initialized new %s bandit", bandit_type)

    env = EdgarExtractionEnv(registry, max_steps_per_episode=max_steps)

    if db_conn:
        seed_db_from_registry(registry, db_conn)

    # Metrics
    episode_rewards: list[float] = []
    field_rewards: dict[str, list[float]] = {f: [] for f in env.observation_space.shape}
    action_counts = np.zeros(len(ACTION_NAMES), dtype=int)
    action_rewards = np.zeros(len(ACTION_NAMES), dtype=float)

    with tqdm(total=n_episodes, desc="Training", unit="ep") as pbar:
        for ep in range(n_episodes):
            obs, info = env.reset()
            field_name = info["field_name"]
            filing_id  = info["filing_id"]
            sic_code   = None

            # Look up SIC from registry
            meta = next((r for r in registry if r.get("filing_id") == filing_id), {})
            sic_code = meta.get("sic_code")

            episode_id = f"{filing_id}:{field_name}:{int(time.time()*1000)}"
            best_reward = -np.inf
            best_value = None
            best_confidence = 0.0
            best_action = None
            tried_actions: set[int] = set()
            step = 0

            # Log episode start to DB
            if db_conn:
                from db import start_episode
                start_episode(db_conn, episode_id, filing_id, field_name, info.get("ground_truth"))

            terminated = False
            while not terminated:
                # Bandit selects action — exclude already-tried actions this episode
                if bandit_type == "context":
                    action = bandit.select(field_name, sic_code=sic_code, exclude=tried_actions)
                else:
                    action = bandit.select(field_name, exclude=tried_actions)
                tried_actions.add(action)

                obs, reward, terminated, truncated, step_info = env.step(action)
                best_reward = max(best_reward, reward)

                # Update bandit
                if bandit_type == "context":
                    bandit.update(field_name, action, reward, sic_code=sic_code)
                else:
                    bandit.update(field_name, action, reward)

                # Track metrics
                action_counts[action] += 1
                action_rewards[action] += reward

                if step_info.get("extracted") is not None and reward > best_confidence:
                    best_value = step_info["extracted"]
                    best_confidence = step_info.get("confidence", 0.0)
                    best_action = action

                # Log step to DB
                if db_conn:
                    from db import record_step
                    record_step(
                        db_conn,
                        episode_id=episode_id,
                        step_index=step,
                        action=action,
                        action_name=ACTION_NAMES[action],
                        extracted_value=step_info.get("extracted"),
                        confidence=step_info.get("confidence", 0.0),
                        reward=reward,
                        latency_ms=step_info.get("latency", 0.0) * 1000,
                        xbrl_tag=step_info.get("xbrl_tag"),
                        error=step_info.get("error"),
                    )
                step += 1

            episode_rewards.append(best_reward if best_reward > -np.inf else -0.5)

            # Close episode in DB
            if db_conn:
                from db import close_episode, upsert_extracted_value
                close_episode(
                    db_conn, episode_id,
                    n_steps=step,
                    total_reward=best_reward,
                    best_action=best_action,
                    best_value=best_value,
                    best_confidence=best_confidence,
                )
                if best_value is not None:
                    upsert_extracted_value(
                        db_conn, filing_id, field_name,
                        value=best_value, unit="USD",
                        confidence=best_confidence,
                        source_action=best_action,
                        source_name=ACTION_NAMES[best_action] if best_action is not None else "?",
                    )

            pbar.update(1)

            # Periodic logging
            if (ep + 1) % log_every == 0:
                recent = episode_rewards[-log_every:]
                mean_r = np.mean(recent)
                logger.info(
                    "ep=%d  mean_reward(last %d)=%.3f  action_dist=%s",
                    ep + 1, log_every, mean_r,
                    dict(zip(ACTION_NAMES, action_counts.tolist())),
                )

            # Checkpoint
            if (ep + 1) % checkpoint_every == 0:
                save_bandit(bandit, BANDIT_PATH)
                _write_metrics_snapshot(bandit, action_counts, action_rewards, episode_rewards)
                logger.info("Checkpoint saved → %s", BANDIT_PATH)

    # Final checkpoint
    save_bandit(bandit, BANDIT_PATH)
    _write_metrics_snapshot(bandit, action_counts, action_rewards, episode_rewards)
    _print_bandit_summary(bandit, action_counts, action_rewards, episode_rewards)


METRICS_PATH = Path(__file__).parent / "checkpoints" / "metrics.json"


def _write_metrics_snapshot(bandit, action_counts, action_rewards, episode_rewards) -> None:
    """Write a JSON snapshot of training metrics for the UI to read.

    The UI reads this file instead of the DuckDB database, so training and
    the Streamlit dashboard can run simultaneously without lock conflicts.
    """
    window = 100
    history = [
        {"episode": i + 1, "reward": float(r)}
        for i, r in enumerate(episode_rewards)
    ]
    rolling = [
        {
            "episode": h["episode"],
            "rolling_avg": float(np.mean(episode_rewards[max(0, i - window + 1): i + 1])),
        }
        for i, h in enumerate(history)
    ]

    summary = bandit.summary()
    field_stats = {}
    for key, entry in summary.items():
        if not isinstance(entry, dict) or "best_action" not in entry:
            continue
        ba = entry["best_action"]
        q_values = entry.get("Q", [])
        field_stats[key] = {
            "best_action": ACTION_NAMES[ba] if ba < len(ACTION_NAMES) else "?",
            "best_q": float(q_values[ba]) if ba < len(q_values) else 0.0,
            "Q": [float(q) for q in q_values],
            "N": [int(n) for n in entry.get("N", [])],
        }

    payload = {
        "total_episodes":  len(episode_rewards),
        "mean_reward":     float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "final_100_avg":   float(np.mean(episode_rewards[-100:])) if episode_rewards else 0.0,
        "action_usage":    {ACTION_NAMES[i]: int(action_counts[i]) for i in range(len(ACTION_NAMES))},
        "action_rewards":  {ACTION_NAMES[i]: float(action_rewards[i] / max(action_counts[i], 1))
                            for i in range(len(ACTION_NAMES))},
        "rolling_history": rolling,
        "field_stats":     field_stats,
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(payload, f)


def _print_bandit_summary(bandit, action_counts, action_rewards, episode_rewards) -> None:
    print("\n" + "=" * 65)
    print("TRAINING SUMMARY")
    print("=" * 65)
    print(f"Total episodes : {len(episode_rewards)}")
    print(f"Mean reward    : {np.mean(episode_rewards):+.4f}")
    print(f"Final 100 avg  : {np.mean(episode_rewards[-100:]):+.4f}")

    print("\nPer-action usage and mean reward:")
    print(f"  {'Action':<18} {'Uses':>6} {'MeanReward':>12}")
    print("  " + "-" * 38)
    for a, name in enumerate(ACTION_NAMES):
        n = action_counts[a]
        mean_r = action_rewards[a] / n if n > 0 else 0.0
        print(f"  {name:<18} {n:>6} {mean_r:>+12.4f}")

    try:
        summary = bandit.summary()
        print("\nBest action per field (bandit Q-values):")
        print(f"  {'Field':<25} {'BestAction':<18} {'Q':>8}")
        print("  " + "-" * 54)
        for field_name in sorted(summary, key=lambda k: summary[k].get("best_action", 0)):
            entry = summary[field_name]
            if not (isinstance(entry, dict) and "best_action" in entry):
                continue
            ba = entry["best_action"]
            # ContextBandit stores Q as a list; FieldBandit stores under "stats"
            if "Q" in entry:
                q = entry["Q"][ba] if ba < len(entry["Q"]) else 0.0
            else:
                q = entry.get("stats", {}).get(ba, {}).get("mean_reward", 0.0)
            print(f"  {field_name:<35} {ACTION_NAMES[ba]:<18} {q:>+8.4f}")
    except Exception:
        pass
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# PPO training (Phase 3 upgrade)
# ---------------------------------------------------------------------------

def run_ppo_training(
    registry: list[dict],
    total_timesteps: int = 100_000,
    max_steps: int = 3,
) -> None:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError:
        raise ImportError("stable-baselines3 required: pip install stable-baselines3")

    from env import EdgarExtractionEnv

    def make_env():
        return EdgarExtractionEnv(registry, max_steps_per_episode=max_steps)

    vec_env = make_vec_env(make_env, n_envs=4)

    if PPO_PATH.with_suffix(".zip").exists():
        logger.info("Loading existing PPO model from %s", PPO_PATH)
        model = PPO.load(str(PPO_PATH), env=vec_env)
    else:
        logger.info("Initializing new PPO model")
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.01,       # encourage exploration
            tensorboard_log="./tb_logs/",
        )

    logger.info("Training PPO for %d timesteps...", total_timesteps)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    PPO_PATH.parent.mkdir(exist_ok=True)
    model.save(str(PPO_PATH))
    logger.info("PPO model saved → %s.zip", PPO_PATH)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_eval(
    registry: list[dict],
    n_episodes: int = 200,
    use_ppo: bool = False,
    max_steps: int = 3,
) -> None:
    from env import EdgarExtractionEnv, FIELD_NAMES

    env = EdgarExtractionEnv(registry, max_steps_per_episode=max_steps)

    if use_ppo:
        from stable_baselines3 import PPO
        model = PPO.load(str(PPO_PATH))
        policy_fn = lambda obs: int(model.predict(obs, deterministic=True)[0])
        label = "PPO"
    elif BANDIT_PATH.exists():
        from bandit import load_bandit
        bandit = load_bandit(BANDIT_PATH)
        def policy_fn(obs):
            # Recover field name from one-hot prefix of obs
            field_idx = int(np.argmax(obs[:len(FIELD_NAMES)]))
            field_name = FIELD_NAMES[field_idx]
            if hasattr(bandit, "select"):
                return bandit.select(field_name)
            return 0
        label = "Bandit"
    else:
        policy_fn = lambda obs: 0   # always XBRL
        label = "XBRL-only baseline"

    rewards_by_field: dict[str, list[float]] = {f: [] for f in FIELD_NAMES}
    all_rewards: list[float] = []

    for _ in tqdm(range(n_episodes), desc=f"Eval ({label})", unit="ep"):
        obs, info = env.reset()
        field_name = info["field_name"]
        total_r = 0.0
        terminated = False
        while not terminated:
            action = policy_fn(obs)
            obs, reward, terminated, _, _ = env.step(action)
            total_r += reward
        all_rewards.append(total_r)
        rewards_by_field[field_name].append(total_r)

    print(f"\n{'='*55}")
    print(f"EVALUATION — {label}  ({n_episodes} episodes)")
    print(f"{'='*55}")
    print(f"Overall mean reward: {np.mean(all_rewards):+.4f}")
    print(f"\n{'Field':<25} {'N':>4} {'MeanReward':>12} {'≥0.8 rate':>10}")
    print("-" * 55)
    for field_name in FIELD_NAMES:
        rs = rewards_by_field[field_name]
        if not rs:
            continue
        high_rate = np.mean([r >= 0.8 for r in rs])
        print(f"{field_name:<25} {len(rs):>4} {np.mean(rs):>+12.4f} {high_rate:>9.1%}")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDGAR RL training loop")
    parser.add_argument("--episodes",   type=int,   default=1000)
    parser.add_argument("--bandit",     type=str,   default="field",
                        choices=["field", "context"],
                        help="Bandit type (default: field)")
    parser.add_argument("--ppo",        action="store_true", help="Use PPO instead of bandit")
    parser.add_argument("--timesteps",  type=int,   default=100_000,
                        help="PPO timesteps (only with --ppo)")
    parser.add_argument("--eval",       action="store_true", help="Eval only, no training")
    parser.add_argument("--max-steps",  type=int,   default=3)
    parser.add_argument("--log-every",  type=int,   default=100)
    parser.add_argument("--no-db",      action="store_true", help="Skip DB logging")
    parser.add_argument("--registry",   type=str,   default=str(REGISTRY_PATH))
    args = parser.parse_args()

    registry = load_registry(Path(args.registry))

    db_conn = None
    if not args.no_db:
        try:
            from db import get_connection
            db_conn = get_connection(DB_PATH)
            logger.info("DB logging enabled → %s", DB_PATH)
        except Exception as e:
            logger.warning("DB locked or unavailable — continuing without logging (%s)", e)

    try:
        if args.eval:
            run_eval(registry, n_episodes=min(args.episodes, 500), use_ppo=args.ppo)
        elif args.ppo:
            run_ppo_training(registry, total_timesteps=args.timesteps, max_steps=args.max_steps)
        else:
            run_bandit_training(
                registry,
                n_episodes=args.episodes,
                bandit_type=args.bandit,
                max_steps=args.max_steps,
                log_every=args.log_every,
                db_conn=db_conn,
            )
    finally:
        if db_conn:
            db_conn.close()
