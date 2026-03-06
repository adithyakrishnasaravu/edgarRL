"""
env.py — Gymnasium environment for the EDGAR 10-K RL extraction task.

One episode = one (filing, field) pair.
The agent selects which extractor to try; reward is computed vs XBRL ground truth.

Action space (Discrete 6):
  0  xbrl_direct   — XBRL tag lookup
  1  html_table    — HTML income statement table parser
  2  regex_txt     — Regex on raw TXT filing
  3  llm_claude    — Claude LLM fallback
  4  derived       — Arithmetic derivation
  5  flag_missing  — Give up / flag low confidence

Observation space (Box, float32, shape=(OBS_DIM,)):
  See _build_observation() for full feature list.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from xbrl_extractor import (
    FilingXBRL,
    ExtractionResult,
    compute_reward,
    extract_field,
    load_xbrl_filing,
    load_xbrl_from_company_facts,
    load_xbrl_from_facts_file,
    FIELDS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIELD_CONFIG_PATH = Path(__file__).parent / "field_config.json"
with open(FIELD_CONFIG_PATH) as f:
    _CFG = json.load(f)

FIELD_NAMES: list[str] = _CFG["field_order"]
N_FIELDS = len(FIELD_NAMES)
N_ACTIONS = len(_CFG["actions"])

# Observation vector dimension — must match _build_observation()
OBS_DIM = (
    N_FIELDS        # which field we are extracting (one-hot)
    + 5             # document format availability flags [xbrl, html, txt, json_facts, derived_possible]
    + 6             # last attempt result: [tried, success, confidence, log_abs_value, sign, rel_err]
    + 3             # filing metadata: [has_xbrl_gt, fiscal_year_norm, sic_norm]
    + N_ACTIONS     # per-action prior success rate (running average)
)


# ---------------------------------------------------------------------------
# Episode data container
# ---------------------------------------------------------------------------

@dataclass
class EpisodeContext:
    filing_id: str                     # accession or CIK+year string
    field_name: str
    filing_xbrl: Optional[FilingXBRL]
    html_path: Optional[Path]
    txt_path: Optional[Path]
    xbrl_path: Optional[Path]
    ground_truth: Optional[float]       # from XBRL oracle; None if unavailable
    fiscal_year: Optional[int]
    sic_code: Optional[int]
    # Mutable state
    attempts: list[dict] = field(default_factory=list)
    current_extracted: Optional[float] = None
    current_confidence: float = 0.0
    step_count: int = 0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class EdgarExtractionEnv(gym.Env):
    """
    Gymnasium environment for learning extraction strategies over SEC 10-K filings.

    Parameters
    ----------
    data_registry : list[dict]
        Each entry describes one filing:
          {
            "filing_id": str,
            "xbrl_path": str | None,
            "html_path": str | None,
            "txt_path":  str | None,
            "fiscal_year": int | None,
            "sic_code": int | None,
          }
    max_steps_per_episode : int
        Maximum extractor attempts before forced termination.
    extractors : dict[int, callable]
        Optional override of extractor callables keyed by action index.
        Defaults to the built-in extractor registry.
    action_success_prior : np.ndarray | None
        Running per-action success rates passed in for warm-starting.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_registry: list[dict],
        max_steps_per_episode: int = 3,
        extractors: Optional[dict] = None,
        action_success_prior: Optional[np.ndarray] = None,
    ):
        super().__init__()

        self.data_registry = data_registry
        self.max_steps = max_steps_per_episode

        # Action space: discrete over 6 extractor strategies
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Observation space: bounded float32 vector
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32
        )

        # Per-action running success rate (initialized to 0.5 — optimistic)
        self._action_success: np.ndarray = (
            action_success_prior.copy()
            if action_success_prior is not None
            else np.full(N_ACTIONS, 0.5, dtype=np.float32)
        )
        self._action_counts: np.ndarray = np.zeros(N_ACTIONS, dtype=np.float32)

        # Extractor callables — lazy import to avoid circular deps
        self._extractors: dict[int, Any] = extractors or {}

        # Episode state (populated by reset())
        self._ctx: Optional[EpisodeContext] = None
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # gym.Env interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Pick a random (filing, field) pair
        filing_meta = self._rng.choice(self.data_registry)
        field_name = FIELD_NAMES[int(self._rng.integers(N_FIELDS))]

        # Load XBRL — route to the right loader based on file type
        xbrl_path = Path(filing_meta["xbrl_path"]) if filing_meta.get("xbrl_path") else None
        facts_path = Path(filing_meta["facts_json_path"]) if filing_meta.get("facts_json_path") else None
        filing_xbrl: Optional[FilingXBRL] = None

        if facts_path and facts_path.exists():
            # Preferred: company-facts JSON (faster, no arelle needed)
            filing_xbrl = load_xbrl_from_facts_file(
                facts_path,
                cik=filing_meta.get("cik", ""),
                accession=filing_meta.get("accession"),
                fiscal_year=filing_meta.get("fiscal_year"),
            )
        elif xbrl_path and xbrl_path.exists() and xbrl_path.suffix == ".xml":
            # Fallback: raw XBRL instance document via arelle
            filing_xbrl = load_xbrl_filing(xbrl_path)

        # Extract ground truth from XBRL (oracle)
        ground_truth: Optional[float] = None
        if filing_xbrl is not None:
            gt_result = extract_field(filing_xbrl, field_name)
            ground_truth = gt_result.value

        self._ctx = EpisodeContext(
            filing_id=filing_meta.get("filing_id", "unknown"),
            field_name=field_name,
            filing_xbrl=filing_xbrl,
            html_path=Path(filing_meta["html_path"]) if filing_meta.get("html_path") else None,
            txt_path=Path(filing_meta["txt_path"]) if filing_meta.get("txt_path") else None,
            xbrl_path=xbrl_path,
            ground_truth=ground_truth,
            fiscal_year=filing_meta.get("fiscal_year"),
            sic_code=filing_meta.get("sic_code"),
        )
        # Cache which formats are actually available for observation features
        self._ctx._has_facts_json = facts_path is not None and facts_path.exists()

        obs = self._build_observation()
        info = {
            "filing_id": self._ctx.filing_id,
            "field_name": field_name,
            "has_ground_truth": ground_truth is not None,
        }
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._ctx is not None, "Call reset() before step()"

        ctx = self._ctx
        ctx.step_count += 1

        # Run the selected extractor
        t0 = time.perf_counter()
        extracted_value, confidence, extractor_info = self._run_extractor(action, ctx)
        latency = time.perf_counter() - t0

        ctx.current_extracted = extracted_value
        ctx.current_confidence = confidence
        ctx.attempts.append({
            "action": action,
            "value": extracted_value,
            "confidence": confidence,
            "latency": latency,
            **extractor_info,
        })

        # Compute reward
        reward = compute_reward(
            extracted=extracted_value,
            ground_truth=ctx.ground_truth,
            action_index=action,
            latency_seconds=latency,
        )

        # Update running success rate (Welford-style)
        success = 1.0 if (extracted_value is not None and reward > 0) else 0.0
        n = self._action_counts[action] + 1
        self._action_success[action] += (success - self._action_success[action]) / n
        self._action_counts[action] = n

        # Termination conditions
        terminated = (
            action == 5                           # agent flagged as missing
            or reward >= 0.8                      # high-quality extraction found
            or ctx.step_count >= self.max_steps   # step budget exhausted
        )
        truncated = False

        obs = self._build_observation()
        info = {
            "action": action,
            "extracted": extracted_value,
            "ground_truth": ctx.ground_truth,
            "reward": reward,
            "latency": latency,
            "confidence": confidence,
            **extractor_info,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        pass  # text rendering on request only

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self) -> np.ndarray:
        ctx = self._ctx
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        ptr = 0

        # --- Field one-hot (N_FIELDS dims) ---
        if ctx is not None:
            field_idx = FIELD_NAMES.index(ctx.field_name) if ctx.field_name in FIELD_NAMES else 0
            obs[ptr + field_idx] = 1.0
        ptr += N_FIELDS

        # --- Document format flags (5 dims) ---
        if ctx is not None:
            obs[ptr + 0] = float(ctx.xbrl_path is not None and ctx.xbrl_path.exists())
            obs[ptr + 1] = float(ctx.html_path is not None and ctx.html_path.exists())
            obs[ptr + 2] = float(ctx.txt_path is not None and ctx.txt_path.exists())
            obs[ptr + 3] = float(ctx.filing_xbrl is not None)   # XBRL loaded successfully
            field_cfg = FIELDS.get(ctx.field_name, {})
            obs[ptr + 4] = float(field_cfg.get("derivable", False))
        ptr += 5

        # --- Last attempt result (6 dims) ---
        if ctx is not None and ctx.attempts:
            last = ctx.attempts[-1]
            obs[ptr + 0] = 1.0                             # tried flag
            obs[ptr + 1] = float(last["value"] is not None)
            obs[ptr + 2] = float(last.get("confidence", 0.0))
            val = last.get("value")
            if val is not None and val != 0:
                obs[ptr + 3] = float(np.clip(np.log10(abs(val)), -3, 15))
                obs[ptr + 4] = float(np.sign(val))
            gt = ctx.ground_truth
            if val is not None and gt is not None and gt != 0:
                obs[ptr + 5] = float(np.clip(abs(val - gt) / abs(gt), 0, 10))
        ptr += 6

        # --- Filing metadata (3 dims) ---
        if ctx is not None:
            obs[ptr + 0] = float(ctx.ground_truth is not None)
            if ctx.fiscal_year is not None:
                obs[ptr + 1] = float(np.clip((ctx.fiscal_year - 2010) / 15.0, 0, 1))
            if ctx.sic_code is not None:
                obs[ptr + 2] = float(np.clip(ctx.sic_code / 9999.0, 0, 1))
        ptr += 3

        # --- Per-action success priors (N_ACTIONS dims) ---
        obs[ptr: ptr + N_ACTIONS] = self._action_success
        ptr += N_ACTIONS

        assert ptr == OBS_DIM, f"OBS_DIM mismatch: ptr={ptr}, OBS_DIM={OBS_DIM}"
        return obs

    # ------------------------------------------------------------------
    # Extractor dispatch
    # ------------------------------------------------------------------

    def _run_extractor(
        self,
        action: int,
        ctx: EpisodeContext,
    ) -> tuple[Optional[float], float, dict]:
        """
        Dispatch to the appropriate extractor.
        Returns (value, confidence, info_dict).
        """
        # Allow injected extractors to override defaults
        if action in self._extractors:
            return self._extractors[action](ctx)

        # Skip LLM action if no API key — treat as unavailable (not a failure)
        if action == 3 and not os.environ.get("ANTHROPIC_API_KEY"):
            return None, 0.0, {"error": "llm_skipped_no_api_key"}

        if action == 0:
            return self._action_xbrl(ctx)
        elif action == 1:
            return self._action_html(ctx)
        elif action == 2:
            return self._action_regex(ctx)
        elif action == 3:
            return self._action_llm(ctx)
        elif action == 4:
            return self._action_derived(ctx)
        elif action == 5:
            return None, 0.0, {"reason": "flagged_missing"}
        else:
            raise ValueError(f"Unknown action: {action}")

    # --- Action 0: XBRL direct (with automatic derivation fallback) ---
    def _action_xbrl(self, ctx: EpisodeContext) -> tuple[Optional[float], float, dict]:
        if ctx.filing_xbrl is None:
            return None, 0.0, {"error": "no_xbrl"}
        result: ExtractionResult = extract_field(ctx.filing_xbrl, ctx.field_name)
        # If XBRL tag missing but field is derivable, try derivation immediately
        if result.value is None:
            field_cfg = FIELDS.get(ctx.field_name, {})
            if field_cfg.get("derivable"):
                derivation = field_cfg.get("derivation", "")
                derived = self._compute_derivation(derivation, ctx)
                if derived is not None:
                    return derived, 0.75, {"xbrl_tag": None, "derivation": derivation, "fallback": "derived"}
        return result.value, result.confidence, {"xbrl_tag": result.xbrl_tag, "error": result.error}

    # --- Action 1: HTML table ---
    def _action_html(self, ctx: EpisodeContext) -> tuple[Optional[float], float, dict]:
        if ctx.html_path is None or not ctx.html_path.exists():
            return None, 0.0, {"error": "no_html"}
        try:
            from html_extractor import extract_from_html
            value, conf = extract_from_html(ctx.html_path, ctx.field_name)
            return value, conf, {}
        except ImportError:
            return None, 0.0, {"error": "html_extractor_not_implemented"}
        except Exception as e:
            return None, 0.0, {"error": str(e)}

    # --- Action 2: Regex TXT ---
    def _action_regex(self, ctx: EpisodeContext) -> tuple[Optional[float], float, dict]:
        if ctx.txt_path is None or not ctx.txt_path.exists():
            return None, 0.0, {"error": "no_txt"}
        try:
            from regex_extractor import extract_from_txt
            value, conf = extract_from_txt(ctx.txt_path, ctx.field_name)
            return value, conf, {}
        except ImportError:
            return None, 0.0, {"error": "regex_extractor_not_implemented"}
        except Exception as e:
            return None, 0.0, {"error": str(e)}

    # --- Action 3: LLM (Claude) ---
    def _action_llm(self, ctx: EpisodeContext) -> tuple[Optional[float], float, dict]:
        try:
            from llm_extractor import extract_with_claude
            txt_path = ctx.txt_path or ctx.html_path
            if txt_path is None or not txt_path.exists():
                return None, 0.0, {"error": "no_document_for_llm"}
            value, conf = extract_with_claude(txt_path, ctx.field_name)
            return value, conf, {"model": "claude"}
        except ImportError:
            return None, 0.0, {"error": "llm_extractor_not_implemented"}
        except Exception as e:
            return None, 0.0, {"error": str(e)}

    # --- Action 4: Derived ---
    def _action_derived(self, ctx: EpisodeContext) -> tuple[Optional[float], float, dict]:
        if ctx.filing_xbrl is None:
            return None, 0.0, {"error": "no_xbrl_for_derivation"}
        field_cfg = FIELDS.get(ctx.field_name, {})
        if not field_cfg.get("derivable"):
            return None, 0.0, {"error": "field_not_derivable"}

        derivation = field_cfg.get("derivation", "")
        try:
            value = self._compute_derivation(derivation, ctx)
            if value is None:
                return None, 0.0, {"error": "derivation_inputs_missing"}
            return value, 0.7, {"derivation": derivation}
        except Exception as e:
            return None, 0.0, {"error": str(e)}

    def _compute_derivation(self, formula: str, ctx: EpisodeContext) -> Optional[float]:
        """Evaluate simple arithmetic derivation formulas."""
        # Map field aliases used in derivation strings to extracted values
        alias_map = {
            "revenue": "revenue",
            "cogs": "cogs",
            "gross_profit": "gross_profit",
            "operating_income": "operating_income",
            "operating_expenses": "total_expenses",
        }
        values: dict[str, float] = {}
        for alias, field_name in alias_map.items():
            if alias in formula:
                res = extract_field(ctx.filing_xbrl, field_name)
                if res.value is None:
                    return None
                values[alias] = res.value

        try:
            return float(eval(formula, {"__builtins__": {}}, values))  # noqa: S307
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def action_success_rates(self) -> np.ndarray:
        """Return current per-action running success rates."""
        return self._action_success.copy()


# ---------------------------------------------------------------------------
# Registry builder helper
# ---------------------------------------------------------------------------

def build_registry_from_dir(filings_dir: Path) -> list[dict]:
    """
    Scan a directory of downloaded 10-K filings and build the data_registry
    list expected by EdgarExtractionEnv.

    Expects structure:
      filings_dir/
        {cik}/
          {accession}/
            *.xml   (XBRL instance document)
            *.htm   (HTML filing)
            *.txt   (full submission text)
    """
    registry = []
    for cik_dir in sorted(filings_dir.iterdir()):
        if not cik_dir.is_dir():
            continue
        for acc_dir in sorted(cik_dir.iterdir()):
            if not acc_dir.is_dir():
                continue
            xbrl = next(acc_dir.glob("*.xml"), None) or next(acc_dir.glob("*_htm.xml"), None)
            html = next(acc_dir.glob("*.htm"), None) or next(acc_dir.glob("*.html"), None)
            txt = next(acc_dir.glob("*.txt"), None)
            registry.append({
                "filing_id": f"{cik_dir.name}/{acc_dir.name}",
                "xbrl_path": str(xbrl) if xbrl else None,
                "html_path": str(html) if html else None,
                "txt_path": str(txt) if txt else None,
                "fiscal_year": None,
                "sic_code": None,
            })
    return registry


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Build a dummy registry with no real files to verify gym plumbing
    dummy_registry = [
        {
            "filing_id": "test/0000000000",
            "xbrl_path": None,
            "html_path": None,
            "txt_path": None,
            "fiscal_year": 2023,
            "sic_code": 7372,
        }
    ]

    env = EdgarExtractionEnv(dummy_registry, max_steps_per_episode=3)
    obs, info = env.reset(seed=42)
    print(f"Observation shape : {obs.shape}  (expected {OBS_DIM})")
    print(f"Action space      : {env.action_space}")
    print(f"Episode info      : {info}")

    # Step through all actions
    for a in range(N_ACTIONS):
        obs, reward, terminated, truncated, info = env.step(a)
        print(f"  action={a}  reward={reward:+.2f}  terminated={terminated}  info={info}")
        if terminated:
            break

    print("\nEnvironment smoke test passed.")
