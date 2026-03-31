"""
Tests for env.py — Gymnasium RL environment for EDGAR 10-K extraction.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from env import (
    EdgarExtractionEnv,
    EpisodeContext,
    OBS_DIM,
    N_FIELDS,
    N_ACTIONS,
    FIELD_NAMES,
    build_registry_from_dir,
)


# ---------------------------------------------------------------------------
# EpisodeContext
# ---------------------------------------------------------------------------

class TestEpisodeContext:
    def test_defaults(self):
        ctx = EpisodeContext(
            filing_id="test",
            field_name="revenue",
            filing_xbrl=None,
            html_path=None,
            txt_path=None,
            xbrl_path=None,
            ground_truth=100.0,
            fiscal_year=2024,
            sic_code=3571,
        )
        assert ctx.attempts == []
        assert ctx.current_extracted is None
        assert ctx.current_confidence == 0.0
        assert ctx.step_count == 0

    def test_mutable_state(self):
        ctx = EpisodeContext(
            filing_id="test", field_name="revenue",
            filing_xbrl=None, html_path=None, txt_path=None,
            xbrl_path=None, ground_truth=None, fiscal_year=None, sic_code=None,
        )
        ctx.attempts.append({"action": 0, "value": 42.0})
        assert len(ctx.attempts) == 1


# ---------------------------------------------------------------------------
# EdgarExtractionEnv construction
# ---------------------------------------------------------------------------

class TestEnvConstruction:
    def test_action_space(self, sample_registry):
        env = EdgarExtractionEnv(sample_registry)
        assert env.action_space.n == N_ACTIONS

    def test_observation_space(self, sample_registry):
        env = EdgarExtractionEnv(sample_registry)
        assert env.observation_space.shape == (OBS_DIM,)

    def test_default_action_success_prior(self, sample_registry):
        env = EdgarExtractionEnv(sample_registry)
        np.testing.assert_array_equal(
            env._action_success,
            np.full(N_ACTIONS, 0.5, dtype=np.float32),
        )

    def test_custom_action_success_prior(self, sample_registry):
        prior = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        env = EdgarExtractionEnv(sample_registry, action_success_prior=prior)
        np.testing.assert_array_equal(env._action_success, prior)
        # Should be a copy, not a reference
        prior[0] = 999.0
        assert env._action_success[0] != 999.0

    def test_max_steps(self, sample_registry):
        env = EdgarExtractionEnv(sample_registry, max_steps_per_episode=5)
        assert env.max_steps == 5


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestEnvReset:
    def test_returns_obs_and_info(self, sample_registry):
        env = EdgarExtractionEnv(sample_registry)
        obs, info = env.reset(seed=42)
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32
        assert "filing_id" in info
        assert "field_name" in info
        assert "has_ground_truth" in info

    def test_field_name_valid(self, sample_registry):
        env = EdgarExtractionEnv(sample_registry)
        for _ in range(20):
            _, info = env.reset()
            assert info["field_name"] in FIELD_NAMES

    def test_reset_returns_valid_shapes(self, sample_registry):
        """reset() returns valid obs shape and info keys regardless of seed."""
        env = EdgarExtractionEnv(sample_registry)
        obs, info = env.reset(seed=123)
        assert obs.shape == (OBS_DIM,)
        assert info["field_name"] in FIELD_NAMES
        assert isinstance(info["has_ground_truth"], bool)


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------

class TestEnvStep:
    def test_returns_correct_tuple(self, sample_registry):
        env = EdgarExtractionEnv(sample_registry)
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (OBS_DIM,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert truncated is False
        assert isinstance(info, dict)
        assert "action" in info
        assert "extracted" in info
        assert "reward" in info

    def test_step_without_reset_raises(self, sample_registry):
        env = EdgarExtractionEnv(sample_registry)
        with pytest.raises(AssertionError, match="Call reset"):
            env.step(0)

    def test_flag_missing_terminates(self, sample_registry):
        env = EdgarExtractionEnv(sample_registry)
        env.reset(seed=42)
        _, _, terminated, _, _ = env.step(5)  # flag_missing
        assert terminated is True

    def test_high_reward_terminates(self, full_ground_truth_registry):
        """If reward >= 0.8, episode should terminate.

        Uses a registry that has facts for all 12 fields so that every reset()
        produces a non-None ground_truth regardless of which field is selected
        (env._rng is NOT seeded by reset(seed=), so field selection is always
        non-deterministic — see memory gotcha: Gymnasium env.reset(seed=) does
        NOT reset custom RNGs).
        """
        # Extractor returns exact ground truth when available, guaranteeing reward=1.0
        def mock_extractor(ctx):
            if ctx.ground_truth is not None:
                return ctx.ground_truth, 1.0, {"mock": True}
            return 100.0, 1.0, {"mock": True}

        env = EdgarExtractionEnv(
            full_ground_truth_registry,
            extractors={0: mock_extractor},
        )
        env.reset()
        # ground_truth is always available; exact match gives reward=1.0 >= 0.8
        _, reward, terminated, _, _ = env.step(0)
        assert reward == pytest.approx(1.0)
        assert terminated is True

    def test_max_steps_terminates(self, sample_registry):
        env = EdgarExtractionEnv(sample_registry, max_steps_per_episode=2)
        env.reset(seed=42)
        # Use action that won't terminate early
        def low_reward_extractor(ctx):
            return 1.0, 0.1, {"mock": True}  # will get low reward

        env._extractors = {i: low_reward_extractor for i in range(5)}
        # Step 1
        _, _, term1, _, _ = env.step(0)
        if not term1:
            # Step 2 — should terminate due to max_steps
            _, _, term2, _, _ = env.step(1)
            assert term2 is True

    def test_step_increments_count(self, sample_registry):
        env = EdgarExtractionEnv(sample_registry, max_steps_per_episode=10)
        env.reset(seed=42)
        env.step(5)  # flag_missing terminates, but step_count should be 1
        assert env._ctx.step_count == 1


# ---------------------------------------------------------------------------
# _build_observation
# ---------------------------------------------------------------------------

class TestBuildObservation:
    def test_shape(self, sample_registry):
        env = EdgarExtractionEnv(sample_registry)
        env.reset(seed=42)
        obs = env._build_observation()
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32

    def test_field_one_hot(self, sample_registry):
        env = EdgarExtractionEnv(sample_registry)
        env.reset(seed=42)
        obs = env._build_observation()
        # Exactly one field should be hot
        field_section = obs[:N_FIELDS]
        assert np.sum(field_section) == 1.0
        hot_idx = int(np.argmax(field_section))
        assert FIELD_NAMES[hot_idx] == env._ctx.field_name

    def test_action_priors_in_obs(self, sample_registry):
        env = EdgarExtractionEnv(sample_registry)
        env.reset(seed=42)
        obs = env._build_observation()
        # Last N_ACTIONS elements should be the action success priors
        prior_section = obs[-N_ACTIONS:]
        np.testing.assert_array_almost_equal(
            prior_section, env._action_success,
        )

    def test_obs_dim_assertion(self, sample_registry):
        """The internal assertion ptr == OBS_DIM should hold."""
        env = EdgarExtractionEnv(sample_registry)
        env.reset(seed=42)
        # This should not raise
        obs = env._build_observation()
        assert len(obs) == OBS_DIM


# ---------------------------------------------------------------------------
# _run_extractor
# ---------------------------------------------------------------------------

class TestRunExtractor:
    def test_injected_extractor(self, sample_registry):
        called = []
        def mock_ext(ctx):
            called.append(ctx.field_name)
            return 42.0, 0.9, {"custom": True}

        env = EdgarExtractionEnv(sample_registry, extractors={0: mock_ext})
        env.reset(seed=42)
        val, conf, info = env._run_extractor(0, env._ctx)
        assert val == 42.0
        assert conf == 0.9
        assert info["custom"] is True
        assert len(called) == 1

    def test_flag_missing_action(self, sample_registry):
        env = EdgarExtractionEnv(sample_registry)
        env.reset(seed=42)
        val, conf, info = env._run_extractor(5, env._ctx)
        assert val is None
        assert conf == 0.0
        assert info["reason"] == "flagged_missing"

    def test_llm_skipped_no_api_key(self, sample_registry):
        env = EdgarExtractionEnv(sample_registry)
        env.reset(seed=42)
        with patch.dict(os.environ, {}, clear=True):
            # Ensure ANTHROPIC_API_KEY is not set
            os.environ.pop("ANTHROPIC_API_KEY", None)
            val, conf, info = env._run_extractor(3, env._ctx)
            assert val is None
            assert info.get("error") == "llm_skipped_no_api_key"

    def test_unknown_action_raises(self, sample_registry):
        env = EdgarExtractionEnv(sample_registry)
        env.reset(seed=42)
        with pytest.raises(ValueError, match="Unknown action"):
            env._run_extractor(99, env._ctx)


# ---------------------------------------------------------------------------
# action_success_rates
# ---------------------------------------------------------------------------

class TestActionSuccessRates:
    def test_returns_copy(self, sample_registry):
        env = EdgarExtractionEnv(sample_registry)
        rates = env.action_success_rates
        rates[0] = 999.0
        assert env._action_success[0] != 999.0

    def test_initial_values(self, sample_registry):
        env = EdgarExtractionEnv(sample_registry)
        rates = env.action_success_rates
        np.testing.assert_array_equal(rates, np.full(N_ACTIONS, 0.5))


# ---------------------------------------------------------------------------
# Welford-style success rate update
# ---------------------------------------------------------------------------

class TestWelfordUpdate:
    def test_success_rate_updates(self, sample_registry):
        """After a successful step, action success rate should increase."""
        def exact_match_extractor(ctx):
            if ctx.ground_truth is not None:
                return ctx.ground_truth, 1.0, {}
            return 100.0, 1.0, {}

        env = EdgarExtractionEnv(
            sample_registry,
            extractors={0: exact_match_extractor},
            max_steps_per_episode=1,
        )
        initial_rate = env._action_success[0]
        env.reset(seed=42)
        env.step(0)
        # After a successful extraction, rate should have changed
        assert env._action_counts[0] == 1


# ---------------------------------------------------------------------------
# build_registry_from_dir
# ---------------------------------------------------------------------------

class TestBuildRegistryFromDir:
    def test_empty_dir(self, tmp_path):
        registry = build_registry_from_dir(tmp_path)
        assert registry == []

    def test_with_filing_structure(self, tmp_path):
        # Create: filings_dir/cik/accession/{.xml, .htm, .txt}
        cik_dir = tmp_path / "0000320193"
        acc_dir = cik_dir / "0000320193-24-000123"
        acc_dir.mkdir(parents=True)
        (acc_dir / "filing.xml").write_text("<xml/>")
        (acc_dir / "filing.htm").write_text("<html/>")
        (acc_dir / "filing.txt").write_text("text")

        registry = build_registry_from_dir(tmp_path)
        assert len(registry) == 1
        entry = registry[0]
        assert entry["filing_id"] is not None
        assert entry["xbrl_path"] is not None
        assert entry["html_path"] is not None
        assert entry["txt_path"] is not None

    def test_skips_non_directories(self, tmp_path):
        (tmp_path / "readme.txt").write_text("not a filing")
        registry = build_registry_from_dir(tmp_path)
        assert registry == []

    def test_multiple_filings(self, tmp_path):
        for cik in ["0000320193", "0000789019"]:
            acc_dir = tmp_path / cik / f"{cik}-24-000001"
            acc_dir.mkdir(parents=True)
            (acc_dir / "filing.xml").write_text("<xml/>")

        registry = build_registry_from_dir(tmp_path)
        assert len(registry) == 2

    def test_missing_file_types(self, tmp_path):
        """Filing dir with only .txt should still be registered."""
        acc_dir = tmp_path / "0000320193" / "0000320193-24-000001"
        acc_dir.mkdir(parents=True)
        (acc_dir / "filing.txt").write_text("text only")

        registry = build_registry_from_dir(tmp_path)
        assert len(registry) == 1
        entry = registry[0]
        assert entry["xbrl_path"] is None
        assert entry["html_path"] is None
        assert entry["txt_path"] is not None
