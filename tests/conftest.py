"""Shared pytest fixtures for the mpc-guided test suite.

Markers (declared in pyproject.toml):
    env      — test creates a ManiSkill env (slow, several seconds setup).
    dataset  — test reads files from a built dataset under data/; auto-skipped
               when the dataset hasn't been built.

Run subsets:
    uv run pytest -m "not env"     # fast unit tests only
    uv run pytest -m "dataset"     # data-dependent integration tests
"""
