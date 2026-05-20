"""Committed benchmark fixtures for the agenda loop (issue #9).

Storing fixed Q/K/V tensors on disk (instead of regenerating them from
``numpy.random`` at runtime) lets reviewers verify the experiment uses a
known, hash-stable dataset rather than per-process synthetic noise. See
``qkv_fixture_512_64.npz`` and the SHA256 constant in
``real_experiment_runner``.
"""
