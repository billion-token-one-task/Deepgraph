# CGGR/CRPP example plugin

This directory contains historical topic-specific runners, aliases, ablations,
tests, and research notes that were previously exposed by the generic runtime.
It is disabled by default, is not production-eligible, and has no resource
authority. Enabling it must be explicit and does not bypass Agenda,
ResourceGrant, evidence, or harness-review policy.

The generic `AgentBoundary` registry must not list these modules or scripts.
The plugin manifest preserves the topic defaults for reproducibility without
making them universal defaults.

Several historical scripts contain workstation/remote assumptions. They are
archival example code until their plugin test lane passes in an isolated
environment; they must not be run on the production host.
