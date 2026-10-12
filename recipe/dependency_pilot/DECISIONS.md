# Pilot decisions — 2026-10-12

1. Base `24305ac8`, isolated managed worktree, branch `exp/dep-prep-pilot-20261012`. Master had five untracked user documents and remains untouched. No push/merge/PR.
2. Replace W&B with existing SwanLab Tracking, as requested. No second logger or worker cloud run.
3. P0 reuses the existing collector and GiGPO core with a bounded Transformers frozen actor. Distributed training stays P1; there is no fake train entry point.
4. ScienceWorld 1.2.3 is pinned and its source checked. Raw simulator completion plus full score is the success criterion; native partial score remains available. No simplifications, gold actions, hidden object state or goal progress are given to the model.
5. New ScienceWorld prompt uses task, public observation/look/inventory, action templates and four recent observation/action pairs. This is an explicit new adapter protocol, not a reproduction claim for another framework.
6. Frozen P0 uses terminal +10 once, no parse penalty and no KL. The scheme allowed inheriting a 0.1 penalty but did not require it; zero is recorded, and no upstream training configuration is changed.
7. Local GPU is RTX 3050 4GB; any smaller local turn/token/group budget is a named explicit override, separated from the 50-turn Orion profile. HF HEAD metadata failed; verified GET downloads at a pinned revision work. No model or environment process is run on Orion's login node.
8. Existing ALFWorld Python 3.10 is reused through an owned subprocess, avoiding upgrades to shared environments. AppWorld gets its own Python 3.11 setup job and legal train/dev enumeration; its semantic dependency labels remain unverified until real interaction review.
9. P1 prefix continuation is selected only after inspecting naturally reached preparation states. P2 is deferred. Unknown conditions are never inferred from a score increment alone.
