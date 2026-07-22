# Logs

Loose, informal running log — whatever happened, in the moment, roughly
chronological. Lower bar than findings.md: half-formed observations, dead ends,
"tried X, didn't work, here's why" all belong here.

---

## 2026-07-22

Repo scaffolded (structure, CLAUDE.md, docs, notebook skeletons). Explored the prior
QAT project at `/mnt/windows/projects/quant research` — confirmed no checkpoint was
ever saved, everything ran ephemeral in Colab. Recipe to reproduce lives in that
repo's `notebooks/15_mixed_precision_guided_targeted_qat.ipynb`.

Cloned `vllm-project/vllm` (shallow) into `vendor/vllm`. Grepped for eagle/spec_decode
files. Found the generic drafter interface (`SpecDecodeBaseProposer` in
`llm_base_proposer.py`) and, importantly, `DraftModelProposer` in `draft_model.py` —
vLLM already has a first-class "arbitrary HF checkpoint as drafter" path, separate from
the EAGLE-specific code. Confirmed via `vllm/config/speculative.py` that
`method="draft_model"` is the *default* fallback when a model path doesn't look like an
EAGLE/ngram name — so plugging in the SGT-QAT checkpoint should mostly be a config
problem, not a new-code problem. Also spotted `qwen3_eagle3.py` — an EAGLE-3 head
already implemented specifically for Qwen3, good reference for the baseline notebook.
Found `examples/features/speculative_decoding/spec_decode_offline.py` in the vllm
checkout — a working example covering exactly both methods we need (`eagle3` and
`draft_model`), with the exact `speculative_config` dict shape for each, plus vLLM's
built-in acceptance-rate metrics (`vllm:spec_decode_num_drafts` etc. via
`llm.get_metrics()`). This effectively finishes Phase 1 orientation — no more
guesswork on the drafter interface or config wiring. Wall-clock and memory metrics
aren't covered by this example, so those still need custom instrumentation in our
harness. Full `propose()` internals still unread but no longer blocking — the example
shows the config-level API is all we need, not proposer internals.

## 2026-07-23

First real run attempt of notebook 01 (Colab, L4, 22GB VRAM). Stage 1 (GPTQ) and the
start of Stage 2 (QAT) ran fine — "Trainable parameters (still-W3 layers only): 1193.3M"
printed correctly — but hit `CUDA out of memory` on the very first `backward()` call.
20.33/22.03 GiB already allocated going into it. Likely two compounding causes: (1)
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` only works if set before torch/CUDA
initializes in-process — if cells get re-run out of order after a mid-session edit
(which is what happened here, after adding the git-clone bootstrap cell), it silently
doesn't apply; a full Runtime restart + top-to-bottom re-run is the fix. (2)
`BATCH_TOKENS=1024` in fp32 with ~1.2B trainable params is just heavy for a single
22GB card — lowered the notebook's default to 256 and added an explicit
`gc.collect()`/`empty_cache()` right before Stage 2 starts to clear Stage-1
fragmentation. Not yet confirmed this actually fixes it — next attempt should confirm.

Second run attempt: switched to an A100-SXM4-40GB Colab instance instead, and ran the
notebook **as originally pulled** (before the 256/OOM-fix commit landed locally) — so
still `BATCH_TOKENS=1024`, matching the prior project's original recipe exactly. Worked
end to end. Real numbers: Stage 1 PPL 22.37, combined PPL 15.91, checkpoint 1184.8MB
compressed. See `docs/findings.md` for the full write-up — corrected recovery (68.0%)
came out higher than the prior project's own two runs of the same recipe (63.4%/65.8%),
flagged as unreplicated rather than trusted yet.

Nearly lost the output: the "Saved: results/..." print made it look like the file was
somewhere findable, but it actually only exists on the Colab VM's ephemeral local disk
(`/content/sgt-qat-draft/results/`), not Google Drive — user was looking in the *prior*
project's Drive `results/` folder, a completely different (and irrelevant) location.
Same ephemeral risk applies to the exported checkpoint itself, which is the more
valuable artifact. Backup (git push or Drive copy) still needs to happen before the
Colab runtime disconnects/recycles.

Also caught: `.gitignore` had `results/*.json` excluded from the start — a mistake,
since result JSONs are meant to be tracked (only `checkpoints/` should stay untracked,
for size reasons). Fixed. Reverted the notebook's `BATCH_TOKENS` default back to 1024
(from the 256 OOM workaround) since 1024 is what actually produced this validated
checkpoint on the A100 — 256 is now just a documented fallback for smaller GPUs.

Tried to `git push` the checkpoint from Colab — hit two blockers in sequence: (1)
`checkpoints/` is `.gitignore`'d (intentional, hadn't been resolved yet at that
point), (2) even after that, `git push` failed with `fatal: could not read Username
for 'https://github.com'` — no credentials configured for a non-interactive HTTPS
push. Fixed the auth problem with a `GITHUB_TOKEN` Colab secret (same pattern the
prior project's notebook 15 used for its own repo clone), wired into the remote URL
for that session only. That got the *results JSON* pushed fine. For the checkpoint
itself: `model.safetensors` turned out to be 1.2GB as a single file — over GitHub's
100MB-per-file limit and over GitHub's free LFS quota (1GB total), so git was never
going to work for it regardless of auth. Considered HF Hub (private repo, would let
vLLM load it by repo id like the EAGLE-3 baseline) vs. Google Drive; user chose
**Drive only**. Convention documented in context.md: Drive path
`MyDrive/sgt-qat-draft-checkpoints/qwen3-1.7b-sgt-qat/`, copy-to-local-disk before
vLLM load (don't read straight off the Drive mount).
