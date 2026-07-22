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
