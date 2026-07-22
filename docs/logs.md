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
Full `propose()` internals and the exact end-to-end `speculative_config` invocation
still need a closer read before writing notebook code (see context.md).
