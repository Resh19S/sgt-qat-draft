# vLLM bug report — draft (not yet filed)

Status: draft, 2026-07-25. **Do not file until the placeholders below are filled
with real, pasted output from `notebooks/06_vllm_draft_model_compressed_tensors_bug_repro.ipynb`**
— per this project's standing rule, never submit fabricated or reconstructed
data to an external tracker. Everything else below (the error message, the
mechanism, the original discovery context) is sourced from `docs/findings.md`
2026-07-24 and `docs/context.md` "RESOLVED: notebook 03 draft_model loading".

This is a draft for the *text* of a GitHub issue against `vllm-project/vllm`.
Filing it (actually opening the issue) is a separate, explicit step — this
file does not do that on its own.

---

## Title

`[Bug]: SpeculativeConfig method="draft_model" cannot load compressed-tensors packed checkpoints`

## Your current environment

The output of `python -m vllm.collect_env`:

```
<PASTE collect_env OUTPUT HERE — from notebook 06, cell 1>
```

`pip show vllm` output:

```
<PASTE pip show vllm OUTPUT HERE — from notebook 06, cell 1>
```

## 🐛 Describe the bug

We're benchmarking a quantized (GPTQ + QAT, via `llmcompressor`, saved with
`compressed-tensors`' `save_compressed=True`) dense model as a speculative-decoding
drafter, compared against vLLM's built-in EAGLE-3 drafters. Loading our
compressed drafter checkpoint via `speculative_config={"method": "draft_model", ...}`
fails — vLLM builds the draft model's linear layers as plain, unquantized
`RowParallelLinear`/`nn.Linear` modules expecting an ordinary `.weight` tensor,
but the checkpoint genuinely stores compressed-tensors packed weights under
`.weight_packed` (plus separate `weight_scale`/`weight_zero_point` buffers) —
so the load fails outright, regardless of whether `quant_config` gets correctly
re-derived for the draft model at the `VllmConfig` level (it appears to; that
resolution just doesn't reach how the draft model's actual `nn.Module` layers
get constructed).

We originally hit this with a mixed-precision (W4/W3) checkpoint on a
Qwen3-1.7B drafter targeting Qwen3-8B — see the minimal repro below for why
that's not actually necessary to trigger the bug: a single, plain, unmixed
`W4A16` GPTQ checkpoint on a tiny public model reproduces the identical failure
shape.

**Steps to reproduce** (minimal repro — no private checkpoints, small model,
runs in a couple of minutes on a single GPU):

1. Quantize any model with `llmcompressor.oneshot()` + `GPTQModifier`, saved via
   `model.save_pretrained(path, save_compressed=True)` (genuine compressed-tensors
   packed export — confirmed via `weight_packed` present in the saved
   `.safetensors` keys, not a full-precision fallback):
   ```python
   from llmcompressor import oneshot
   from llmcompressor.modifiers.quantization import GPTQModifier

   recipe = GPTQModifier(targets="Linear", ignore=["lm_head"], scheme="W4A16", dampening_frac=0.01)
   oneshot(model=model, dataset=calib_dataset, recipe=recipe, max_seq_length=2048, num_calibration_samples=32)
   model.save_pretrained("checkpoints/tiny-w4a16-compressed", save_compressed=True)
   ```
2. Load it as a vLLM speculative-decoding drafter:
   ```python
   from vllm import LLM

   llm = LLM(
       model="Qwen/Qwen3-0.6B",
       speculative_config={
           "method": "draft_model",
           "model": "checkpoints/tiny-w4a16-compressed",
           "num_speculative_tokens": 3,
       },
       max_model_len=2048,
   )
   ```

Full runnable version (including the calibration-data setup and environment
bootstrap): `notebooks/06_vllm_draft_model_compressed_tensors_bug_repro.ipynb`
at https://github.com/Resh19S/sgt-qat-draft (see that notebook's cells 2-3
for the exact code executed).

**Actual behavior**: raises (originally observed on our mixed-precision
Qwen3-1.7B checkpoint, `layers.0.mlp.down_proj` — the exact module name will
differ on the minimal repro's tiny model, but the failure shape is the same):

```
<PASTE FULL TRACEBACK HERE — from notebook 06, cell "Trigger the bug", after
running it against the minimal repro. Original error, for reference, was:>

ValueError: There is no module or parameter named 'layers.0.mlp.down_proj.weight_packed'
in Qwen3Model. The available parameters belonging to layers.0.mlp.down_proj
(RowParallelLinear) are: {'layers.0.mlp.down_proj.weight'}
```

**Expected behavior**: `method="draft_model"` should either (a) correctly
construct the draft model's layers using the quantized module type implied by
its own `config.json`'s `quantization_config` (the same way the *target* model's
quantized layers get constructed when the target itself is quantized), or (b) if
compressed-tensors packed draft checkpoints are genuinely unsupported, fail with
a clear, actionable error at config-validation time rather than a generic
attribute-lookup `ValueError` deep in model construction.

## Additional context

- We worked around this by decompressing the checkpoint ourselves before
  loading (`compressed_tensors.ModelCompressor.from_pretrained_model()` +
  `.decompress_model()`, then explicitly replacing each still-quantized-typed
  module with a plain `nn.Linear` holding the dequantized weight — the wrapper
  module's own serialization logic kept re-adding `weight_scale`/`weight_shape`
  even after we deleted the underlying buffers directly, so a straight
  attribute-delete wasn't enough) and re-saving without `save_compressed=True`.
  This works, but obviously discards the whole point of shipping a compressed
  drafter — the "loaded" checkpoint we benchmark is back to full-precision
  size in VRAM.
- As a separate, useful data point even without a fix: we measured the
  genuinely compressed checkpoint's **standalone** (no vLLM, direct
  `transformers.AutoModelForCausalLM.from_pretrained()` load, no KV cache, no
  serving overhead) VRAM footprint independently, to at least get a real
  memory number for the compressed weights even though we can't benchmark them
  in-serving. Happy to share that methodology/numbers if useful signal for
  prioritizing a fix — quantized drafters only make sense as a memory-saving
  lever if this path works end-to-end.
- We did not dig further into *why* the config-level `quant_config`
  re-derivation (which does appear to correctly identify the draft model as
  quantized, per `VllmConfig.__post_init__`) doesn't propagate to how the draft
  model's layers actually get instantiated — flagging in case that's a useful
  starting pointer for whoever picks this up, but we haven't traced the exact
  code path ourselves beyond confirming the empirical failure.

---

## Before filing — checklist

- [ ] Ran `notebooks/06_vllm_draft_model_compressed_tensors_bug_repro.ipynb`
      end-to-end on a fresh Colab session.
- [ ] Confirmed cell 2's `assert has_packed` passed (checkpoint genuinely saved
      compressed, not a silent full-precision fallback).
- [ ] Confirmed cell 3 actually raised an error (if it didn't, **stop** — either
      this is fixed upstream already, or the repro needs adjusting; don't file
      a bug that no longer reproduces).
- [ ] Pasted the real `collect_env`/`pip show vllm` output into the placeholders
      above.
- [ ] Pasted the real, full traceback from cell 3 into the placeholder above.
- [ ] Re-read the filled-in issue once more before submitting — this file was
      drafted with placeholders precisely so nothing gets filed without a human
      checking the actual pasted data matches what's claimed in the prose.
