# vLLM bug report — FILED, fix in review

**Filed as [vllm-project/vllm#49893](https://github.com/vllm-project/vllm/issues/49893),
2026-07-27.** Confirmed by maintainer `harjothkhara` within hours (labeled
`bug`/`quantization`), root cause matches our own finding: the draft model
loads under a `draft_model` prefix at runtime, which breaks exact-name/
anchored-regex `config_groups` target matching — single-scheme worked because
`Linear`-class-name matching is substring-based, mixed-precision (name-based)
targets weren't. Fix PR:
[vllm-project/vllm#49900](https://github.com/vllm-project/vllm/pull/49900)
(Python-only, no compile needed).

**Install path, resolved 2026-07-27**: `VLLM_USE_PRECOMPILED=1` 404s (their
nightly wheel index has nothing published for the auto-resolved commit, in
any variant — reported on the PR, no reply needed to proceed). Worked around
with a full source build instead (`pip install "git+..."`, no
`VLLM_USE_PRECOMPILED`, ~1hr on a Colab A100 — real compute cost). This also
surfaced that `llmcompressor` (needed to build the test checkpoints) and this
build's `torch==2.13.0` pin are mutually incompatible — checkpoint-building
and vllm-loading now happen in two separate Colab sessions, handed off via
Google Drive (see notebook 06 section "6-prep").

**Verification result, 2026-07-27**: Point 1 CONFIRMED — the mixed-precision
checkpoint no longer hits the original `weight_packed` `ValueError` at all,
the fix works for the issue as filed. But loading now hits a **new**,
different `AssertionError` in `vllm/model_executor/parameter.py:175`
(`load_merged_column_weight`) further into the process — `param_data.shape`
(`[3072, 96]`) vs. `loaded_weight.shape` (`[3072, 103]`) when loading a merged
column-parallel weight (looks like `gate_up_proj`). Confirmed via `%debug`
post-mortem (not a guess), and confirmed this isn't an artifact of our own
test checkpoint's construction — rebuilt it with a layer-boundary-aligned
split (so no merged weight straddles two bit-widths) and it failed
identically. **Posted this as a follow-up comment on the PR** with the exact
shape data. `notebooks/06_vllm_draft_model_compressed_tensors_bug_repro.ipynb`
section 6 has the full diagnostic trail and the 4-point check they originally
requested.

**Maintainer's reply, 2026-07-29 — superseded our merged-weight hypothesis
with the real diagnosis.** vLLM expects **dense** W3 packing (96 int32
columns, sub-byte-boundary packed — landed in `compressed-tensors` 0.17.0);
our checkpoint's W3 tensor has 103 columns (older whole-values-per-int32
layout). Coincide exactly for 4-bit, which is why only W3 ever tripped
anything. **Not a `draft_model`-path bug, not a merged-weight bug — a
packing-format/version mismatch.** He asked for our exact library versions
and suggested two isolation checks — see
`notebooks/07_w3_packing_format_isolation_checks.ipynb`.

**Isolation check 1 result, 2026-07-30: CONFIRMED exactly as predicted.**
Identical `AssertionError` at the same location, loading the same W3
checkpoint as a completely plain model — no `speculative_config`, no
`draft_model` at all. Clean proof this is unrelated to the PR. **Surprise**:
pinning `compressed-tensors>=0.17.0` (resolved to 0.17.1) did **not** fix
the packed shape — still `(3072, 103)`, not the expected dense 96-column
layout. Versions: `llmcompressor==0.12.0`, `compressed-tensors==0.17.1`,
`torch==2.11.0+cu128`. Reported both back on the PR, asking whether
`llmcompressor` needs its own version bump or a specific setting to invoke
the updated packer.

**Second negative result, 2026-07-30 — levers exhausted.** Tried upgrading
`llmcompressor` itself, not just `compressed-tensors`: `pip install -U
llmcompressor "compressed-tensors>=0.17.0"` resolved to `llmcompressor==0.12.0`
(unchanged — apparently already the latest on PyPI) and `compressed-tensors==0.17.0`
(dropped from 0.17.1 — `llmcompressor` pins it tighter than `>=0.17.0`
allows). Packed shape still `(3072, 103)`, unchanged. The latest installable
combination of both libraries still produces the old layout — reported this
back on the PR, asking directly whether a specific `llmcompressor`
version/branch or a recipe/config setting is needed to invoke the new
packer. **Isolation check 2 (W4/W4 `config_groups` drafter, needs the
fix-branch session) still pending — independent of this packing-format
question.**

**Status as of 2026-08-10 — holding, no further comment posted.** Checked
via the GitHub API: issue open; fix PR #49900 open/unmerged, with a
mixed-config_groups test added (Jul 30) but **merge-blocked on vLLM CI** —
harjothkhara asked (Aug 4) for a maintainer to add the `ready` label since
`pre-run-check` gates CI behind 4+ merged PRs and they have 1. So the fix is
done but stuck on vLLM core maintainers, not on us or harjothkhara. Decided
against another comment: our 2026-07-30 comment already logs the version
findings + open W3 question, so re-posting would just duplicate stale info,
and harjothkhara is blocked themselves. Plan: wait, re-check activity after
~another week (~2026-08-17+); if a comment ever becomes warranted it should
carry new data (isolation check 2), not repeat the thread. See
`docs/context.md` for the full state.

The rest of this file is the original filed issue text, kept for the record.

---

Status: draft, 2026-07-26 — scope **confirmed** via
`notebooks/06_vllm_draft_model_compressed_tensors_bug_repro.ipynb`. Section 3
(plain, single-scheme W4A16 compressed-tensors checkpoint) loaded
successfully. Section 5/5b (mixed-precision `config_groups`, W4 on half the
layers / W3 on the rest, same tiny model) reproduced the exact failure —
same `ValueError`, same `layers.0.mlp.down_proj.weight_packed` module path as
our original discovery on the real Qwen3-1.7B checkpoint. **The bug is
specifically about mixed-precision (`config_groups`) compressed-tensors
checkpoints, not compressed-tensors checkpoints in general** — the report
below is scoped to that, not the broader (and wrong) claim from the first
draft.

`collect_env`/`pip show vllm` output pasted in below (2026-07-26) — confirms
this reproduces on **vLLM 0.26.0** (newer than the "0.25.1" noted in the
original discovery back in `docs/findings.md` 2026-07-24 — good thing that
stale number wasn't reused here, since the bug evidently survived at least
one version bump). All placeholders now filled with real data; only the
final "re-read before submitting" checklist item remains.

This is a draft for the *text* of a GitHub issue against `vllm-project/vllm`.
Filing it (actually opening the issue) is a separate, explicit step — this
file does not do that on its own.

---

## Title

`[Bug]: SpeculativeConfig method="draft_model" cannot load mixed-precision compressed-tensors checkpoints (config_groups)`

## Your current environment

`pip show vllm` output:

```
Name: vllm
Version: 0.26.0
Summary: A high-throughput and memory-efficient inference and serving engine for LLMs
Home-page: https://github.com/vllm-project/vllm
Author: vLLM Team
Author-email:
License:
Location: /usr/local/lib/python3.12/dist-packages
Requires: aiohttp, anthropic, apache-tvm-ffi, blake3, cachetools, cbor2, cloudpickle, compressed-tensors, depyf, einops, fastapi, fastsafetensors, filelock, flashinfer-python, humming-kernels, ijson, jsonschema, lark, llguidance, lm-format-enforcer, mcp, mistral_common, model-hosting-container-standards, msgspec, ninja, numba, numpy, nvidia-cudnn-frontend, nvidia-cutlass-dsl, nvtx, openai, openai-harmony, opencv-python-headless, opentelemetry-api, opentelemetry-exporter-otlp, opentelemetry-sdk, opentelemetry-semantic-conventions-ai, outlines_core, partial-json-parser, pillow, prometheus-fastapi-instrumentator, prometheus_client, protobuf, psutil, py-cpuinfo, pybase64, pydantic, PyNvVideoCodec, python-json-logger, pyyaml, pyzmq, quack-kernels, regex, requests, safetensors, sentencepiece, setproctitle, setuptools, six, starlette, tiktoken, tilelang, tokenizers, tokenspeed-mla, torch, torchaudio, torchcodec, torchvision, tqdm, transformers, typing_extensions, watchfiles, xgrammar
Required-by:
```

The output of `python -m vllm.collect_env`:

```
Collecting environment information...
==============================
        System Info
==============================
OS                           : Ubuntu 22.04.5 LTS (x86_64)
GCC version                  : (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0
Clang version                : Could not collect
CMake version                : version 3.31.10
Libc version                 : glibc-2.35

==============================
       PyTorch Info
==============================
PyTorch version              : 2.11.0+cu128
Is debug build               : False
CUDA used to build PyTorch   : 12.8
ROCM used to build PyTorch   : N/A
XPU used to build PyTorch    : N/A

==============================
      Python Environment
==============================
Python version               : 3.12.13 (main, Mar  4 2026, 09:23:07) [GCC 11.4.0] (64-bit runtime)
Python platform              : Linux-6.6.122+-x86_64-with-glibc2.35

==============================
       CUDA / GPU Info
==============================
Is CUDA available            : True
CUDA runtime version         : 12.8.93
GPU models and configuration : GPU 0: NVIDIA L4
Nvidia driver version        : 580.82.07
cuDNN version                : Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.9.8.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9.8.0
/usr/lib/x86_64-linux-gnu/libcudnn_cnn.so.9.8.0
/usr/lib/x86_64-linux-gnu/libcudnn_engines_precompiled.so.9.8.0
/usr/lib/x86_64-linux-gnu/libcudnn_engines_runtime_compiled.so.9.8.0
/usr/lib/x86_64-linux-gnu/libcudnn_graph.so.9.8.0
/usr/lib/x86_64-linux-gnu/libcudnn_heuristic.so.9.8.0
/usr/lib/x86_64-linux-gnu/libcudnn_ops.so.9.8.0
HIP runtime version          : N/A
MIOpen runtime version       : N/A
Is XNNPACK available         : True

==============================
          CPU Info
==============================
Architecture:                            x86_64
CPU(s):                                  12
Vendor ID:                               GenuineIntel
Model name:                              Intel(R) Xeon(R) CPU @ 2.20GHz
Hypervisor vendor:                       KVM
Virtualization type:                     full
NUMA node(s):                            1

==============================
       CUDA / GPU Info (topology)
==============================
GPU0	 X 	0-11	0		N/A

==============================
Versions of relevant libraries
==============================
[pip3] flashinfer-python==0.6.14
[pip3] numpy==2.0.2
[pip3] torch==2.11.0+cu128
[pip3] torchaudio==2.11.0+cu128
[pip3] torchvision==0.26.0+cu128
[pip3] transformers==5.13.1
[pip3] triton==3.6.0
[conda] Could not collect

==============================
         vLLM Info
==============================
ROCM Version                 : Could not collect
vLLM Version                 : 0.26.0
vLLM Build Flags:
  CUDA Archs: Not Set; ROCm: Disabled; XPU: Disabled
```

(Full pip freeze and CPU flags/vulnerability listing omitted here for length
— available in the original notebook output if a maintainer needs the
complete listing; nothing omitted is relevant to this bug.)

## 🐛 Describe the bug

We're benchmarking a quantized (GPTQ + QAT, via `llmcompressor`, saved with
`compressed-tensors`' `save_compressed=True`) dense model as a speculative-decoding
drafter, compared against vLLM's built-in EAGLE-3 drafters. Our checkpoint uses
a **mixed-precision** recipe — `GPTQModifier(config_groups={...})` with two
groups at different bit-widths (W4 on one layer subset, W3 on the rest, in our
case sensitivity-ranked but that's irrelevant to this bug). Loading it via
`speculative_config={"method": "draft_model", ...}` fails.

**We isolated this to mixed precision specifically**, not compressed-tensors
checkpoints in general: a plain, single-scheme `GPTQModifier(scheme="W4A16")`
checkpoint (no `config_groups`, uniform bit-width) loads as a `draft_model`
with no issue. Only the `config_groups`-based, per-layer-different-bit-width
checkpoint fails — see the minimal repro below, which demonstrates both the
passing and failing case side by side on the same tiny model.

**Steps to reproduce** (minimal repro — no private checkpoints, small model,
runs in a few minutes on a single GPU):

1. Quantize a small model with `llmcompressor.oneshot()` + `GPTQModifier`
   using **two `config_groups` at different bit-widths**, save via
   `model.save_pretrained(path, save_compressed=True)`:
   ```python
   from llmcompressor import oneshot
   from llmcompressor.modifiers.quantization import GPTQModifier

   all_linear_names = [n for n, m in model.named_modules()
                        if isinstance(m, torch.nn.Linear) and n != "lm_head"]
   mid = len(all_linear_names) // 2
   group_w4, group_w3 = all_linear_names[:mid], all_linear_names[mid:]

   recipe = GPTQModifier(
       dampening_frac=0.01, ignore=["lm_head"],
       config_groups={
           "w4_group": {
               "targets": group_w4, "input_activations": None, "output_activations": None,
               "weights": {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "group", "group_size": 128},
           },
           "w3_group": {
               "targets": group_w3, "input_activations": None, "output_activations": None,
               "weights": {"num_bits": 3, "type": "int", "symmetric": True, "strategy": "group", "group_size": 128},
           },
       },
   )
   oneshot(model=model, dataset=calib_dataset, recipe=recipe, max_seq_length=2048, num_calibration_samples=32)
   model.save_pretrained("checkpoints/tiny-mixed-compressed", save_compressed=True)
   ```
2. Load it as a vLLM speculative-decoding drafter:
   ```python
   from vllm import LLM

   llm = LLM(
       model="Qwen/Qwen3-0.6B",
       speculative_config={
           "method": "draft_model",
           "model": "checkpoints/tiny-mixed-compressed",
           "num_speculative_tokens": 3,
       },
       max_model_len=2048,
   )
   ```
3. For contrast, the exact same steps with a single-scheme recipe
   (`GPTQModifier(targets="Linear", ignore=["lm_head"], scheme="W4A16", dampening_frac=0.01)`,
   no `config_groups`) load without error — confirming this is specific to the
   mixed-precision/`config_groups` case, not compressed-tensors checkpoints
   generally.

Full runnable version (both the passing single-scheme case and the failing
mixed-precision case, plus the environment bootstrap):
`notebooks/06_vllm_draft_model_compressed_tensors_bug_repro.ipynb` at
https://github.com/Resh19S/sgt-qat-draft (sections 2-3 = passing case,
sections 5-5b = failing case).

**Actual behavior**: raises, both on our original Qwen3-1.7B mixed-precision
checkpoint and on the tiny public-model minimal repro above (identical failure
shape, only the layer name/model differs):

```
DEBUG 07-26 16:50:41 [model_executor/models/utils.py:283] Loaded weight layers.0.input_layernorm.weight with shape torch.Size([1024])
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_960/3853740456.py in <cell line: 0>()
      5 from vllm import LLM
      6
----> 7 llm_mixed = LLM(
      8     model=MODEL_ID,
      9     speculative_config={

31 frames
/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py in _load_module(self, base_prefix, module, weights)
    393                     f"({module._get_name()}) are: {desc_param_keys}"
    394                 )
--> 395                 raise ValueError(msg)
    396
    397     @support_quantized_model_reload_from_hp_weights

ValueError: There is no module or parameter named 'layers.0.mlp.down_proj.weight_packed' in Qwen3Model. The available parameters belonging to layers.0.mlp.down_proj (RowParallelLinear) are: {'layers.0.mlp.down_proj.weight'}
```

(Colab collapsed the middle frames to "31 frames" — happy to expand the full
stack if useful, but the `_load_module` frame at
`vllm/model_executor/models/utils.py:395` is where it actually raises.)

**Expected behavior**: `method="draft_model"` should either (a) correctly
construct the draft model's layers using the quantized module type implied by
its own `config.json`'s `quantization_config` — the same way this already
works for a *single-scheme* compressed-tensors checkpoint (confirmed working,
see above) — extended to handle the `config_groups`/mixed-precision case, or
(b) if mixed-precision draft checkpoints are genuinely unsupported for some
structural reason, fail with a clear, actionable error at config-validation
time rather than a generic attribute-lookup `ValueError` deep in
`_load_module`.

## Additional context

- **The single-scheme case already works** — this is not a blanket
  "compressed-tensors draft models are unsupported" issue, it's specifically
  the mixed-precision/`config_groups` path. That's a narrower, hopefully
  easier fix surface than the general case.
- The `_load_module` frame is immediately followed (in the source, per the
  traceback context above) by a decorator named
  `support_quantized_model_reload_from_hp_weights` — we haven't traced
  whether that decorator is relevant to this path or just adjacent in the
  file, but it seemed like a plausible starting pointer given the name, so
  flagging it rather than silently omitting it.
- We worked around this by decompressing the checkpoint ourselves before
  loading (`compressed_tensors.ModelCompressor.from_pretrained_model()` +
  `.decompress_model()`, then explicitly replacing each still-quantized-typed
  module with a plain `nn.Linear` holding the dequantized weight — the wrapper
  module's own serialization logic kept re-adding `weight_scale`/`weight_shape`
  even after we deleted the underlying buffers directly, so a straight
  attribute-delete wasn't enough) and re-saving without `save_compressed=True`.
  This works, but discards the whole point of shipping a compressed drafter —
  the "loaded" checkpoint we actually benchmark is back to full-precision size
  in VRAM.
- As a separate, useful data point even without a fix: we measured the
  genuinely compressed (mixed-precision) checkpoint's **standalone** (no
  vLLM, direct `transformers.AutoModelForCausalLM.from_pretrained()` load, no
  KV cache, no serving overhead) VRAM footprint independently, to at least get
  a real memory number for the compressed weights even though we can't
  benchmark them in-serving. Happy to share that methodology/numbers if
  useful signal for prioritizing a fix — mixed-precision quantized drafters
  only make sense as a memory-saving lever if this path works end-to-end.

---

## Before filing — checklist

- [x] Ran `notebooks/06_vllm_draft_model_compressed_tensors_bug_repro.ipynb`
      end-to-end on a fresh Colab session.
- [x] Confirmed the mixed-precision checkpoint's `assert has_packed` passed
      (checkpoint genuinely saved compressed, not a silent full-precision
      fallback).
- [x] Confirmed the mixed-precision trigger cell actually raised the error
      (single-scheme case did NOT raise — that's the scope-narrowing finding
      already folded into this draft).
- [x] Pasted the real `collect_env`/`pip show vllm` output into the
      placeholders above (2026-07-26 — confirms this reproduces on vLLM 0.26.0).
- [x] Pasted the real, full traceback into the placeholder above (from the
      2026-07-26 run).
- [ ] **Re-read the filled-in issue once more before submitting** — this is
      the only remaining item, and it's yours to do, not something to
      check off on your behalf. Everything above is now real, pasted data;
      nothing left is fabricated or a placeholder. Once you've read it
      through, this is ready to file at
      https://github.com/vllm-project/vllm/issues/new/choose.
