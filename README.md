[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# torch-embeddingbag-freq-scale-guard

A call-site workaround and diagnostic for a real MPS-only PyTorch correctness bug: `torch.nn.functional.embedding_bag(..., scale_grad_by_freq=True)` **silently ignores the flag on the MPS (Apple Silicon GPU) backend**, returning the unscaled gradient with no error or warning, while CPU (and CUDA) correctly divide each index's gradient contribution by how often that index occurs in the batch. Upstream reference: [pytorch/pytorch#190061](https://github.com/pytorch/pytorch/issues/190061) ("[MPS] embedding_bag silently ignores scale_grad_by_freq=True"), currently **open, labeled `module: correctness (silent)`**, with an unmerged fix PR ([#190062](https://github.com/pytorch/pytorch/pull/190062), state=OPEN as of this repo's creation) — independently re-checked via `gh pr view`, never trusted from a cached issue summary.

```python
import torch

weight = torch.zeros(4, 3, requires_grad=True)
idx = torch.tensor([1, 1, 0, 2])   # index 1 occurs twice
offsets = torch.tensor([0])

# CPU: correctly divides index 1's gradient by its frequency (2)
out_cpu = torch.nn.functional.embedding_bag(idx, weight, offsets, mode="sum", scale_grad_by_freq=True)
out_cpu.sum().backward()
weight.grad[1]  # tensor([1., 1., 1.])  -- correct: 2.0 / frequency 2

# MPS: silently returns the UNSCALED gradient -- same as scale_grad_by_freq=False
weight_mps = torch.zeros(4, 3, device="mps", requires_grad=True)
out_mps = torch.nn.functional.embedding_bag(idx.to("mps"), weight_mps, offsets.to("mps"), mode="sum", scale_grad_by_freq=True)
out_mps.sum().backward()
weight_mps.grad[1]  # tensor([2., 2., 2.], device='mps:0')  -- WRONG: flag silently ignored
```

Reproduced from scratch on real Apple Silicon (M4, macOS 15.7.7, torch 2.14.0): CPU and MPS silently disagree on the exact same input, with no error, warning, or documented device-dependent behavior anywhere. A control test with `scale_grad_by_freq=False` confirms CPU and MPS agree in that case — isolating the defect to the `scale_grad_by_freq=True` path specifically. Root cause (confirmed by reading `aten/src/ATen/native/mps/operations/EmbeddingBag.mm`): the MPS backward kernel accepts the flag as a parameter but never computes or applies the frequency-based scale factor at all.

**Real-world impact:** any training loop on Apple Silicon relying on `scale_grad_by_freq=True` to down-weight gradients from frequently-repeated tokens/IDs (a standard technique from the original word2vec/GloVe literature, still used in some recommendation-system and NLP embedding training setups) silently gets a *different, uncorrected* gradient on MPS than on CPU/CUDA — a silent, Apple-Silicon-specific training-divergence bug.

## Install and check

Requires Python 3.9+ and a compatible PyTorch installation (`torch>=2.0` in the optional extra).

```bash
git clone https://github.com/zhuhroscar-tech/torch-embeddingbag-freq-scale-guard.git
cd torch-embeddingbag-freq-scale-guard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[torch]"
torch-embeddingbag-freq-scale-guard
torch-embeddingbag-freq-scale-guard --json
```

The CLI reruns the repro on `cpu` always, and on `mps` **only when this host's MPS device passes a real allocation probe** — not just `torch.backends.mps.is_available()`, which is known to report `True` on virtualized CI runners (e.g. GitHub Actions macOS runners) that cannot actually allocate GPU memory ([actions/runner-images#9918](https://github.com/actions/runner-images/issues/9918), [pytorch/torchchat#1416](https://github.com/pytorch/torchchat/issues/1416)). When MPS isn't functional, the tool honestly reports it skipped that device rather than silently claiming untested behavior passed.

Its JSON includes the installed torch version, `mps_functional`, per-device native/guard gradients compared against the CPU oracle, `any_native_silently_wrong`, and `guard_fully_correct`.

Exit codes describe the **guard check**, not just native bug detection: `0` means the guard matched the CPU-oracle contract on every device this run exercised, `1` means a guard check failed, and `2` means torch could not be imported.

## Use in Python

```python
from torch_embeddingbag_freq_scale_guard import safe_embedding_bag

out = safe_embedding_bag(idx, weight, offsets, mode="sum", scale_grad_by_freq=True)
out.sum().backward()
# weight.grad now has the CORRECT 1/frequency-scaled gradient on MPS, CPU, or CUDA alike
```

`safe_embedding_bag` (and its factory `make_safe_embedding_bag(torch)`, for binding to a specific torch module) applies the 1/frequency gradient scaling itself, via a custom `torch.autograd.Function`, **independent of which backend's native kernel does or doesn't implement the flag** — so the result is guaranteed correct regardless of device. When `scale_grad_by_freq=False` (the common case, and unaffected by this bug on every backend) it delegates straight to the native op unchanged — no double-guarding of an already-correct path.

## Scope and limitations

- This tool does **not** patch PyTorch itself. You apply the guard function explicitly at your own `embedding_bag(..., scale_grad_by_freq=True)` call sites.
- The guard recomputes the backward pass via an extra `torch.autograd.grad` call and a `bincount` over the input indices — a real but small additional cost versus the (buggy, cheaper) native path; acceptable for the training-loop use case this bug affects, not appropriate for an extremely hot inner loop where every microsecond counts.
- Currently supports `mode="sum"` and `mode="mean"` fully (the `mode` argument is threaded through to the native forward call and the frequency-scaling math is mode-independent); `mode="max"` is accepted by the signature but its `scale_grad_by_freq=True` gradient semantics are less standard in practice and are not independently verified by this repo's test suite.
- Reproduced and verified on real Apple Silicon (M4) hardware for the MPS case, and on CPU (the correct-behavior oracle) — this repo does **not** have CUDA hardware to independently verify CUDA's behavior matches CPU's documented semantics; CUDA is assumed correct per PyTorch's own documentation, not independently tested here.
- Reproduced and verified only against **torch 2.14.0**. If a future released torch version merges PR #190062 and fixes this upstream, `any_native_silently_wrong` should report `False` for the `mps` case on that version, and this guard remains a safe (if slightly more expensive) no-op-equivalent fallback.
- GitHub Actions macOS CI runners cannot exercise the actual MPS-functional code path (see the virtualized-runner note above) — CI on `macos-latest` verifies the CPU path and the `mps_functional=False` skip-honestly branch, not the MPS repro itself. The MPS repro is verified on this project's real Apple Silicon development host, disclosed as such rather than claimed as CI-verified.

## Development

```bash
python -m pip install -e ".[dev,torch]"
python -m pytest -v --cov=torch_embeddingbag_freq_scale_guard
```

See [implementation](src/torch_embeddingbag_freq_scale_guard/core.py), [tests](tests/test_core.py), and [CI config](.github/workflows/ci.yml). Licensed under [MIT](LICENSE).
