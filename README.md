# torch-embeddingbag-freq-scale-guard

This guard has moved into the consolidated [`torch-correctness-guards`](https://github.com/zhuhroscar-tech/torch-correctness-guards) package.

Use the umbrella package instead:

```bash
python -m pip install git+https://github.com/zhuhroscar-tech/torch-correctness-guards.git
torch-guard run embeddingbag-freq-scale
```

Python API:

```python
from torch_correctness_guards import safe_embedding_bag

out = safe_embedding_bag(idx, weight, offsets, mode="sum", scale_grad_by_freq=True)
```

This source repository is archived as a historical migration stub. The maintained implementation, tests, and CLI now live in `torch-correctness-guards`.
