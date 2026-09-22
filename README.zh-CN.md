# torch-embeddingbag-freq-scale-guard

此防护已迁移到统一的 [`torch-correctness-guards`](https://github.com/zhuhroscar-tech/torch-correctness-guards) 包中。

请改用 umbrella package：

```bash
python -m pip install git+https://github.com/zhuhroscar-tech/torch-correctness-guards.git
torch-guard run embeddingbag-freq-scale
```

Python API：

```python
from torch_correctness_guards import safe_embedding_bag

out = safe_embedding_bag(idx, weight, offsets, mode="sum", scale_grad_by_freq=True)
```

本仓库已作为历史迁移存根归档。维护中的实现、测试和 CLI 现位于 `torch-correctness-guards`。
