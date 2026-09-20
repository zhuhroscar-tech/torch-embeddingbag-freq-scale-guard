[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# torch-embeddingbag-freq-scale-guard

一个针对 PyTorch 一个真实 MPS 专属正确性缺陷的调用点变通方案与诊断工具：`torch.nn.functional.embedding_bag(..., scale_grad_by_freq=True)` 在 **MPS（Apple Silicon GPU）后端上会静默忽略该标志**，返回未缩放的梯度且没有任何报错或警告；而 CPU（以及 CUDA）会正确地按索引在批次中出现的频率对每个索引的梯度贡献进行缩放。上游参考：[pytorch/pytorch#190061](https://github.com/pytorch/pytorch/issues/190061)（"[MPS] embedding_bag silently ignores scale_grad_by_freq=True"），截至本仓库创建时**处于 open 状态，标记为 `module: correctness (silent)`**，存在一个尚未合并的修复 PR（[#190062](https://github.com/pytorch/pytorch/pull/190062)，本仓库创建时状态为 OPEN）——已通过 `gh pr view` 独立复核，未采信缓存的 issue 摘要。

```python
import torch

weight = torch.zeros(4, 3, requires_grad=True)
idx = torch.tensor([1, 1, 0, 2])   # 索引 1 出现两次
offsets = torch.tensor([0])

# CPU：正确地按频率（2）缩小索引 1 的梯度
out_cpu = torch.nn.functional.embedding_bag(idx, weight, offsets, mode="sum", scale_grad_by_freq=True)
out_cpu.sum().backward()
weight.grad[1]  # tensor([1., 1., 1.])  -- 正确：2.0 / 频率 2

# MPS：静默返回未缩放的梯度 -- 与 scale_grad_by_freq=False 时相同
weight_mps = torch.zeros(4, 3, device="mps", requires_grad=True)
out_mps = torch.nn.functional.embedding_bag(idx.to("mps"), weight_mps, offsets.to("mps"), mode="sum", scale_grad_by_freq=True)
out_mps.sum().backward()
weight_mps.grad[1]  # tensor([2., 2., 2.], device='mps:0')  -- 错误：标志被静默忽略
```

已在真实 Apple Silicon 硬件（M4，macOS 15.7.7，torch 2.14.0）上从零复现：CPU 与 MPS 在完全相同的输入上给出不一致的静默结果，没有任何报错、警告，也没有任何文档说明这种设备相关的行为差异。将 `scale_grad_by_freq=False` 作为对照测试确认此时 CPU 与 MPS 结果一致——这将缺陷范围精确限定在 `scale_grad_by_freq=True` 这一路径上。根因（通过阅读 `aten/src/ATen/native/mps/operations/EmbeddingBag.mm` 确认）：MPS 反向传播内核虽然接受该标志作为参数，但从未计算或应用基于频率的缩放因子。

**实际影响：** 任何在 Apple Silicon 上训练、依赖 `scale_grad_by_freq=True` 来降低高频重复 token/ID 梯度权重的训练循环（这是源自 word2vec/GloVe 文献的标准技巧，至今仍用于部分推荐系统和 NLP 嵌入训练场景）在 MPS 上会静默得到与 CPU/CUDA *不同、未经修正*的梯度——这是一个静默的、Apple Silicon 专属的训练发散缺陷。

## 安装与检查

需要 Python 3.9+ 及兼容的 PyTorch 安装（可选依赖中的 `torch>=2.0`）。

```bash
git clone https://github.com/zhuhroscar-tech/torch-embeddingbag-freq-scale-guard.git
cd torch-embeddingbag-freq-scale-guard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[torch]"
torch-embeddingbag-freq-scale-guard
torch-embeddingbag-freq-scale-guard --json
```

命令行工具始终在 `cpu` 上重跑复现，并且**仅当本机 MPS 设备通过真实的分配探测时**才在 `mps` 上运行——而不仅仅依赖 `torch.backends.mps.is_available()`，后者在虚拟化的 CI 运行环境（例如 GitHub Actions 的 macOS runner）上会报告 `True`，但实际上无法分配 GPU 内存（[actions/runner-images#9918](https://github.com/actions/runner-images/issues/9918)，[pytorch/torchchat#1416](https://github.com/pytorch/torchchat/issues/1416)）。当 MPS 不可用时，工具会如实报告该设备被跳过，而不是静默地声称未经测试的行为已通过。

其 JSON 输出包含已安装的 torch 版本、`mps_functional`、与 CPU 参照值对比的各设备原生/防护梯度、`any_native_silently_wrong` 以及 `guard_fully_correct`。

退出码描述的是**防护检查结果**，而非仅仅是原生缺陷检测：`0` 表示本次运行所验证的每个设备上防护结果均匹配 CPU 参照契约，`1` 表示某项防护检查失败，`2` 表示无法导入 torch。

## 在 Python 中使用

```python
from torch_embeddingbag_freq_scale_guard import safe_embedding_bag

out = safe_embedding_bag(idx, weight, offsets, mode="sum", scale_grad_by_freq=True)
out.sum().backward()
# 无论在 MPS、CPU 还是 CUDA 上，weight.grad 现在都具有正确的 1/频率 缩放梯度
```

`safe_embedding_bag`（及其工厂函数 `make_safe_embedding_bag(torch)`，用于绑定到特定的 torch 模块）通过自定义的 `torch.autograd.Function` 自行应用 1/频率 梯度缩放，**与具体后端的原生内核是否实现该标志无关**——因此无论设备如何，结果都能保证正确。当 `scale_grad_by_freq=False`（常见情况，且在任何后端上都不受此缺陷影响）时，它会直接委托给原生操作、不做任何改动——不会对本就正确的路径进行重复防护。

## 范围与局限性

- 本工具**不会**修补 PyTorch 本身。你需要在自己调用 `embedding_bag(..., scale_grad_by_freq=True)` 的地方显式应用该防护函数。
- 该防护通过额外调用一次 `torch.autograd.grad` 并对输入索引执行 `bincount` 来重新计算反向传播——相较于（有缺陷但更廉价的）原生路径会带来真实但较小的额外开销；对于该缺陷影响的训练循环场景是可接受的，但不适合对每微秒都很敏感的极热内层循环。
- 目前完整支持 `mode="sum"` 与 `mode="mean"`（`mode` 参数会被透传给原生前向调用，且频率缩放的数学计算与 mode 无关）；`mode="max"` 在函数签名中被接受，但其 `scale_grad_by_freq=True` 的梯度语义在实践中不够标准，本仓库的测试套件未对其进行独立验证。
- 已在真实 Apple Silicon（M4）硬件上验证 MPS 场景，并在 CPU（正确行为的参照基准）上验证——本仓库**没有** CUDA 硬件来独立验证 CUDA 的行为是否与 CPU 的文档化语义一致；CUDA 的正确性是依据 PyTorch 自身文档假定的，本仓库未对其进行独立测试。
- 仅针对 **torch 2.14.0** 进行了复现与验证。如果未来发布的 torch 版本合并了 PR #190062 并修复了该上游缺陷，那么在该版本上 `any_native_silently_wrong` 的 `mps` 结果应报告为 `False`，而本防护仍将作为一个安全（尽管略有额外开销）的等效空操作回退方案继续生效。
- GitHub Actions 的 macOS CI 运行环境无法真正触发 MPS 功能正常的代码路径（见上文关于虚拟化运行环境的说明）——`macos-latest` 上的 CI 验证的是 CPU 路径以及 `mps_functional=False` 时如实跳过的分支，而非 MPS 复现本身。MPS 复现是在本项目真实的 Apple Silicon 开发主机上验证的，本文档如实披露这一点，而非声称已通过 CI 验证。

## 开发

```bash
python -m pip install -e ".[dev,torch]"
python -m pytest -v --cov=torch_embeddingbag_freq_scale_guard
```

参见[实现代码](src/torch_embeddingbag_freq_scale_guard/core.py)、[测试](tests/test_core.py)与 [CI 配置](.github/workflows/ci.yml)。基于 [MIT 许可证](LICENSE)开源。
