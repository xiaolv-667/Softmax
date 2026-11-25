#!/usr/bin/env python3
"""
bench_official.py —— 用 CANN 官方 Softmax 实现（aclnn，经 torch_npu）做对照测试

目的：给出与本项目同口径（1024x1024 fp16，沿最后一维）的官方有效带宽，
      作为对照基准；并通过大尺寸测量判断"当前尺寸是否受带宽限制"。

用途说明（避免误解）：本脚本只调用 torch.softmax 一行 API 取对照数据，
不含任何模型/训练逻辑。本项目的算子实现是纯 C++（Ascend C），不使用 PyTorch。

实测结论（详见 README）：
    1024x1024   33.16 us   126.48 GB/s   <- 本项目 v6 为 14.72 us / 284.92 GB/s（快 2.25x）
    2048x2048   33.92 us   494.54 GB/s
    4096x4096  101.25 us   662.81 GB/s

    关键观察：1024 与 2048 的耗时几乎相同（33.16 vs 33.92）而带宽差 4 倍，
    说明小尺寸下时间花在固定开销上，不是搬数据。
    这个判断直接指导了 v5（省掉 max 归约）与 v6（行级预取）的方向。

口径说明（与本项目 bench.sh 对齐）：
  - 沿最后一维做 softmax
  - 必要流量 = 读 1 次 + 写 1 次 = 2*M*N*2 字节
  - 用 device 侧 Event 计时（不是 host wall clock），排除 launch 抖动
  - 预热若干次，取 N 次平均
"""
import torch
import torch_npu  # noqa: F401  必须导入才能启用 npu 后端


def bench_softmax(m, n, iters, dtype=torch.float16):
    x = torch.randn(m, n, dtype=dtype, device="npu")

    for _ in range(5):                      # warmup
        torch.softmax(x, dim=-1)
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        y = torch.softmax(x, dim=-1)
    end.record()
    torch.npu.synchronize()

    ms = start.elapsed_time(end) / iters    # 单次平均耗时（device 侧）
    min_bytes = 2.0 * m * n * 2.0           # 读 + 写
    bw = min_bytes / (ms * 1e-3) / 1e9      # GB/s
    print(f"  softmax {m}x{n}  {ms * 1000:8.2f} us  eff_bw={bw:8.2f} GB/s")
    del x, y
    return bw


def main():
    print("=== CANN 官方 Softmax（aclnn via torch_npu）===")
    print(f"torch {torch.__version__} / torch_npu {torch.npu.get_device_name(0)}")
    print("")
    print("[同口径对比] 1024x1024 fp16，沿最后一维")
    bench_softmax(1024, 1024, 200)
    print("")
    print("[大尺寸：判断当前尺寸是否受带宽限制]")
    for s in (2048, 4096):
        bench_softmax(s, s, 100)


if __name__ == "__main__":
    main()
