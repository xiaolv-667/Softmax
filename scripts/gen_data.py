import os
import numpy as np

# Softmax 测试数据：M 行 x N 列，沿行（axis=1）做 softmax
M = int(os.environ.get("SM_M", "1024"))
N = int(os.environ.get("SM_N", "1024"))
DT = np.float16

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

rng = np.random.default_rng(42)
x = rng.standard_normal((M, N)).astype(DT)
x.tofile("input/x.bin")

# golden：用 fp32 计算（先减最大值保证数值稳定），再转回 fp16
xf = x.astype(np.float32)
xf = xf - xf.max(axis=1, keepdims=True)
e = np.exp(xf)
golden = (e / e.sum(axis=1, keepdims=True)).astype(DT)
golden.tofile("output/golden.bin")

print(f"gen softmax {M}x{N} {DT.__name__}, x={x.nbytes} bytes")
