#!/usr/bin/env python3
"""精度校验：比对 output/y.bin 与 output/golden.bin

判定：fro_err < 1e-3 且 cos > 0.9999 视为通过。
额外显式检查 NaN/Inf —— 输出里出现 NaN 时 numpy 的范数也会变 NaN，
不做显式判断的话容易误判（Matmul 项目里踩过这个坑）。
"""
import sys
import numpy as np


def main():
    out = np.fromfile(sys.argv[1], dtype=np.float16).astype(np.float32)
    gold = np.fromfile(sys.argv[2], dtype=np.float16).astype(np.float32)

    n_nan = int(np.isnan(out).sum())
    n_inf = int(np.isinf(out).sum())
    if n_nan or n_inf:
        print(f"fro_err=nan  cos=nan")
        print(f"[FAIL] 输出含 {n_nan} 个 NaN / {n_inf} 个 Inf（共 {out.size} 个元素）")
        sys.exit(1)

    diff = out - gold
    fro = float(np.linalg.norm(diff) / np.linalg.norm(gold))
    denom = float(np.linalg.norm(out) * np.linalg.norm(gold))
    cos = float(np.dot(out, gold) / denom) if denom > 0 else 0.0

    print(f"fro_err={fro:.3e}  cos={cos:.8f}")
    if fro < 1e-3 and cos > 0.9999:
        print("test pass")
    else:
        print("result error")
        sys.exit(1)


if __name__ == "__main__":
    main()
