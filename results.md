# Softmax 性能实验台账

**固定条件**

| 项 | 值 |
|---|---|
| 问题规模 | M = N = 1024（沿行做 softmax） |
| 数据类型 | fp16 输入/输出，fp32 中间计算 |
| 必要流量 | 读 1 次 + 写 1 次 = **4 MB**（GM 流量理论下限） |
| 精度判据 | `fro_err < 1e-3` 且 `cos > 0.9999`，且无 NaN/Inf |
| 计时方式 | `aclrtEvent` 夹住 N 次 launch，去 warmup，取平均 |
| 设备 | Ascend 910B3 |

`eff_bw_gbs` = 4 MB ÷ 单次耗时，越接近硬件带宽说明 GM 流量越接近下限。

| 时间 | 版本 | SOC/构建类型 | blockDim | repeat | avg_ms | eff_bw_gbs | fro_err | cos | 结果 |
|------|------|--------------|----------|--------|--------|-----------|---------|-----|------|
| 2026-11-25 16:30 | softmax@v5-multicore-9-g2014adf-dirty | Ascend910B3/Release | 1 | 100 | 0.861923 | 4.87 | 1.061e-05 | 0.99999991 | PASS |
| 2026-11-25 16:31 | softmax@v5-multicore-10-gd8b64e2-dirty | Ascend910B3/Release | 40 | 100 | 0.025499 | 164.49 | 1.061e-05 | 0.99999991 | PASS |
| 2026-11-25 16:32 | softmax@v5-multicore-11-g2c4a444-dirty | Ascend910B3/Release | 40 | 100 | 0.019589 | 214.11 | 1.061e-05 | 0.99999991 | PASS |
| 2026-11-25 16:32 | softmax@v5-multicore-12-gd5c206e-dirty | Ascend910B3/Release | 40 | 100 | 0.019620 | 213.77 | 1.061e-05 | 0.99999991 | PASS |
| 2026-11-25 16:33 | softmax@v5-multicore-12-gd5c206e-dirty | Ascend910B3/Release | 40 | 100 | 0.019388 | 216.34 | 1.061e-05 | 0.99999991 | PASS |
| 2026-11-25 16:34 | softmax@v5-multicore-12-gd5c206e-dirty | Ascend910B3/Release | 40 | 100 | 0.018730 | 223.93 | 1.061e-05 | 0.99999991 | PASS |
| 2026-11-25 16:42 | softmax@softmax-v4-1-g9d42b8f-dirty | Ascend910B3/Release | 40 | 100 | 0.015064 | 278.44 | 6.414e-06 | 0.99999982 | PASS |
| 2026-11-25 17:54 | softmax@softmax-v5-dirty | Ascend910B3/Release | 40 | 100 | 0.014721 | 284.92 | 6.414e-06 | 0.99999982 | PASS |
