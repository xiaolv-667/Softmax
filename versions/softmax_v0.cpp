#include "kernel_operator.h"

using AscendC::TPosition;

namespace {
constexpr uint32_t M = 1024;   // 行数
constexpr uint32_t N = 1024;   // 每行长度

// ---- UB 地址布局（字节）----
constexpr uint32_t XH_ADDR = 0;                     // half[N]  2048 B
constexpr uint32_t YH_ADDR = XH_ADDR + N * 2;       // half[N]  2048 B
constexpr uint32_t XF_ADDR = YH_ADDR + N * 2;       // float[N] 4096 B
constexpr uint32_t XT_ADDR = XF_ADDR + N * 4;       // float[N] 4096 B（归约 ping-pong）
constexpr uint32_t XE_ADDR = XT_ADDR + N * 4;       // float[N] 4096 B（Exp 输出）
// 共 16 KB，UB 有 192 KB，余量充足

// ---- 行内归约 ----
// 两个 buffer 交替做 dst/src，避免 dst 与 src 同 buffer 重叠。
//
// 【关键约束】向量指令要求 LocalTensor 操作数 **32 字节对齐**。
// fp32 下 32B = 8 个元素，所以 src[half] 这个偏移视图只有在 half 是 8 的倍数时合法。
// 之前一路降到 stride=1/2/4（字节偏移 4/8/16）会直接触发
// AIV 越界异常（plog: `[vector core exception] retCode=0x31`，aclError 507035）。
// 因此向量归约只做到 stride >= 8，最后剩余 8 个元素在标量侧处理。
constexpr uint32_t kAlignElems = 8;   // 32B / sizeof(float)

__aicore__ inline float TreeReduceMax(AscendC::LocalTensor<float> &a,
                                      AscendC::LocalTensor<float> &b, uint32_t len)
{
    AscendC::LocalTensor<float> *src = &a;
    AscendC::LocalTensor<float> *dst = &b;
    while (len > kAlignElems) {
        const uint32_t half = len >> 1;                        // half >= 8，偏移满足 32B 对齐
        AscendC::Max(*dst, *src, (*src)[half], static_cast<int32_t>(half));
        AscendC::LocalTensor<float> *t = src;
        src = dst;
        dst = t;
        len = half;
    }
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    float m = src->GetValue(0);
    for (uint32_t k = 1; k < len; ++k) {
        const float v = src->GetValue(k);
        if (v > m) {
            m = v;
        }
    }
    return m;
}

__aicore__ inline float TreeReduceSum(AscendC::LocalTensor<float> &a,
                                      AscendC::LocalTensor<float> &b, uint32_t len)
{
    AscendC::LocalTensor<float> *src = &a;
    AscendC::LocalTensor<float> *dst = &b;
    while (len > kAlignElems) {
        const uint32_t half = len >> 1;
        AscendC::Add(*dst, *src, (*src)[half], static_cast<int32_t>(half));
        AscendC::LocalTensor<float> *t = src;
        src = dst;
        dst = t;
        len = half;
    }
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    float s = 0.0f;
    for (uint32_t k = 0; k < len; ++k) {
        s += src->GetValue(k);
    }
    return s;
}
}

// ===================== softmax v0：朴素实现 =====================
// 单核、逐行、**三遍独立遍历**，每遍都从 GM 重新读原始数据：
//   pass1: 求本行最大值 rowMax
//   pass2: 重读 → exp(x-rowMax) → 求和 rowSum
//   pass3: 再重读 → exp(x-rowMax) → 除以 rowSum → 写回
//
// 作为基线，代价一眼可见：
//   - GM 流量 = 3 读 + 1 写 = 8 MB（理论下限是 1 读 1 写 = 4 MB）
//   - exp 算了两遍（pass2 / pass3 各一次）
//   - 单核，40 个 AIV 只用了 1 个
//   - 每遍 SetFlag/WaitFlag 全串行，搬运与计算零重叠
//
// 【本版最重要的一个坑】xHalf 被 MTE2 写入、被 Vector 读取，且每行要写三次。
// 只加 MTE2_V（写→读）不够，还必须加 **V_MTE2（读→写）**：
// 否则下一遍的 DataCopy 可能在上一遍 Cast 还在读 xHalf 时就把它覆盖掉。
// 症状极具迷惑性：约 40% 的行出错、其余行完全正确，看起来像随机竞态而非逻辑错误。
// 排查方式：把 pass3 简化成「读→转fp32→转回fp16→写」的纯拷贝，
// 发现输出都不等于输入（610/1024 行正确），才定位到这条缺失的同步。
extern "C" __global__ __aicore__ void softmax_v0(GM_ADDR x, GM_ADDR y)
{
    // 纯 Vector 算子，必须显式声明为 AIV-only，
    // 否则工具链会编成 mix 核（每块 1 AIC + 2 AIV），blockDim 语义翻倍。
    // 这个坑在 Matmul 项目里踩过（导致 1.6x 性能回退），这里提前避开。
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    AscendC::InitSocState();

    AscendC::GlobalTensor<half> xGm, yGm;
    xGm.SetGlobalBuffer((__gm__ half *)x, M * N);
    yGm.SetGlobalBuffer((__gm__ half *)y, M * N);

    AscendC::LocalTensor<half>  xHalf(TPosition::VECCALC, XH_ADDR, N);
    AscendC::LocalTensor<half>  yHalf(TPosition::VECCALC, YH_ADDR, N);
    AscendC::LocalTensor<float> xF   (TPosition::VECCALC, XF_ADDR, N);
    AscendC::LocalTensor<float> xT   (TPosition::VECCALC, XT_ADDR, N);
    AscendC::LocalTensor<float> xE   (TPosition::VECCALC, XE_ADDR, N);

    for (uint32_t i = 0; i < M; ++i) {
        // ---------------- 第 1 遍：求本行最大值 ----------------
        AscendC::DataCopy(xHalf, xGm[i * N], N);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast(xF, xHalf, AscendC::RoundMode::CAST_NONE, N);
        // 读完 xHalf 立刻反向上锁：保证后续 DataCopy 覆盖它之前，本遍的读已完成
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        const float rowMax = TreeReduceMax(xF, xT, N);

        // ---------------- 第 2 遍：exp(x - max) 并求和 ----------------
        AscendC::DataCopy(xHalf, xGm[i * N], N);      // 重读 GM
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast(xF, xHalf, AscendC::RoundMode::CAST_NONE, N);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::Adds(xF, xF, -rowMax, N);
        AscendC::Exp(xE, xF, N);                        // 输出到独立 buffer，避免就地
        const float rowSum = TreeReduceSum(xE, xT, N);

        // ---------------- 第 3 遍：归一化并写回 ----------------
        AscendC::DataCopy(xHalf, xGm[i * N], N);      // 第三次重读
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast(xF, xHalf, AscendC::RoundMode::CAST_NONE, N);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::Adds(xF, xF, -rowMax, N);
        AscendC::Exp(xE, xF, N);                        // exp 第二次计算（写到独立 buffer）
        AscendC::Muls(xF, xE, 1.0f / rowSum, N);
        AscendC::Cast(yHalf, xF, AscendC::RoundMode::CAST_NONE, N);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(yGm[i * N], yHalf, N);
        // yHalf 被 MTE3 读过之后，下一行才允许 Vector 再次写它
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }
}

#ifndef ASCENDC_CPU_DEBUG
void softmax_v0_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y)
{
    softmax_v0<<<blockDim, nullptr, stream>>>(x, y);
}
#endif
