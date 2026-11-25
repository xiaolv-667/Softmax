#include "kernel_operator.h"

using AscendC::TPosition;

namespace {
constexpr uint32_t M = 1024;   // 行数
constexpr uint32_t N = 1024;   // 每行长度

// ---- UB 地址布局（字节）与 v0 相同 ----
constexpr uint32_t XH_ADDR = 0;                     // half[N]  2048 B
constexpr uint32_t YH_ADDR = XH_ADDR + N * 2;       // half[N]  2048 B
constexpr uint32_t XF_ADDR = YH_ADDR + N * 2;       // float[N] 4096 B
constexpr uint32_t XT_ADDR = XF_ADDR + N * 4;       // float[N] 4096 B（归约 ping-pong）
constexpr uint32_t XE_ADDR = XT_ADDR + N * 4;       // float[N] 4096 B（Exp 输出）

constexpr uint32_t kAlignElems = 8;   // 32B / sizeof(float)

__aicore__ inline float TreeReduceMax(AscendC::LocalTensor<float> &a,
                                      AscendC::LocalTensor<float> &b, uint32_t len)
{
    AscendC::LocalTensor<float> *src = &a;
    AscendC::LocalTensor<float> *dst = &b;
    while (len > kAlignElems) {
        const uint32_t half = len >> 1;
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

// ===================== softmax v1：多核 =====================
// 相对 v0 的唯一改动：把 M 行按核切分，每核只处理自己的行区间。
// 一切数据结构、三遍遍历、exp 算两遍、串行同步全部保留。
//
// 行分配公式沿用 Matmul 项目验证过的最优余数分配：
//   rowBegin = (M * blk) / blkNum
//   rowEnd   = (M * (blk + 1)) / blkNum
// 1024 行 / 40 核 = 25.6，即 24 个核分 26 行、16 个核分 25 行，
// 理论负载失配 26/25.6 = 1.6%。行是天然的切分单位，无对齐要求（不像 Matmul 要 16 对齐），
// 所以这里的核间均衡几乎免费。
extern "C" __global__ __aicore__ void softmax_v1(GM_ADDR x, GM_ADDR y)
{
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

    // ---- 本核负责的行区间 ----
    const uint32_t blk = AscendC::GetBlockIdx();
    const uint32_t blkNum = AscendC::GetBlockNum();
    const uint32_t rowBegin = (M * blk) / blkNum;
    const uint32_t rowEnd = (M * (blk + 1)) / blkNum;

    for (uint32_t i = rowBegin; i < rowEnd; ++i) {
        // ---------------- 第 1 遍：求本行最大值 ----------------
        AscendC::DataCopy(xHalf, xGm[i * N], N);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast(xF, xHalf, AscendC::RoundMode::CAST_NONE, N);
        // V->MTE2：保证后续覆盖 xHalf 前，本遍的向量读已完成
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        const float rowMax = TreeReduceMax(xF, xT, N);

        // ---------------- 第 2 遍：exp(x - max) 并求和 ----------------
        AscendC::DataCopy(xHalf, xGm[i * N], N);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast(xF, xHalf, AscendC::RoundMode::CAST_NONE, N);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::Adds(xF, xF, -rowMax, N);
        AscendC::Exp(xE, xF, N);
        const float rowSum = TreeReduceSum(xE, xT, N);

        // ---------------- 第 3 遍：归一化并写回 ----------------
        AscendC::DataCopy(xHalf, xGm[i * N], N);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast(xF, xHalf, AscendC::RoundMode::CAST_NONE, N);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::Adds(xF, xF, -rowMax, N);
        AscendC::Exp(xE, xF, N);
        AscendC::Muls(xF, xE, 1.0f / rowSum, N);
        AscendC::Cast(yHalf, xF, AscendC::RoundMode::CAST_NONE, N);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(yGm[i * N], yHalf, N);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }
}

#ifndef ASCENDC_CPU_DEBUG
void softmax_v1_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y)
{
    softmax_v1<<<blockDim, nullptr, stream>>>(x, y);
}
#endif
