#include "kernel_operator.h"

using AscendC::TPosition;

namespace {
constexpr uint32_t M = 1024;   // 行数
constexpr uint32_t N = 1024;   // 每行长度

// ---- UB 地址布局（字节）----
constexpr uint32_t XH_ADDR = 0;                     // half[N]  2048 B
constexpr uint32_t YH_ADDR = XH_ADDR + N * 2;       // half[N]  2048 B
constexpr uint32_t XF_ADDR = YH_ADDR + N * 2;       // float[N] 4096 B（原始值，全程保留）
constexpr uint32_t XT_ADDR = XF_ADDR + N * 4;       // float[N] 4096 B（工作区/归约 ping-pong）
constexpr uint32_t XE_ADDR = XT_ADDR + N * 4;       // float[N] 4096 B（exp 结果，保留）
// 共 16 KB

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

// ===================== softmax v2：整行驻留 UB，一遍遍历 =====================
// 相对 v1 的关键改动：把「三遍独立遍历、每遍重读 GM」合并成「读一次、算完写一次」。
//
// v1 的浪费：
//   - GM 流量 = 3 读 + 1 写 = 8 MB（理论上限只需 1 读 1 写 = 4 MB）
//   - exp 被算了两遍（pass2 求 sum 一次、pass3 归一化再算一次）
//
// v2 的做法（一行只读一次）：
//   DataCopy 读入 → Cast 到 xF          // xF 保存原始值，全程不再重读 GM
//   求 max：先把 xF 拷到 xT，在 xT 上归约（保留 xF）
//   exp  ：xT = xF - max → Exp 到 xE     // xE 保存 exp 结果，只算一次
//   求 sum：先把 xE 拷到 xT，在 xT 上归约（保留 xE）
//   归一化：xF = xE * (1/sum) → Cast → 写回
//
// 预期：流量 8 MB → 4 MB，exp 次数 2 → 1，相对 v1 应有接近 2x 的提升。
extern "C" __global__ __aicore__ void softmax_v2(GM_ADDR x, GM_ADDR y)
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

    const uint32_t blk = AscendC::GetBlockIdx();
    const uint32_t blkNum = AscendC::GetBlockNum();
    const uint32_t rowBegin = (M * blk) / blkNum;
    const uint32_t rowEnd = (M * (blk + 1)) / blkNum;

    for (uint32_t i = rowBegin; i < rowEnd; ++i) {
        // ---- 唯一一次读入 ----
        AscendC::DataCopy(xHalf, xGm[i * N], N);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast(xF, xHalf, AscendC::RoundMode::CAST_NONE, N);
        // V->MTE2：xHalf 下一行要被覆盖，等本行的向量读完成
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);

        // ---- 求行最大值（在副本 xT 上归约，保留 xF 原始值）----
        AscendC::Muls(xT, xF, 1.0f, N);
        const float rowMax = TreeReduceMax(xT, xE, N);

        // ---- exp(x - max)，结果存 xE（只算一次）----
        AscendC::Adds(xT, xF, -rowMax, N);
        AscendC::Exp(xE, xT, N);

        // ---- 求行和（在副本 xT 上归约，保留 xE）----
        AscendC::Muls(xT, xE, 1.0f, N);
        const float rowSum = TreeReduceSum(xT, xF, N);

        // ---- 归一化并写回 ----
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
void softmax_v2_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y)
{
    softmax_v2<<<blockDim, nullptr, stream>>>(x, y);
}
#endif
