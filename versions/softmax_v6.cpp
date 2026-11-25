#include "kernel_operator.h"

using AscendC::TPosition;

namespace {
constexpr uint32_t M = 1024;   // 行数
constexpr uint32_t N = 1024;   // 每行长度

// ---- UB 地址布局（字节）----
constexpr uint32_t XH0_ADDR = 0;                    // half[N]  2048 B  输入 ping-pong 0
constexpr uint32_t XH1_ADDR = XH0_ADDR + N * 2;     // half[N]  2048 B  输入 ping-pong 1
constexpr uint32_t YH_ADDR  = XH1_ADDR + N * 2;     // half[N]  2048 B
constexpr uint32_t XF_ADDR  = YH_ADDR + N * 2;      // float[N] 4096 B（原始值）
constexpr uint32_t XT_ADDR  = XF_ADDR + N * 4;      // float[N] 4096 B（工作区）
constexpr uint32_t XE_ADDR  = XT_ADDR + N * 4;      // float[N] 4096 B（exp 结果）
// 共 18 KB

constexpr uint32_t kAlignElems = 8;   // 32B / sizeof(float)

// 与 v5 一致：默认不做 max 平移（见 v5 的说明）。需要数值保护时改 true。
constexpr bool kUseMaxShift = false;

__aicore__ inline float TreeReduceMax(AscendC::LocalTensor<float> &base,
                                      AscendC::LocalTensor<float> &w0,
                                      AscendC::LocalTensor<float> &w1, uint32_t len)
{
    uint32_t half = len >> 1;
    AscendC::Max(w0, base, base[half], static_cast<int32_t>(half));
    AscendC::LocalTensor<float> *src = &w0;
    AscendC::LocalTensor<float> *dst = &w1;
    len = half;
    while (len > kAlignElems) {
        half = len >> 1;
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

__aicore__ inline float RowSumFrom(AscendC::LocalTensor<float> &base,
                                   AscendC::LocalTensor<float> &w0,
                                   AscendC::LocalTensor<float> &w1, uint32_t len)
{
    uint32_t half = len >> 1;
    AscendC::Add(w0, base, base[half], static_cast<int32_t>(half));   // 首步：读 base，写 w0
    AscendC::LocalTensor<float> *src = &w0;
    AscendC::LocalTensor<float> *dst = &w1;
    len = half;
    while (len > kAlignElems) {
        half = len >> 1;
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

// ===================== softmax v6：v5 + 行级预取 =====================
// 数据依据（v0~v5 的 msprof 汇总，见 README）：
//   ver  Vector  MTE2读  MTE3写  Scalar
//   v2    43.4    34.7    17.0    25.3
//   v3    47.2    38.1    19.2    31.4   <- v3 就是"v2+预取"，当时 Vector 已是最高项
//   v4    40.3    37.4    18.2    26.3
//   v5    34.7    45.4    22.1    26.7   <- v5 省掉 max 归约后，MTE2 反成最高项
//
// 所以 v3 当年零收益的原因不是"预取没用"，而是**当时预取掩盖的是非瓶颈项**：
// v2/v3 时 Vector(43.4/47.2) 高于 MTE2(34.7/38.1)，掩盖 MTE2 自然没有收益。
// 而 v5 把 Vector 压到 34.7 后，MTE2 升到 45.4 成为头号项——
// 此时再做预取，掩盖的才是真正的瓶颈。
//
// 实现：两块 xHalf 做 ping-pong，先发下一行的 DataCopy，再算本行，
// 让 MTE2 的读与整段 Vector 计算在时间上重叠。
extern "C" __global__ __aicore__ void softmax_v6(GM_ADDR x, GM_ADDR y)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();

    AscendC::GlobalTensor<half> xGm, yGm;
    xGm.SetGlobalBuffer((__gm__ half *)x, M * N);
    yGm.SetGlobalBuffer((__gm__ half *)y, M * N);

    AscendC::LocalTensor<half>  xH0  (TPosition::VECCALC, XH0_ADDR, N);
    AscendC::LocalTensor<half>  xH1  (TPosition::VECCALC, XH1_ADDR, N);
    AscendC::LocalTensor<half>  yHalf(TPosition::VECCALC, YH_ADDR, N);
    AscendC::LocalTensor<float> xF   (TPosition::VECCALC, XF_ADDR, N);
    AscendC::LocalTensor<float> xT   (TPosition::VECCALC, XT_ADDR, N);
    AscendC::LocalTensor<float> xE   (TPosition::VECCALC, XE_ADDR, N);

    const uint32_t blk = AscendC::GetBlockIdx();
    const uint32_t blkNum = AscendC::GetBlockNum();
    const uint32_t rowBegin = (M * blk) / blkNum;
    const uint32_t rowEnd = (M * (blk + 1)) / blkNum;
    if (rowBegin >= rowEnd) {
        return;
    }

    // 预取第 0 行
    AscendC::DataCopy(xH0, xGm[rowBegin * N], N);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

    for (uint32_t i = rowBegin; i < rowEnd; ++i) {
        const uint32_t cur = (i - rowBegin) & 1u;
        AscendC::LocalTensor<half> &xCur = (cur == 0) ? xH0 : xH1;
        AscendC::LocalTensor<half> &xNxt = (cur == 0) ? xH1 : xH0;
        const event_t evCur = (cur == 0) ? EVENT_ID0 : EVENT_ID1;
        const event_t evNxt = (cur == 0) ? EVENT_ID1 : EVENT_ID0;

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evCur);   // 本行数据到位

        // 立刻预取下一行：这次 MTE2 会与下面整段 Vector 计算并行
        if (i + 1 < rowEnd) {
            AscendC::DataCopy(xNxt, xGm[(i + 1) * N], N);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evNxt);
        }

        AscendC::Cast(xF, xCur, AscendC::RoundMode::CAST_NONE, N);
        // V->MTE2：本行读完后，这块 buffer 才能被两行后的预取覆盖
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evCur);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evCur);

        float rowMax = 0.0f;
        if constexpr (kUseMaxShift) {
            rowMax = TreeReduceMax(xF, xT, xE, N);
        }

        AscendC::Adds(xT, xF, -rowMax, N);
        AscendC::Exp(xE, xT, N);

        const float rowSum = RowSumFrom(xE, xT, xF, N);

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
void softmax_v6_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y)
{
    softmax_v6<<<blockDim, nullptr, stream>>>(x, y);
}
#endif
