#include "kernel_operator.h"

using AscendC::TPosition;

namespace {
constexpr uint32_t M = 1024;   // 行数
constexpr uint32_t N = 1024;   // 每行长度

constexpr uint32_t XH_ADDR = 0;                     // half[N]  2048 B
constexpr uint32_t YH_ADDR = XH_ADDR + N * 2;       // half[N]  2048 B
constexpr uint32_t XF_ADDR = YH_ADDR + N * 2;       // float[N] 4096 B（原始值）
constexpr uint32_t XT_ADDR = XF_ADDR + N * 4;       // float[N] 4096 B（工作区）
constexpr uint32_t XE_ADDR = XT_ADDR + N * 4;       // float[N] 4096 B（exp 结果）
// 共 16 KB

constexpr uint32_t kAlignElems = 8;   // 32B / sizeof(float)

// ---- 行内归约（带"首步从外部源读"的版本）----
// 相对 v2 的关键改进：v2 用 `Muls(xT, xF, 1.0f, N)` 整行拷贝一次再归约，
// 每行两次归约就多出两条整行拷贝（每条 16 个向量指令）。
// 这里让归约的第一步直接「读源 buffer、写工作 buffer」，
// 既保留源数据又省掉那次整行拷贝。
//
// 两条硬约束（都是踩过的坑）：
//   1) 向量操作数需 32 字节对齐 => 偏移视图只有 stride>=8（fp32）才合法，
//      所以向量归约做到 8，余下 8 个元素走标量。
//   2) dst 与 src 不能同 buffer => 用两个工作 buffer 交替。
__aicore__ inline float RowMaxFrom(AscendC::LocalTensor<float> &base,
                                   AscendC::LocalTensor<float> &w0,
                                   AscendC::LocalTensor<float> &w1, uint32_t len)
{
    uint32_t half = len >> 1;
    AscendC::Max(w0, base, base[half], static_cast<int32_t>(half));   // 首步：读 base，写 w0
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

// ===================== softmax v4：归约去掉整行拷贝 =====================
// 基于 v2（一遍遍历、流量已达理论下限 4 MB）再做一步：
//   求 max：v2 是「整行拷到 xT，再拷到 xE 归约」；v4 直接「读 xF 写 xT」起步，省一次拷贝
//   求 sum：v2 是「整行拷到 xT，再归约」；        v4 直接「读 xE 写 xT」起步，省一次拷贝
// 每行省下 2 条整行向量拷贝（各 16 个向量指令），并保留 xF / xE 的原始内容。
extern "C" __global__ __aicore__ void softmax_v4(GM_ADDR x, GM_ADDR y)
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
        AscendC::DataCopy(xHalf, xGm[i * N], N);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast(xF, xHalf, AscendC::RoundMode::CAST_NONE, N);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);

        // 求 max：直接以 xF 为源，不拷贝
        const float rowMax = RowMaxFrom(xF, xT, xE, N);

        AscendC::Adds(xT, xF, -rowMax, N);
        AscendC::Exp(xE, xT, N);

        // 求 sum：直接以 xE 为源，不拷贝
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
void softmax_v4_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y)
{
    softmax_v4<<<blockDim, nullptr, stream>>>(x, y);
}
#endif
