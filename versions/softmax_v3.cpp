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

// ===================== softmax v3：行级双缓冲 =====================
// 相对 v2 的唯一改动：把「读本行 -> 算本行」的串行链改成
// 「先预取下一行，再算本行」，让 MTE2 的读与 Vector 的计算在时间上重叠。
//
// v2 的瓶颈（实测）：每行只有 4 KB 数据（读 2KB + 写 2KB），
// 但总时长折算下来每核只有 5.2 GB/s——说明时间花在等，不在搬。
// 每行的串行链是：
//   读->等->Cast->归约(10步向量+8标量)->Adds->Exp->归约->Muls->Cast->等->写
// 行与行之间零重叠。
//
// v3 用两块 xHalf 做 ping-pong：
//   WaitFlag<MTE2_V>(cur)        本行数据已到
//   DataCopy(xHalf[nxt], 下一行)  立刻发起预取，与前后的 Vector 计算并行
//   算本行 ...
// 这样 MTE2 搬下一行的时间被本行的 Vector 计算掩盖掉。
extern "C" __global__ __aicore__ void softmax_v3(GM_ADDR x, GM_ADDR y)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();

    AscendC::GlobalTensor<half> xGm, yGm;
    xGm.SetGlobalBuffer((__gm__ half *)x, M * N);
    yGm.SetGlobalBuffer((__gm__ half *)y, M * N);

    AscendC::LocalTensor<half>  xH0(TPosition::VECCALC, XH0_ADDR, N);
    AscendC::LocalTensor<half>  xH1(TPosition::VECCALC, XH1_ADDR, N);
    AscendC::LocalTensor<half>  yHalf(TPosition::VECCALC, YH_ADDR, N);
    AscendC::LocalTensor<float> xF (TPosition::VECCALC, XF_ADDR, N);
    AscendC::LocalTensor<float> xT (TPosition::VECCALC, XT_ADDR, N);
    AscendC::LocalTensor<float> xE (TPosition::VECCALC, XE_ADDR, N);

    const uint32_t blk = AscendC::GetBlockIdx();
    const uint32_t blkNum = AscendC::GetBlockNum();
    const uint32_t rowBegin = (M * blk) / blkNum;
    const uint32_t rowEnd = (M * (blk + 1)) / blkNum;
    if (rowBegin >= rowEnd) {
        return;   // 该核没有分到行
    }

    // ---- 预取第 0 行 ----
    AscendC::DataCopy(xH0, xGm[rowBegin * N], N);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

    for (uint32_t i = rowBegin; i < rowEnd; ++i) {
        const uint32_t cur = (i - rowBegin) & 1u;      // 0 -> xH0, 1 -> xH1
        AscendC::LocalTensor<half> &xCur = (cur == 0) ? xH0 : xH1;
        AscendC::LocalTensor<half> &xNxt = (cur == 0) ? xH1 : xH0;
        const event_t evCur = (cur == 0) ? EVENT_ID0 : EVENT_ID1;
        const event_t evNxt = (cur == 0) ? EVENT_ID1 : EVENT_ID0;

        // 本行数据已到位
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evCur);

        // 立刻预取下一行：这次 MTE2 与下面整段 Vector 计算并行
        if (i + 1 < rowEnd) {
            AscendC::DataCopy(xNxt, xGm[(i + 1) * N], N);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evNxt);
        }

        // ---- 以下全部是本行的 Vector 计算，与上面的预取重叠 ----
        AscendC::Cast(xF, xCur, AscendC::RoundMode::CAST_NONE, N);
        // V->MTE2：本行读完后，该 buffer 才能被两行后的预取覆盖
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evCur);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evCur);

        AscendC::Muls(xT, xF, 1.0f, N);
        const float rowMax = TreeReduceMax(xT, xE, N);

        AscendC::Adds(xT, xF, -rowMax, N);
        AscendC::Exp(xE, xT, N);

        AscendC::Muls(xT, xE, 1.0f, N);
        const float rowSum = TreeReduceSum(xT, xF, N);

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
void softmax_v3_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y)
{
    softmax_v3<<<blockDim, nullptr, stream>>>(x, y);
}
#endif
