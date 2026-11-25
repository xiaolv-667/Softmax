#include "kernel_operator.h"

using AscendC::TPosition;

namespace {
constexpr uint32_t M = 1024;   // 行数
constexpr uint32_t N = 1024;   // 每行长度
constexpr uint32_t G = 8;      // 一次处理的行数（v7 的批式搬运）

constexpr uint32_t XH_ADDR = 0;                          // half[G*N]
constexpr uint32_t YH_ADDR = XH_ADDR + G * N * 2;        // half[G*N]
constexpr uint32_t XF_ADDR = YH_ADDR + G * N * 2;        // float[G*N]
constexpr uint32_t XE_ADDR = XF_ADDR + G * N * 4;        // float[G*N]
constexpr uint32_t WT0_ADDR = XE_ADDR + G * N * 4;       // float[N]   归约 ping-pong
constexpr uint32_t WT1_ADDR = WT0_ADDR + N * 4;          // float[N]
// 归约结果槽：每行一个槽，每槽间隔 8 个 float（32 字节），
// 保证 WholeReduceSum 的 dst 满足 32B 对齐（r*32 字节恒 32B 对齐）
constexpr uint32_t SLOT_ADDR = WT1_ADDR + N * 4;         // float[G*8]
constexpr uint32_t SLOT_STRIDE = 8;

constexpr uint32_t kAlignElems = 8;   // 32B / sizeof(float)

// 把一行归约到 dstSlot（1 个 float），**不做 V_S 同步**。
// 调用方把所有行都算完之后统一 sync 一次，避免每行一次强制排空。
__aicore__ inline void ReduceRowToSlot(AscendC::LocalTensor<float> &base,
                                       AscendC::LocalTensor<float> &w0,
                                       AscendC::LocalTensor<float> &w1,
                                       AscendC::LocalTensor<float> &dstSlot,
                                       uint32_t len)
{
    uint32_t half = len >> 1;
    AscendC::Add(w0, base, base[half], static_cast<int32_t>(half));
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
    uint64_t mask[1] = {0xFFULL};   // 低 8 位 <=> 剩余 8 个 fp32 元素
    AscendC::WholeReduceSum(dstSlot, *src, mask, 1, 1, 1, 1);
}
}

// ===================== softmax v9：批量归约 + 单次同步 =====================
// 依据（v7 的 msprof，占比乘回绝对时间）：
//   Vector 4.70us | MTE2 1.63us | MTE3 0.83us | Scalar 4.68us（总 10.54us）
//   MTE2/MTE3 已压到 2.46us，瓶颈转为 Vector + Scalar，而两者都围绕"每行归约"。
//
// v7 的问题：每批 8 行，每行归约完立刻 SetFlag/WaitFlag(V_S) 一次再去读标量，
//   于是每批 8 次强制排空，每核约 3.2 批 x 8 = 26 次。V_S 要求 Vector 流水完全排空，
//   这个停顿很可能就是 Scalar 那一项的大头。
// v9 的做法：先把 8 行全部归约到各自的槽（纯向量操作，无需同步），
//   再统一做一次 V_S，一次读 8 个标量，然后批量归一化。
//   同步次数 26 -> 约 3.2 次/核。
extern "C" __global__ __aicore__ void softmax_v9(GM_ADDR x, GM_ADDR y)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();

    AscendC::GlobalTensor<half> xGm, yGm;
    xGm.SetGlobalBuffer((__gm__ half *)x, M * N);
    yGm.SetGlobalBuffer((__gm__ half *)y, M * N);

    AscendC::LocalTensor<half>  xHalf(TPosition::VECCALC, XH_ADDR, G * N);
    AscendC::LocalTensor<half>  yHalf(TPosition::VECCALC, YH_ADDR, G * N);
    AscendC::LocalTensor<float> xF   (TPosition::VECCALC, XF_ADDR, G * N);
    AscendC::LocalTensor<float> xE   (TPosition::VECCALC, XE_ADDR, G * N);
    AscendC::LocalTensor<float> wt0  (TPosition::VECCALC, WT0_ADDR, N);
    AscendC::LocalTensor<float> wt1  (TPosition::VECCALC, WT1_ADDR, N);
    AscendC::LocalTensor<float> slot (TPosition::VECCALC, SLOT_ADDR, G * SLOT_STRIDE);

    const uint32_t blk = AscendC::GetBlockIdx();
    const uint32_t blkNum = AscendC::GetBlockNum();
    const uint32_t rowBegin = (M * blk) / blkNum;
    const uint32_t rowEnd = (M * (blk + 1)) / blkNum;

    for (uint32_t i = rowBegin; i < rowEnd; i += G) {
        const uint32_t cnt = ((rowEnd - i) < G) ? (rowEnd - i) : G;
        const uint32_t elems = cnt * N;

        AscendC::DataCopy(xHalf, xGm[i * N], elems);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        AscendC::Cast(xF, xHalf, AscendC::RoundMode::CAST_NONE, elems);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::Exp(xE, xF, elems);

        // ---- 8 行全部归约到各自槽，中间不做任何 V_S 同步 ----
        for (uint32_t r = 0; r < cnt; ++r) {
            AscendC::LocalTensor<float> srcR = xE[r * N];
            AscendC::LocalTensor<float> dstSlot = slot[r * SLOT_STRIDE];
            ReduceRowToSlot(srcR, wt0, wt1, dstSlot, N);
        }

        // ---- 统一同步一次，一次读回 cnt 个行和 ----
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        float sums[G];
        for (uint32_t r = 0; r < cnt; ++r) {
            sums[r] = slot[r * SLOT_STRIDE].GetValue(0);
        }

        // ---- 批量归一化 ----
        for (uint32_t r = 0; r < cnt; ++r) {
            AscendC::LocalTensor<float> srcR = xE[r * N];
            AscendC::LocalTensor<float> dstR = xF[r * N];
            AscendC::Muls(dstR, srcR, 1.0f / sums[r], N);
        }

        AscendC::Cast(yHalf, xF, AscendC::RoundMode::CAST_NONE, elems);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(yGm[i * N], yHalf, elems);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }
}

#ifndef ASCENDC_CPU_DEBUG
void softmax_v9_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y)
{
    softmax_v9<<<blockDim, nullptr, stream>>>(x, y);
}
#endif
