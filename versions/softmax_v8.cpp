#include "kernel_operator.h"

using AscendC::TPosition;

namespace {
constexpr uint32_t M = 1024;   // 行数
constexpr uint32_t N = 1024;   // 每行长度

// 一次处理的行数（v7 引入的批式搬运，保留）
constexpr uint32_t G = 8;

constexpr uint32_t XH_ADDR = 0;                          // half[G*N]  G*2048 B
constexpr uint32_t YH_ADDR = XH_ADDR + G * N * 2;        // half[G*N]
constexpr uint32_t XF_ADDR = YH_ADDR + G * N * 2;        // float[G*N] G*4096 B
constexpr uint32_t XE_ADDR = XF_ADDR + G * N * 4;        // float[G*N]
constexpr uint32_t WT0_ADDR = XE_ADDR + G * N * 4;       // float[N]   归约 ping-pong
constexpr uint32_t WT1_ADDR = WT0_ADDR + N * 4;          // float[N]
constexpr uint32_t WT2_ADDR = WT1_ADDR + N * 4;          // float[32]  WholeReduce 输出
// 合计约 104 KB（UB 192 KB）

constexpr uint32_t kAlignElems = 8;   // 32B / sizeof(float)

// ---------------------------------------------------------------
// 单行求和。相对 v7 的改动：最后 8 个部分和不再用 8 次标量 GetValue 累加，
// 而是一句 WholeReduceSum（mask 版本）归约成 1 个，再读 1 次标量。
//
// 依据（v7 的 msprof，占比乘回绝对时间）：
//   Vector 4.70us | MTE2 1.63us | MTE3 0.83us | Scalar 4.68us（总 10.54us）
//   => MTE2/MTE3 已被压到 2.5us，瓶颈转为 Scalar(44.4%)+Vector(44.6%)，
//      而 Scalar 的大头就是每行的标量收尾：25.6 行 x 8 次 = 约 205 次/核。
// ---------------------------------------------------------------
__aicore__ inline float RowSumFrom(AscendC::LocalTensor<float> &base,
                                   AscendC::LocalTensor<float> &w0,
                                   AscendC::LocalTensor<float> &w1,
                                   AscendC::LocalTensor<float> &redDst, uint32_t len)
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
    // 剩 8 个部分和：一次向量归约 + 一次标量读
    uint64_t mask[1] = {0xFFULL};          // 低 8 位 <=> 前 8 个 fp32 元素
    AscendC::WholeReduceSum(redDst, *src, mask, 1, 1, 1, 1);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    return redDst.GetValue(0);
}
}

// ===================== softmax v8：归约收尾向量化 =====================
// v7 = 批式搬运（一次 DataCopy 搬 G=8 行），把 MTE2/MTE3 从 7.24/3.61us 压到 1.63/0.83us。
// v8 在此基础上压 Scalar：每行的标量收尾从 8 次 GetValue 降为 1 次。
// 每核标量读次数约从 205 降到 26。
extern "C" __global__ __aicore__ void softmax_v8(GM_ADDR x, GM_ADDR y)
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
    AscendC::LocalTensor<float> wt2  (TPosition::VECCALC, WT2_ADDR, 32);

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

        for (uint32_t r = 0; r < cnt; ++r) {
            AscendC::LocalTensor<float> srcR = xE[r * N];
            const float rowSum = RowSumFrom(srcR, wt0, wt1, wt2, N);
            AscendC::LocalTensor<float> dstR = xF[r * N];
            AscendC::Muls(dstR, srcR, 1.0f / rowSum, N);
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
void softmax_v8_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y)
{
    softmax_v8<<<blockDim, nullptr, stream>>>(x, y);
}
#endif
