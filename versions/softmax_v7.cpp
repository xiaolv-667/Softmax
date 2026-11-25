#include "kernel_operator.h"

using AscendC::TPosition;

namespace {
constexpr uint32_t M = 1024;   // 行数
constexpr uint32_t N = 1024;   // 每行长度

// ---- 一次处理的行数（批大小）----
// v6 的瓶颈：MTE2 绝对时间恒定 7.2 µs = 25.6 行 × 每次 DataCopy 约 280 ns 固定延迟，
// MTE3 的 3.6 µs 同理。流量早在 v2 就到 4 MB 下限，所以这不是带宽而是延迟。
// 解法：一次搬 G 行（一次 DataCopy），把调用次数降到 1/G。
// G=8 时每核 25.6/8 ≈ 3.2 次 DataCopy。
// UB 预算：G*N*(2+4+4+2) 字节 + 两份 4KB 归约工作区 = 8*1024*12 + 8K = 104 KB（UB 192 KB）。
constexpr uint32_t G = 8;

constexpr uint32_t XH_ADDR = 0;                          // half[G*N]  G*2048 B
constexpr uint32_t YH_ADDR = XH_ADDR + G * N * 2;        // half[G*N]
constexpr uint32_t XF_ADDR = YH_ADDR + G * N * 2;        // float[G*N] G*4096 B
constexpr uint32_t XE_ADDR = XF_ADDR + G * N * 4;        // float[G*N]
constexpr uint32_t WT0_ADDR = XE_ADDR + G * N * 4;       // float[N]   归约 ping-pong
constexpr uint32_t WT1_ADDR = WT0_ADDR + N * 4;          // float[N]

constexpr uint32_t kAlignElems = 8;   // 32B / sizeof(float)

// 单行求和：首步直接读源、写工作区，省掉一次整行拷贝（v4 的优化，保留）
__aicore__ inline float RowSumFrom(AscendC::LocalTensor<float> &base,
                                   AscendC::LocalTensor<float> &w0,
                                   AscendC::LocalTensor<float> &w1, uint32_t len)
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
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    float s = 0.0f;
    for (uint32_t k = 0; k < len; ++k) {
        s += src->GetValue(k);
    }
    return s;
}
}

// ===================== softmax v7：批式搬运（一次搬多行）=====================
// 数据依据（v6 的 msprof，占比乘回绝对时间）：
//   Vector 5.51 us | MTE2 读 7.24 us | MTE3 写 3.61 us | Scalar 4.16 us  （总 13.33 us）
//   => MTE2+MTE3 = 10.85 us，占 81%，而它俩的绝对时间在所有版本里都恒定不动，
//      说明是"每次 DataCopy 的固定延迟 × 调用次数"，不是带宽。
//
// v7 的做法：把"逐行搬"改成"一次搬 G 行"。
//   逐元素的操作（Cast / Exp）本来就可以整批做 → 调用次数直接降到 1/G
//   只有"按行归约"和"按行归一化"必须逐行 → 保持 G 次
// 这样 MTE2/MTE3 的调用次数从 25.6 次降到约 3.2 次。
extern "C" __global__ __aicore__ void softmax_v7(GM_ADDR x, GM_ADDR y)
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

    const uint32_t blk = AscendC::GetBlockIdx();
    const uint32_t blkNum = AscendC::GetBlockNum();
    const uint32_t rowBegin = (M * blk) / blkNum;
    const uint32_t rowEnd = (M * (blk + 1)) / blkNum;

    for (uint32_t i = rowBegin; i < rowEnd; i += G) {
        // 末尾不足 G 行时按实际行数处理（避免单独写尾块分支）
        const uint32_t cnt = ((rowEnd - i) < G) ? (rowEnd - i) : G;
        const uint32_t elems = cnt * N;

        // ---- 一次搬 cnt 行 ----
        AscendC::DataCopy(xHalf, xGm[i * N], elems);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        // ---- 整批 Cast + Exp（逐元素，无跨行依赖）----
        AscendC::Cast(xF, xHalf, AscendC::RoundMode::CAST_NONE, elems);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::Exp(xE, xF, elems);

        // ---- 逐行求 sum 并归一化（只有这部分必须按行）----
        for (uint32_t r = 0; r < cnt; ++r) {
            AscendC::LocalTensor<float> srcR = xE[r * N];
            const float rowSum = RowSumFrom(srcR, wt0, wt1, N);
            AscendC::LocalTensor<float> dstR = xF[r * N];
            AscendC::Muls(dstR, srcR, 1.0f / rowSum, N);
        }

        // ---- 整批 Cast 回 fp16 并一次写回 ----
        AscendC::Cast(yHalf, xF, AscendC::RoundMode::CAST_NONE, elems);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(yGm[i * N], yHalf, elems);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }
}

#ifndef ASCENDC_CPU_DEBUG
void softmax_v7_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y)
{
    softmax_v7<<<blockDim, nullptr, stream>>>(x, y);
}
#endif
