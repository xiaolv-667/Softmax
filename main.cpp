#include "data_utils.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"

#include <cstdlib>

// 由 CMake 传入 SOFTMAX_VER 选择内核版本（./build.sh -n <0..5>）
#ifndef SOFTMAX_VER
#define SOFTMAX_VER 5
#endif

#if SOFTMAX_VER == 0
extern void softmax_v0_do(uint32_t, void *, uint8_t *, uint8_t *);
#define SOFTMAX_DO softmax_v0_do
#elif SOFTMAX_VER == 1
extern void softmax_v1_do(uint32_t, void *, uint8_t *, uint8_t *);
#define SOFTMAX_DO softmax_v1_do
#elif SOFTMAX_VER == 2
extern void softmax_v2_do(uint32_t, void *, uint8_t *, uint8_t *);
#define SOFTMAX_DO softmax_v2_do
#elif SOFTMAX_VER == 3
extern void softmax_v3_do(uint32_t, void *, uint8_t *, uint8_t *);
#define SOFTMAX_DO softmax_v3_do
#elif SOFTMAX_VER == 4
extern void softmax_v4_do(uint32_t, void *, uint8_t *, uint8_t *);
#define SOFTMAX_DO softmax_v4_do
#elif SOFTMAX_VER == 5
extern void softmax_v5_do(uint32_t, void *, uint8_t *, uint8_t *);
#define SOFTMAX_DO softmax_v5_do
#elif SOFTMAX_VER == 6
extern void softmax_v6_do(uint32_t, void *, uint8_t *, uint8_t *);
#define SOFTMAX_DO softmax_v6_do
#elif SOFTMAX_VER == 7
extern void softmax_v7_do(uint32_t, void *, uint8_t *, uint8_t *);
#define SOFTMAX_DO softmax_v7_do
#elif SOFTMAX_VER == 8
extern void softmax_v8_do(uint32_t, void *, uint8_t *, uint8_t *);
#define SOFTMAX_DO softmax_v8_do
#elif SOFTMAX_VER == 9
extern void softmax_v9_do(uint32_t, void *, uint8_t *, uint8_t *);
#define SOFTMAX_DO softmax_v9_do
#else
#error "SOFTMAX_VER must be in 0..9"
#endif

namespace {
// 从环境变量读正整数，读不到或非法则用默认值
int EnvInt(const char *name, int fallback)
{
    const char *v = getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return fallback;
    }
    const int n = atoi(v);
    return (n > 0) ? n : fallback;
}
}

int32_t main(int32_t argc, char **argv)
{
    // blockDim 由命令行第 1 个参数指定
    uint32_t blockDim = 1;
    if (argc > 1) {
        const int v = atoi(argv[1]);
        if (v > 0) {
            blockDim = static_cast<uint32_t>(v);
        }
    }
    const int repeat = EnvInt("BENCH_REPEAT", 100);
    const int warmup = EnvInt("BENCH_WARMUP", 5);

    const uint32_t M = 1024, N = 1024;
    size_t bytes = (size_t)M * N * 2;   // fp16（ReadFile 的文件长度参数需要非 const 引用）

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *xHost = nullptr, *yHost = nullptr;
    CHECK_ACL(aclrtMallocHost((void **)&xHost, bytes));
    CHECK_ACL(aclrtMallocHost((void **)&yHost, bytes));
    ReadFile("./input/x.bin", bytes, xHost, bytes);

    uint8_t *xDev = nullptr, *yDev = nullptr;
    CHECK_ACL(aclrtMalloc((void **)&xDev, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&yDev, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(xDev, bytes, xHost, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    // 输出先清零：若 kernel 没写满，能立刻看出来（这个诊断手段在 Matmul 项目里立过功）
    CHECK_ACL(aclrtMemset(yDev, bytes, 0, bytes));

    aclrtEvent startEv = nullptr, endEv = nullptr;
    CHECK_ACL(aclrtCreateEvent(&startEv));
    CHECK_ACL(aclrtCreateEvent(&endEv));

    // 预热：首次 launch 含 kernel 库加载等一次性开销
    for (int i = 0; i < warmup; ++i) {
        SOFTMAX_DO(blockDim, stream, xDev, yDev);
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtRecordEvent(startEv, stream));
    for (int it = 0; it < repeat; ++it) {
        SOFTMAX_DO(blockDim, stream, xDev, yDev);
    }
    CHECK_ACL(aclrtRecordEvent(endEv, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    float totalMs = 0.0f;
    CHECK_ACL(aclrtEventElapsedTime(&totalMs, startEv, endEv));
    const double avgMs = static_cast<double>(totalMs) / repeat;

    // 有效带宽：只算"必要"的流量（读 1 次 + 写 1 次 = 4 MB），
    // 用它和 v0 实际的 3读+1写 对比，能直观看出还有几倍优化空间。
    const double minBytes = 2.0 * M * N * 2.0;                 // 4 MB
    const double effBw = minBytes / (avgMs * 1e-3) / 1e9;      // GB/s

    printf("[BENCH] block=%u repeat=%d warmup=%d avg_ms=%.6f total_ms=%.3f "
           "min_traffic_mb=%.2f eff_bw_gbs=%.2f\n",
           blockDim, repeat, warmup, avgMs, static_cast<double>(totalMs),
           minBytes / (1024.0 * 1024.0), effBw);

    CHECK_ACL(aclrtMemcpy(yHost, bytes, yDev, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/y.bin", yHost, bytes);

    CHECK_ACL(aclrtDestroyEvent(startEv));
    CHECK_ACL(aclrtDestroyEvent(endEv));
    CHECK_ACL(aclrtFree(xDev));
    CHECK_ACL(aclrtFree(yDev));
    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(yHost));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    return 0;
}
#else
int32_t main() { return 0; }
#endif
