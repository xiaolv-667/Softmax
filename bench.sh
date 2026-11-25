#!/bin/bash
###############################################################################
# softmax/bench.sh —— 跑实验并记账，不编译
#
# 依赖: 先在 softmax/ 下执行 ./build.sh
#
# 用法:
#   ./bench.sh              # blockDim=1（v0 朴素版单核）
#   ./bench.sh 40           # 多核版本
#   REPEAT=200 ./bench.sh 40
###############################################################################
set -e

CURRENT_DIR=$(cd $(dirname "${BASH_SOURCE:-$0}") && pwd)
cd "${CURRENT_DIR}"

BLOCK_DIM="${1:-${BLOCK_DIM:-1}}"
REPEAT="${REPEAT:-100}"
WARMUP="${WARMUP:-5}"
RUN_MODE="npu"
BIN="${CURRENT_DIR}/out/bin/softmax_bbit"
LIBDIR="${CURRENT_DIR}/out/lib"

# ---- 定位 CANN ----
ASCEND_INSTALL_PATH="${ASCEND_INSTALL_PATH:-${ASCEND_HOME_PATH}}"
if [ -z "${ASCEND_INSTALL_PATH}" ]; then
    if [ -d "${HOME}/Ascend/ascend-toolkit/latest" ]; then
        ASCEND_INSTALL_PATH="${HOME}/Ascend/ascend-toolkit/latest"
    else
        ASCEND_INSTALL_PATH="/usr/local/Ascend/ascend-toolkit/latest"
    fi
fi
export ASCEND_TOOLKIT_HOME="${ASCEND_INSTALL_PATH}"
export ASCEND_HOME_PATH="${ASCEND_INSTALL_PATH}"

# ---- 前置检查 ----
if [ ! -x "${BIN}" ]; then
    echo "[ERROR] 找不到可执行文件: ${BIN}"
    echo "        请先执行 ./build.sh"
    exit 1
fi
if [ ! -f "${LIBDIR}/libascendc_kernels_${RUN_MODE}.so" ]; then
    echo "[ERROR] 找不到 kernel 库: ${LIBDIR}/libascendc_kernels_${RUN_MODE}.so"
    exit 1
fi

# ---- 数据（尺寸可能变，每次重生成，seed 固定可复现）----
echo "[INFO] 生成数据 softmax (SM_M=${SM_M:-1024} x SM_N=${SM_N:-1024})"
mkdir -p input output
python3 scripts/gen_data.py

# ---- 记录实际构建配置 ----
build_soc=$(sed -n 's/^SOC_VERSION:STRING=//p' build/CMakeCache.txt 2>/dev/null | head -1)
build_type=$(sed -n 's/^CMAKE_BUILD_TYPE:STRING=//p' build/CMakeCache.txt 2>/dev/null | head -1)
build_soc="${build_soc:-unknown}"
build_type="${build_type:-unknown}"

if [ "${build_soc}" != "Ascend910B3" ]; then
    echo "[WARN] 当前构建 SOC_VERSION=${build_soc}，与本机 910B3 不一致，数据不可信！"
fi
if [ "${build_type}" != "Release" ]; then
    echo "[WARN] 当前构建 CMAKE_BUILD_TYPE=${build_type}，性能对比建议用 Release！"
fi

export LD_LIBRARY_PATH="${LIBDIR}:${LIBDIR}64:${ASCEND_INSTALL_PATH}/lib64:${LD_LIBRARY_PATH}"
export BENCH_REPEAT="${REPEAT}"
export BENCH_WARMUP="${WARMUP}"
export SM_M="${SM_M:-1024}"
export SM_N="${SM_N:-1024}"

mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="${CURRENT_DIR}/logs/bench_${STAMP}_blk${BLOCK_DIM}.log"

echo "=============================================="
echo " blockDim : ${BLOCK_DIM}"
echo " repeat   : ${REPEAT}  (warmup ${WARMUP})"
echo " 数据规模 : ${SM_M} x ${SM_N}"
echo " soc/type : ${build_soc} / ${build_type}"
echo " log      : ${LOG}"
echo "=============================================="

# 先删旧输出：避免进程异常退出时 verify 校验到上一次的结果（假阳性）
rm -f output/y.bin
"${BIN}" "${BLOCK_DIM}" 2>&1 | tee "${LOG}"

# ---- 精度校验：先判退出码，再判精度 ----
if [ ! -f output/y.bin ]; then
    echo "[FAIL] 进程未产出 output/y.bin，不做精度判定"
    verdict="**FAIL(crash)**"
else
    set +e
    python3 scripts/verify_result.py output/y.bin output/golden.bin 2>&1 | tee -a "${LOG}"
    vrc=${PIPESTATUS[0]}
    set -e
    if [ "${vrc}" = "0" ]; then verdict="PASS"; else verdict="**FAIL**"; fi
fi

# ---- 解析机读结果 ----
avg_ms=$(sed -n 's/.*avg_ms=\([0-9.]*\).*/\1/p' "${LOG}" | head -1)
eff_bw=$(sed -n 's/.*eff_bw_gbs=\([0-9.]*\).*/\1/p' "${LOG}" | head -1)
fro=$(sed -n 's/.*fro_err=\([^ ]*\).*/\1/p' "${LOG}" | head -1)
cos=$(sed -n 's/.*cos=\([^ ]*\).*/\1/p' "${LOG}" | head -1)
avg_ms="${avg_ms:-NA}"; eff_bw="${eff_bw:-NA}"
fro="${fro:-NA}"; cos="${cos:-NA}"

if git rev-parse --git-dir >/dev/null 2>&1; then
    # 加 softmax@ 前缀：本仓库同时有 matmul 和 softmax，tag 是共用的，
    # 不加前缀的话台账里分不清这行是哪个算子的数据
    version="softmax@$(git describe --tags --always 2>/dev/null || echo unknown)"
    if [ -n "$(git status --porcelain -- . ':(exclude)softmax/results.md' ':(exclude)results.md' 2>/dev/null)" ]; then
        version="${version}-dirty"
    fi
else
    version="no-git"
fi

RESULTS="${CURRENT_DIR}/results.md"
if [ ! -f "${RESULTS}" ]; then
    cat > "${RESULTS}" <<'EOF'
# Softmax 性能实验台账

**固定条件**

| 项 | 值 |
|---|---|
| 问题规模 | M = N = 1024（沿行做 softmax） |
| 数据类型 | fp16 输入/输出，fp32 中间计算 |
| 必要流量 | 读 1 次 + 写 1 次 = **4 MB**（GM 流量理论下限） |
| 精度判据 | `fro_err < 1e-3` 且 `cos > 0.9999`，且无 NaN/Inf |
| 计时方式 | `aclrtEvent` 夹住 N 次 launch，去 warmup，取平均 |
| 设备 | Ascend 910B3 |

`eff_bw_gbs` = 4 MB ÷ 单次耗时，越接近硬件带宽说明 GM 流量越接近下限。

| 时间 | 版本 | SOC/构建类型 | blockDim | repeat | avg_ms | eff_bw_gbs | fro_err | cos | 结果 |
|------|------|--------------|----------|--------|--------|-----------|---------|-----|------|
EOF
fi

printf '| %s | %s | %s/%s | %s | %s | %s | %s | %s | %s | %s |\n' \
    "$(date '+%Y-%m-%d %H:%M')" "${version}" "${build_soc}" "${build_type}" \
    "${BLOCK_DIM}" "${REPEAT}" "${avg_ms}" "${eff_bw}" "${fro}" "${cos}" "${verdict}" \
    >> "${RESULTS}"

echo ""
echo "[INFO] 已追加一行到 softmax/results.md:"
tail -1 "${RESULTS}"
