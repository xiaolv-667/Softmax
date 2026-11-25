#!/bin/bash
###############################################################################
# softmax/scripts/prof.sh —— 用 msprof op 采集 softmax 单算子性能数据
#
# 用法:
#   ./scripts/prof.sh                  # blockDim=40 + PipeUtilization（最常用）
#   ./scripts/prof.sh 1                # 单核（对应 v0 的核数配置）
#   ./scripts/prof.sh 40 Memory        # 看各级存储流量
#   ./scripts/prof.sh 40 ResourceConflictRatio   # 看同步/资源冲突
#
# 注意（都是踩过的坑）:
#   1. msprof 的 --application 只接受可执行文件路径、不能带参数，
#      所以自动生成一个把 blockDim 固化进去的包装器。
#   2. 必须 skip 掉程序自己的 warmup 发射，否则采到的是预热那一发。
#   3. profiling 有额外开销，程序自打印的 avg_ms 不可用于性能对比；
#      权威时长看 OpBasicInfo.csv 的 Task Duration(us)。
#   4. 采之前确认内核版本：本脚本不会改构建，请先用 ./build.sh -c -n <v> 选好版本。
###############################################################################
set -e

CURRENT_DIR=$(cd $(dirname "${BASH_SOURCE:-$0}")/.. && pwd)
cd "${CURRENT_DIR}"

BLOCK_DIM="${1:-${BLOCK_DIM:-40}}"
METRICS="${2:-${AIC_METRICS:-PipeUtilization}}"
REPEAT="${REPEAT:-5}"
WARMUP="${WARMUP:-2}"

BIN="${CURRENT_DIR}/out/bin/softmax_bbit"
if [ ! -x "${BIN}" ]; then
    echo "[ERROR] 找不到可执行文件: ${BIN}，请先执行 ./build.sh -c -n <版本>"
    exit 1
fi

# 当前构建的是哪个版本（从 CMakeCache 读，防止采错版本）
CUR_VER=$(sed -n 's/^SOFTMAX_VER:STRING=//p' build/CMakeCache.txt 2>/dev/null | head -1)
CUR_VER="${CUR_VER:-unknown}"

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
set +e
source "${ASCEND_INSTALL_PATH}/bin/setenv.bash" >/dev/null 2>&1
set -e

STAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="${CURRENT_DIR}/prof/${STAMP}_v${CUR_VER}_blk${BLOCK_DIM}_${METRICS//,/_}"
mkdir -p "${OUTDIR}"

WRAPPER="${OUTDIR}/run_blk${BLOCK_DIM}.sh"
cat > "${WRAPPER}" <<EOF
#!/bin/bash
# 自动生成：msprof 启动入口，固定 blockDim=${BLOCK_DIM}
cd "${CURRENT_DIR}"
export LD_LIBRARY_PATH="${CURRENT_DIR}/out/lib:${CURRENT_DIR}/out/lib64:${ASCEND_INSTALL_PATH}/lib64:\${LD_LIBRARY_PATH}"
exec "${BIN}" ${BLOCK_DIM}
EOF
chmod +x "${WRAPPER}"

export BENCH_REPEAT="${REPEAT}"
export BENCH_WARMUP="${WARMUP}"

echo "=============================================="
echo " kernel ver  : v${CUR_VER}"
echo " blockDim    : ${BLOCK_DIM}"
echo " aic-metrics : ${METRICS}"
echo " repeat      : ${REPEAT} (warmup ${WARMUP})"
echo " 输出目录    : ${OUTDIR}"
echo "=============================================="

msprof op \
    --application="${WRAPPER}" \
    --output="${OUTDIR}" \
    --aic-metrics="${METRICS}" \
    --launch-count=1 \
    --launch-skip-before-match="${WARMUP}" \
    --replay-mode=kernel

echo ""
echo "PROF_DIR=${OUTDIR}"
