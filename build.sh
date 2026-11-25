#!/bin/bash
###############################################################################
# softmax/build.sh —— 只负责编译，不跑实验
#
# 与根目录的 Matmul 构建完全独立（各自的 build/ 和 out/）。
#
# 用法:
#   ./build.sh                      # 增量编译，910B3 + Release
#   ./build.sh -c                   # 全量重编
###############################################################################
set -e

CURRENT_DIR=$(cd $(dirname "${BASH_SOURCE:-$0}") && pwd)
cd "${CURRENT_DIR}"

SOC_VERSION="${SOC_VERSION:-Ascend910B3}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
RUN_MODE="npu"
INSTALL_PREFIX="${INSTALL_PREFIX:-${CURRENT_DIR}/out}"
ASCEND_INSTALL_PATH="${ASCEND_INSTALL_PATH:-}"
CLEAN=0

SHORT=r:,v:,i:,b:,p:,n:,c
LONG=run-mode:,soc-version:,install-path:,build-type:,install-prefix:,kernel-ver:,clean
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

KERNEL_VER="${KERNEL_VER:-5}"

while :; do
    case "$1" in
    -r | --run-mode)       RUN_MODE="$2"; shift 2 ;;
    -v | --soc-version)    SOC_VERSION="$2"; shift 2 ;;
    -i | --install-path)   ASCEND_INSTALL_PATH="$2"; shift 2 ;;
    -b | --build-type)     BUILD_TYPE="$2"; shift 2 ;;
    -p | --install-prefix) INSTALL_PREFIX="$2"; shift 2 ;;
    -n | --kernel-ver)     KERNEL_VER="$2"; shift 2 ;;
    -c | --clean)          CLEAN=1; shift ;;
    --)                    shift; break ;;
    *) echo "[ERROR] Unexpected option: $1"; exit 1 ;;
    esac
done

# ---- 定位 CANN ----
if [ -z "${ASCEND_INSTALL_PATH}" ]; then
    if [ -n "${ASCEND_HOME_PATH}" ]; then
        ASCEND_INSTALL_PATH="${ASCEND_HOME_PATH}"
    elif [ -d "${HOME}/Ascend/ascend-toolkit/latest" ]; then
        ASCEND_INSTALL_PATH="${HOME}/Ascend/ascend-toolkit/latest"
    else
        ASCEND_INSTALL_PATH="/usr/local/Ascend/ascend-toolkit/latest"
    fi
fi

if [ ! -f "${ASCEND_INSTALL_PATH}/bin/setenv.bash" ]; then
    echo "[ERROR] 找不到 CANN 环境脚本: ${ASCEND_INSTALL_PATH}/bin/setenv.bash"
    echo "        请用 -i 指定 CANN 安装路径，或导出 ASCEND_HOME_PATH"
    exit 1
fi

export ASCEND_TOOLKIT_HOME="${ASCEND_INSTALL_PATH}"
export ASCEND_HOME_PATH="${ASCEND_INSTALL_PATH}"
set +e
source "${ASCEND_INSTALL_PATH}/bin/setenv.bash"
set -e

echo "=============================================="
echo " SOFTMAX 子项目"
echo " SOC_VERSION : ${SOC_VERSION}"
echo " BUILD_TYPE  : ${BUILD_TYPE}"
echo " KERNEL_VER  : v${KERNEL_VER}"
echo " CANN        : ${ASCEND_INSTALL_PATH}"
echo " CLEAN BUILD : ${CLEAN}"
echo "=============================================="

if [ "${CLEAN}" = "1" ]; then
    echo "[INFO] 全量重编，清理 build/ 与 out/"
    rm -rf build out
fi

mkdir -p build
cmake -B build \
    -DRUN_MODE="${RUN_MODE}" \
    -DSOC_VERSION="${SOC_VERSION}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DSOFTMAX_VER="${KERNEL_VER}" \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_INSTALL_PATH}"

cmake --build build -j"$(nproc)"
cmake --install build

echo "[INFO] 编译完成:"
ls -l "${INSTALL_PREFIX}/bin/softmax_bbit" "${INSTALL_PREFIX}/lib/libascendc_kernels_${RUN_MODE}.so"
