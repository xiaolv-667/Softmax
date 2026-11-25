#!/bin/bash
###############################################################################
# softmax/scripts/sweep_prof.sh —— 逐版采集 msprof，产物归档到 docs/prof/v<N>/
#
# 用途：给 softmax 的每个优化版本补一份 profiling 证据，
#       让"瓶颈定位 -> 解决 -> 再看瓶颈"这条链每步都有数据支撑。
#
# 用法:
#   ./scripts/sweep_prof.sh            # 跑 v0..v5
#   ./scripts/sweep_prof.sh 2 5        # 只跑指定版本
#   BLOCK_DIM=1 ./scripts/sweep_prof.sh 0   # 单核（v0 原始配置）
###############################################################################
set -e

CURRENT_DIR=$(cd $(dirname "${BASH_SOURCE:-$0}")/.. && pwd)
cd "${CURRENT_DIR}"

BLOCK_DIM="${BLOCK_DIM:-40}"
if [ "$#" -gt 0 ]; then
    VERS="$*"
else
    VERS="0 1 2 3 4 5"
fi

echo "################################################################"
echo "# 逐版 profiling: v${VERS// / v}   blockDim=${BLOCK_DIM}"
echo "################################################################"

for v in ${VERS}; do
    echo ""
    echo "================= v${v} ================="

    if ! ./build.sh -c -n "${v}" > "/tmp/sm_build_v${v}.log" 2>&1; then
        echo "[FAIL] v${v} 编译失败，跳过"
        tail -5 "/tmp/sm_build_v${v}.log"
        continue
    fi

    if ! ./scripts/prof.sh "${BLOCK_DIM}" PipeUtilization > "/tmp/sm_prof_v${v}.log" 2>&1; then
        echo "[FAIL] v${v} profiling 失败，跳过"
        tail -5 "/tmp/sm_prof_v${v}.log"
        continue
    fi

    d=$(sed -n 's/^PROF_DIR=//p' "/tmp/sm_prof_v${v}.log" | tail -1)
    if [ -z "${d}" ] || [ ! -d "${d}" ]; then
        echo "[FAIL] v${v} 未取到输出目录"
        continue
    fi

    mkdir -p "docs/prof/v${v}"
    find "${d}" -name "PipeUtilization.csv" -exec cp {} "docs/prof/v${v}/" \; 2>/dev/null
    find "${d}" -name "OpBasicInfo.csv" -exec cp {} "docs/prof/v${v}/" \; 2>/dev/null
    echo "[OK] v${v} -> docs/prof/v${v}/  ($(ls docs/prof/v${v} | tr '\n' ' '))"
done

echo ""
echo "################################################################"
echo "# 汇总（AIV 流水线占空比）"
echo "################################################################"
printf "%-4s %-10s %-8s %-8s %-8s %-8s\n" "ver" "time(us)" "Vector" "MTE2读" "MTE3写" "Scalar"
for v in ${VERS}; do
    csv="docs/prof/v${v}/PipeUtilization.csv"
    [ -f "${csv}" ] || continue
    awk -F, -v ver="v${v}" '
    NR==1{for(i=1;i<=NF;i++)h[$i]=i; next}
    { if (($h["aiv_time(us)"] "") !~ /^[0-9.]+$/) next
      n++; t=$h["aiv_time(us)"]+0
      sv+=$h["aiv_vec_ratio"]+0; sm2+=$h["aiv_mte2_ratio"]+0
      sm3+=$h["aiv_mte3_ratio"]+0; ss+=$h["aiv_scalar_ratio"]+0 }
    END{ if(n>0) printf "%-4s %-10.2f %-8.1f %-8.1f %-8.1f %-8.1f\n",
              ver, t, sv/n*100, sm2/n*100, sm3/n*100, ss/n*100 }' "${csv}"
done
