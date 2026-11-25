cmake_minimum_required(VERSION 3.16)
project(Ascend_softmax)

# Softmax 子项目：与根目录的 Matmul 构建相互独立，互不影响。
# 用法：cd softmax && ./build.sh

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend910B3" CACHE STRING "system on chip type")

if(DEFINED ENV{ASCEND_HOME_PATH} AND NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
    set(_cann_default "$ENV{ASCEND_HOME_PATH}")
elseif(EXISTS "$ENV{HOME}/Ascend/ascend-toolkit/latest")
    set(_cann_default "$ENV{HOME}/Ascend/ascend-toolkit/latest")
else()
    set(_cann_default "/usr/local/Ascend/ascend-toolkit/latest")
endif()
set(ASCEND_CANN_PACKAGE_PATH "${_cann_default}"
    CACHE STRING "ASCEND CANN package installation directory"
)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release/Debug" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
endif()

# 内核版本：./build.sh -n <0..5> 切换，便于逐版做 msprof 对比
set(SOFTMAX_VER 5 CACHE STRING "which softmax kernel version (0..5)")

# 当前版本对应的 kernel 源文件
file(GLOB KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/softmax_v${SOFTMAX_VER}.cpp
)

set(CUSTOM_ASCEND310P_LIST "Ascend310P1" "Ascend310P3")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/npu_lib.cmake)

add_executable(softmax_bbit
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_compile_options(softmax_bbit PRIVATE
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Werror
)

target_compile_definitions(softmax_bbit PRIVATE
    SOC_VERSION="${SOC_VERSION}"
    SOFTMAX_VER=${SOFTMAX_VER}
)

target_link_libraries(softmax_bbit PRIVATE
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
    ascendc_kernels_${RUN_MODE}
    register
    platform
    ascendalog
    dl
)

install(TARGETS softmax_bbit
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
