#!/bin/sh

# $1 -> Project directory
# $2 -> Binary output directory
# $3 -> Build configuration
# $4 -> Slang directory
# $5 -> DLSS directory

EXT_DIR=$1/external/packman/
OUT_DIR=$2

IS_DEBUG=false
if [ "$3" = "Debug" ]; then
    IS_DEBUG=true
fi
SLANG_DIR=$4

# Copy externals
if [ "${IS_DEBUG}" = false ]; then
    cp -frp ${EXT_DIR}/deps/lib/*.so* ${OUT_DIR}
else
    cp -frp ${EXT_DIR}/deps/debug/lib/*.so* ${OUT_DIR}
fi

cp -fp ${EXT_DIR}/python/lib/libpython*.so* ${OUT_DIR}
mkdir -p ${OUT_DIR}/pythondist
cp -frp ${EXT_DIR}/python/* ${OUT_DIR}/pythondist

# Copy slang
cp -f ${SLANG_DIR}/lib/lib*.so ${OUT_DIR}

# Copy CUDA
CUDA_DIR=${EXT_DIR}/cuda
if [ -d ${CUDA_DIR} ]; then
    cp -fp ${CUDA_DIR}/lib64/libcudart.so* ${OUT_DIR}
    cp -fp ${CUDA_DIR}/lib64/libnvrtc.so* ${OUT_DIR}
    cp -fp ${CUDA_DIR}/lib64/libcublas.so* ${OUT_DIR}
    cp -fp ${CUDA_DIR}/lib64/libcurand.so* ${OUT_DIR}
fi

# Copy Aftermath
AFTERMATH_DIR=${EXT_DIR}/aftermath
if [ -d ${AFTERMATH_DIR} ]; then
    cp -fp ${AFTERMATH_DIR}/lib/x64/libGFSDK_Aftermath_Lib.x64.so ${OUT_DIR}
fi

# Copy RTXDI SDK shaders
RTXDI_DIR=${EXT_DIR}/rtxdi/rtxdi-sdk/include/rtxdi
RTXDI_TARGET_DIR=${OUT_DIR}/shaders/rtxdi
if [ -d ${RTXDI_DIR} ]; then
    mkdir -p ${RTXDI_TARGET_DIR}
    cp ${RTXDI_DIR}/ResamplingFunctions.hlsli ${RTXDI_TARGET_DIR}
    cp ${RTXDI_DIR}/Reservoir.hlsli ${RTXDI_TARGET_DIR}
    cp ${RTXDI_DIR}/RtxdiHelpers.hlsli ${RTXDI_TARGET_DIR}
    cp ${RTXDI_DIR}/RtxdiMath.hlsli ${RTXDI_TARGET_DIR}
    cp ${RTXDI_DIR}/RtxdiParameters.h ${RTXDI_TARGET_DIR}
    cp ${RTXDI_DIR}/RtxdiTypes.h ${RTXDI_TARGET_DIR}
fi

# Copy NanoVDB
NANOVDB_DIR=${EXT_DIR}/nanovdb
NANOVDB_TARGET_DIR=${OUT_DIR}/shaders/nanovdb
if [ -d ${NANOVDB_DIR} ]; then
    mkdir -p ${NANOVDB_TARGET_DIR}
    cp ${NANOVDB_DIR}/include/nanovdb/PNanoVDB.h ${NANOVDB_TARGET_DIR}
fi

# Copy USD
if [ "${IS_DEBUG}" = false ]; then
    cp -fp ${EXT_DIR}/nv-usd-release/lib/libusd_ms.so ${OUT_DIR}
    cp -frp ${EXT_DIR}/nv-usd-release/lib/usd ${OUT_DIR}/usd
else
    cp -fp ${EXT_DIR}/nv-usd-debug/lib/libusd_ms.so ${OUT_DIR}
    cp -frp ${EXT_DIR}/nv-usd-debug/lib/usd ${OUT_DIR}/usd
fi

# Copy MDL
MDL_DIR=${EXT_DIR}/mdl-sdk
if [ -d ${MDL_DIR} ]; then
    cp -fp ${MDL_DIR}/linux-x86-64/lib/*.so* ${OUT_DIR}
    mkdir -p ${OUT_DIR}/mdl/nvidia
    cp -frp ${MDL_DIR}/examples/mdl/nvidia/core* ${OUT_DIR}/mdl/nvidia
fi

# Copy NVTT
cp ${EXT_DIR}/nvtt/libcudart.so.11.0 ${OUT_DIR}
cp ${EXT_DIR}/nvtt/libnvtt.so ${OUT_DIR}
