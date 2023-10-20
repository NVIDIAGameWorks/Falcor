/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "OptixUtils.h"
#include "Utils/CudaUtils.h"
#include <cuda_runtime.h>

// Apparently: this include may only appear in a single source file:
#include <optix_function_table_definition.h>

// Some debug macros
#define OPTIX_CHECK(call)                                                                                                            \
    {                                                                                                                                \
        OptixResult result = call;                                                                                                   \
        if (result != OPTIX_SUCCESS)                                                                                                 \
        {                                                                                                                            \
            FALCOR_THROW("Optix call {} failed with error {} ({}).", #call, optixGetErrorName(result), optixGetErrorString(result)); \
        }                                                                                                                            \
    }

#define CUDA_CHECK(call)                                                                                                          \
    {                                                                                                                             \
        cudaError_t result = call;                                                                                                \
        if (result != cudaSuccess)                                                                                                \
        {                                                                                                                         \
            FALCOR_THROW("CUDA call {} failed with error {} ({}).", #call, cudaGetErrorName(result), cudaGetErrorString(result)); \
        }                                                                                                                         \
    }

void optixLogCallback(unsigned int level, const char* tag, const char* message, void*)
{
    Falcor::logWarning("[Optix][{:2}][{:12}]: {}", level, tag, message);
}

// This initialization now seems verbose / excessive as CUDA and OptiX initialization
// has evolved.  TODO: Simplify?
OptixDeviceContext initOptix(Falcor::Device* pDevice)
{
    FALCOR_CHECK(pDevice->initCudaDevice(), "Failed to initialize CUDA device.");

    OPTIX_CHECK(optixInit());

    FALCOR_CHECK(g_optixFunctionTable.optixDeviceContextCreate, "OptiX function table not initialized.");

    // Build our OptiX context
    OptixDeviceContext optixContext;
    OPTIX_CHECK(optixDeviceContextCreate(pDevice->getCudaDevice()->getContext(), 0, &optixContext));

    // Tell Optix how to write to our Falcor log.
    OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, optixLogCallback, nullptr, 4));

    return optixContext;
}
