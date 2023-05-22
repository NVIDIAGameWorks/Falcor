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
#define OPTIX_CHECK( call )                                             \
  {                                                                     \
    OptixResult res = call;                                             \
    if( res != OPTIX_SUCCESS )                                          \
      {                                                                 \
        char buf[1024]; \
        sprintf( buf, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__ ); \
        Falcor::reportFatalError(std::string(buf));                     \
      }                                                                 \
  }
#define CUDA_CHECK(call)							                    \
    {							                                		\
      cudaError_t rc = call;                                            \
      if (rc != cudaSuccess) {                                          \
        std::stringstream txt;                                          \
        cudaError_t err =  rc; /*cudaGetLastError();*/                  \
        txt << "CUDA Error " << cudaGetErrorName(err)                   \
            << " (" << cudaGetErrorString(err) << ")";                  \
        Falcor::reportFatalError(txt.str());                            \
      }                                                                 \
    }

void optixLogCallback(unsigned int level, const char* tag, const char* message, void*)
{
    Falcor::logWarning("[Optix][{:2}][{:12}]: {}", level, tag, message);
}

// This initialization now seems verbose / excessive as CUDA and OptiX initialization
// has evolved.  TODO: Simplify?
bool initOptix(OptixDeviceContext& optixContext)
{
    // Initialize CUDA
    uint32_t devices = initCuda();
    if (devices <= 0) return false;

    // Initialize Optix.
    OPTIX_CHECK(optixInit());

    // Check if we have a valid OptiX function table.  If not, return now.
    if (!g_optixFunctionTable.optixDeviceContextCreate) return false;

    // Setup which device to work on.  Hard coded to device #0
    int32_t deviceId = 0;
    CUDA_CHECK(cudaSetDevice(deviceId));

    // Create a CUDA stream
    CUstream stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Get device information
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, deviceId);

    // Get the current context
    CUcontext cudaContext;
    CUresult cuRes = cuCtxGetCurrent(&cudaContext);

    // Build our OptiX context
    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));

    // Given the OPTIX_CHECK() and CUDA_CHECK() wrappers above that simply log errors,
    // explicitly check for successful initialization by seeing if we have a non-null context
    if (optixContext == nullptr) return false;

    // Tell Optix how to write to our Falcor log.
    OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext,
        optixLogCallback, nullptr, 4));

    return true;
}
