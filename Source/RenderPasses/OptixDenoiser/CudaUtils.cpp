/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "CudaUtils.h"

#include <cuda_runtime.h>
#include <sstream>

// These live in _BootstrapUtils.cpp since they use Falcor includes / namespace,
//    which does not appear to play nice with the CUDA includes / namespace.
extern void reportFatalError(std::string str);
extern void optixLogCallback(unsigned int level, const char* tag, const char* message, void*);

// Apparently: this include may only appear in a single source file:
#include <optix_function_table_definition.h>

// Some debug macros
#define CUDA_CHECK(call)							                    \
    {							                                		\
      cudaError_t rc = call;                                            \
      if (rc != cudaSuccess) {                                          \
        std::stringstream txt;                                          \
        cudaError_t err =  rc; /*cudaGetLastError();*/                  \
        txt << "CUDA Error " << cudaGetErrorName(err)                   \
            << " (" << cudaGetErrorString(err) << ")";                  \
        reportFatalError(txt.str());                                    \
      }                                                                 \
    }

#define CUDA_CHECK_NOEXCEPT(call)                                       \
    {					                                				\
      call;                                                             \
    }

#define OPTIX_CHECK( call )                                             \
  {                                                                     \
    OptixResult res = call;                                             \
    if( res != OPTIX_SUCCESS )                                          \
      {                                                                 \
        char buf[1024]; \
        sprintf( buf, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__ ); \
        reportFatalError(std::string(buf));                             \
      }                                                                 \
  }

#define CUDA_SYNC_CHECK()                                               \
  {                                                                     \
    cudaDeviceSynchronize();                                            \
    cudaError_t error = cudaGetLastError();                             \
    if( error != cudaSuccess )                                          \
      {                                                                 \
        char buf[1024]; \
        sprintf( buf, "error (%s: line %d): %s\n", __FILE__, __LINE__, cudaGetErrorString( error ) ); \
        reportFatalError(std::string(buf));                             \
      }                                                                 \
  }


unsigned int initCuda(void)
{
    cudaFree(0);
    int32_t numDevices;
    cudaGetDeviceCount(&numDevices);
    return numDevices;
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

void CudaBuffer::allocate(size_t size)
{
    if (mpDevicePtr) free();
    mSizeBytes = size;
    CUDA_CHECK(cudaMalloc((void**)&mpDevicePtr, mSizeBytes));
}

void CudaBuffer::resize(size_t size)
{
    allocate(size);
}

void CudaBuffer::free(void)
{
    CUDA_CHECK(cudaFree(mpDevicePtr));
    mpDevicePtr = nullptr;
    mSizeBytes = 0;
}

template<typename T>
bool CudaBuffer::download(T* t, size_t count)
{
    if (!mpDevicePtr) return false;
    if (mSizeBytes <= (count * sizeof(T))) return false;

    CUDA_CHECK(cudaMemcpy((void*)t, mpDevicePtr, count * sizeof(T), cudaMemcpyDeviceToHost));
    return true; // might be an error caught by CUDA_CHECK?  TODO: process any such error through
}

template<typename T>
bool CudaBuffer::upload(const T* t, size_t count)
{
    if (!mpDevicePtr) return false;
    if (mSizeBytes <= (count * sizeof(T))) return false;

    CUDA_CHECK(cudaMemcpy(mpDevicePtr, (void*)t, count * sizeof(T), cudaMemcpyHostToDevice));
    return true; // might be an error caught by CUDA_CHECK?  TODO: process any such error through
}

template<typename T>
void CudaBuffer::allocAndUpload(const std::vector<T>& vt)
{
    allocate(vt.size() * sizeof(T));
    upload((const T*)vt.data(), vt.size());
}


void* getSharedDevicePtr(HANDLE sharedHandle, uint32_t bytes)
{
    // No handle?  No pointer!
    if (sharedHandle == NULL) return nullptr;

    // Create the descriptor of our shared memory buffer
    cudaExternalMemoryHandleDesc externalMemoryHandleDesc;
    memset(&externalMemoryHandleDesc, 0, sizeof(externalMemoryHandleDesc));
    externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    externalMemoryHandleDesc.handle.win32.handle = sharedHandle;
    externalMemoryHandleDesc.size = bytes;
    externalMemoryHandleDesc.flags = cudaExternalMemoryDedicated;

    // Get a handle to that memory
    cudaExternalMemory_t externalMemory;
    CUDA_CHECK(cudaImportExternalMemory(&externalMemory, &externalMemoryHandleDesc));

    // Create a descriptor for our shared buffer pointer
    cudaExternalMemoryBufferDesc bufDesc;
    memset(&bufDesc, 0, sizeof(bufDesc));
    bufDesc.size = bytes;

    // Actually map the buffer
    void* devPtr = nullptr;
    CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&devPtr, externalMemory, &bufDesc));
    return devPtr;
}

bool freeSharedDevicePtr(void* ptr)
{
    if (!ptr) return false;
    return cudaSuccess == cudaFree(ptr);
}
