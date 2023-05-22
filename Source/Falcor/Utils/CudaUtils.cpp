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
#include "CudaUtils.h"
#include "Core/Errors.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <sstream>

// Some debug macros
#define CUDA_CHECK(call)                                                                             \
    {                                                                                                \
        cudaError_t rc = call;                                                                       \
        if (rc != cudaSuccess)                                                                       \
        {                                                                                            \
            std::stringstream txt;                                                                   \
            cudaError_t err = rc; /*cudaGetLastError();*/                                            \
            txt << "CUDA Error " << cudaGetErrorName(err) << " (" << cudaGetErrorString(err) << ")"; \
            Falcor::reportFatalError(txt.str());                                                     \
        }                                                                                            \
    }
#define CUDA_CHECK_NOEXCEPT(call) \
    {                             \
        call;                     \
    }

#define CUDA_SYNC_CHECK()                                                                             \
    {                                                                                                 \
        cudaDeviceSynchronize();                                                                      \
        cudaError_t error = cudaGetLastError();                                                       \
        if (error != cudaSuccess)                                                                     \
        {                                                                                             \
            char buf[1024];                                                                           \
            sprintf(buf, "error (%s: line %d): %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            Falcor::reportFatalError(std::string(buf));                                               \
        }                                                                                             \
    }

#define CU_CHECK_SUCCESS(x)                                                                       \
    do                                                                                            \
    {                                                                                             \
        CUresult result = x;                                                                      \
        if (result != CUDA_SUCCESS)                                                               \
        {                                                                                         \
            const char* msg;                                                                      \
            cuGetErrorName(result, &msg);                                                         \
            Falcor::reportFatalError("CUDA Error: " #x " failed with error " + std::string(msg)); \
        }                                                                                         \
    } while (0)

#define CUDA_CHECK_SUCCESS(x)                                                                                            \
    do                                                                                                                   \
    {                                                                                                                    \
        cudaError_t result = x;                                                                                          \
        if (result != cudaSuccess)                                                                                       \
        {                                                                                                                \
            Falcor::reportFatalError("CUDA Error: " #x " failed with error " + std::string(cudaGetErrorString(result))); \
            return 0;                                                                                                    \
        }                                                                                                                \
    } while (0)

namespace
{
uint32_t gNodeMask;
CUdevice gCudaDevice;
CUcontext gCudaContext;
CUstream gCudaStream;
} // namespace

unsigned int initCuda(void)
{
#if 0
    cudaFree(0);
    int32_t numDevices;
    cudaGetDeviceCount(&numDevices);
    return numDevices;
#endif

    CU_CHECK_SUCCESS(cuInit(0));
    int32_t firstGPUID = -1;
    cudaDeviceProp prop;
    int32_t count;
    cudaError_t err = cudaGetDeviceCount(&count);

    for (int32_t i = 0; i < count; ++i)
    {
        err = cudaGetDeviceProperties(&prop, i);
        if (prop.major >= 3)
        {
            firstGPUID = i;
            break;
        }
    }

    if (firstGPUID < 0)
    {
        Falcor::reportFatalError("No CUDA 10 compatible GPU found");
        return false;
    }
    gNodeMask = prop.luidDeviceNodeMask;
    CUDA_CHECK_SUCCESS(cudaSetDevice(firstGPUID));
    CU_CHECK_SUCCESS(cuDeviceGet(&gCudaDevice, firstGPUID));
    CU_CHECK_SUCCESS(cuCtxCreate(&gCudaContext, 0, gCudaDevice));
    CU_CHECK_SUCCESS(cuStreamCreate(&gCudaStream, CU_STREAM_DEFAULT));
    return true;
}

void setCUDAContext()
{
    CU_CHECK_SUCCESS(cuCtxSetCurrent(gCudaContext));
}

void syncCudaDevice()
{
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        Falcor::reportFatalError("Failed to sync CUDA device");
    }
}

void myCudaMemset(void* devPtr, int value, size_t count)
{
    CUDA_CHECK(cudaMemset(devPtr, value, count));
}

void CudaBuffer::allocate(size_t size)
{
    if (mpDevicePtr)
        free();
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
    if (!mpDevicePtr)
        return false;
    if (mSizeBytes <= (count * sizeof(T)))
        return false;

    CUDA_CHECK(cudaMemcpy((void*)t, mpDevicePtr, count * sizeof(T), cudaMemcpyDeviceToHost));
    return true; // might be an error caught by CUDA_CHECK?  TODO: process any such error through
}

template<typename T>
bool CudaBuffer::upload(const T* t, size_t count)
{
    if (!mpDevicePtr)
        return false;
    if (mSizeBytes <= (count * sizeof(T)))
        return false;

    CUDA_CHECK(cudaMemcpy(mpDevicePtr, (void*)t, count * sizeof(T), cudaMemcpyHostToDevice));
    return true; // might be an error caught by CUDA_CHECK?  TODO: process any such error through
}

template<typename T>
void CudaBuffer::allocAndUpload(const std::vector<T>& vt)
{
    allocate(vt.size() * sizeof(T));
    upload((const T*)vt.data(), vt.size());
}

void cudaCopyDeviceToDevice(void* dst, const void* src, size_t count)
{
    CUDA_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice));
}

void cudaCopyHostToDevice(void* dst, const void* src, size_t count)
{
    CUDA_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
}

void* getSharedDevicePtr(Falcor::SharedResourceApiHandle sharedHandle, uint32_t bytes)
{
    // No handle?  No pointer!
    if (sharedHandle == NULL)
    {
        Falcor::reportFatalError("FalcorCUDA::importBufferToCudaPointer - texture shared handle creation failed");
        return nullptr;
    }

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
    if (!ptr)
        return false;
    return cudaSuccess == cudaFree(ptr);
}
