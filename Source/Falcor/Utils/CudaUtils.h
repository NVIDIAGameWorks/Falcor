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
#pragma once
#include "Core/Macros.h"
#include "Core/Object.h"
#include "Core/API/fwd.h"
#include "Core/API/Handles.h"
#include "Core/API/Device.h"
#include "Core/API/Fence.h"

#include <cuda.h>
#include "CudaRuntime.h" // Instead of <cuda_runtime.h> to avoid name clashes.

#include <vector>

#define FALCOR_CUDA_CHECK(call)                                                                     \
    {                                                                                               \
        cudaError_t result = call;                                                                  \
        if (result != cudaSuccess)                                                                  \
        {                                                                                           \
            const char* errorName = cudaGetErrorName(result);                                       \
            const char* errorString = cudaGetErrorString(result);                                   \
            FALCOR_THROW("CUDA call {} failed with error {} ({}).", #call, errorName, errorString); \
        }                                                                                           \
    }

#define FALCOR_CU_CHECK(call)                                                                       \
    do                                                                                              \
    {                                                                                               \
        CUresult result = call;                                                                     \
        if (result != CUDA_SUCCESS)                                                                 \
        {                                                                                           \
            const char* errorName;                                                                  \
            cuGetErrorName(result, &errorName);                                                     \
            const char* errorString;                                                                \
            cuGetErrorString(result, &errorString);                                                 \
            FALCOR_THROW("CUDA call {} failed with error {} ({}).", #call, errorName, errorString); \
        }                                                                                           \
    } while (0)

/// CUDA device pointer.
typedef unsigned long long CUdeviceptr;

namespace Falcor
{

namespace cuda_utils
{
FALCOR_API void deviceSynchronize();

FALCOR_API void* mallocDevice(size_t size);
FALCOR_API void freeDevice(void* devPtr);

FALCOR_API void memcpyDeviceToDevice(void* dst, const void* src, size_t count);
FALCOR_API void memcpyHostToDevice(void* dst, const void* src, size_t count);
FALCOR_API void memcpyDeviceToHost(void* dst, const void* src, size_t count);

FALCOR_API void memsetDevice(void* devPtr, int value, size_t count);

FALCOR_API cudaExternalMemory_t importExternalMemory(const Buffer* buffer);
FALCOR_API void destroyExternalMemory(cudaExternalMemory_t extMem);
FALCOR_API void* externalMemoryGetMappedBuffer(cudaExternalMemory_t extMem, size_t offset, size_t size);

FALCOR_API cudaExternalSemaphore_t importExternalSemaphore(const Fence* fence);
FALCOR_API void destroyExternalSemaphore(cudaExternalSemaphore_t extSem);
FALCOR_API void signalExternalSemaphore(cudaExternalSemaphore_t extSem, uint64_t value, cudaStream_t stream = 0);
FALCOR_API void waitExternalSemaphore(cudaExternalSemaphore_t extSem, uint64_t value, cudaStream_t stream = 0);

/**
 * Get CUDA device pointer for shared resource.
 * This takes a Windows Handle (e.g., from Resource::getSharedApiHandle()) on a resource
 * that has been declared "shared" [with ResourceBindFlags::Shared] plus a size of that
 * resource in bytes and returns a CUDA device pointer from that resource that can be
 * passed into OptiX or CUDA.  This pointer is a value returned from
 * cudaExternalMemoryGetMappedBuffer(), so should follow its rules (e.g., the docs claim
 * you are responsible for calling cudaFree() on this pointer).
 */
FALCOR_API void* getSharedDevicePtr(SharedResourceApiHandle sharedHandle, uint32_t bytes);

/**
 * Calls cudaFree() on the provided pointer.
 */
FALCOR_API bool freeSharedDevicePtr(void* ptr);

/**
 * Imports the texture into a CUDA mipmapped array and returns the array in mipmappedArray. This method should only be
 * called once per texture resource.
 * @param pTex Pointer to the texture being imported
 * @param usageFlags The requested flags to be bound to the mipmapped array
 * @return Returns the imported mipmapped array.
 */
FALCOR_API cudaMipmappedArray_t importTextureToMipmappedArray(ref<Texture> pTex, uint32_t cudaUsageFlags);

/**
 * Maps a texture to a surface object which can be read and written within a CUDA kernel.
 * This method should only be called once per texture on initial load. Store the returned surface object for repeated
 * use.
 * @param pTex Pointer to the texture being mapped
 * @param usageFlags The requested flags to be bound to the underlying mipmapped array that will be used to create the
 * surface object
 * @return The surface object that the input texture is bound to.
 */
FALCOR_API cudaSurfaceObject_t mapTextureToSurface(ref<Texture> pTex, uint32_t usageFlags);

/// Wraps a CUDA device, context and stream.
class FALCOR_API CudaDevice : public Object
{
    FALCOR_OBJECT(cuda_utils::CudaDevice)
public:
    /// Constructor.
    /// Creates a CUDA device on the same adapter as the Falcor device.
    CudaDevice(const Device* pDevice);
    ~CudaDevice();

    CUdevice getDevice() const { return mCudaDevice; }
    CUcontext getContext() const { return mCudaContext; }
    CUstream getStream() const { return mCudaStream; }

private:
    CUdevice mCudaDevice;
    CUcontext mCudaContext;
    CUstream mCudaStream;
};

/// Wraps an external memory resource.
class ExternalMemory : public Object
{
    FALCOR_OBJECT(cuda_utils::ExternalMemory)
public:
    ExternalMemory(ref<Resource> pResource) : mpResource(pResource.get())
    {
        FALCOR_CHECK(mpResource, "'resource' is null.");
        if (auto pBuffer = pResource->asBuffer())
        {
            mExternalMemory = importExternalMemory(pBuffer.get());
            mSize = pBuffer->getSize();
        }
        else
        {
            FALCOR_THROW("'resource' must be a buffer.");
        }
    }

    ~ExternalMemory() { destroyExternalMemory(mExternalMemory); }

    size_t getSize() const { return mSize; }

    void* getMappedData() const
    {
        if (!mMappedData)
            mMappedData = externalMemoryGetMappedBuffer(mExternalMemory, 0, mSize);
        return mMappedData;
    }

private:
    /// Keep a non-owning pointer to the resource.
    /// TODO: If available, we should use a weak_ref here.
    Resource* mpResource;
    cudaExternalMemory_t mExternalMemory;
    size_t mSize;
    mutable void* mMappedData{nullptr};
};

/// Wraps an external semaphore.
class ExternalSemaphore : public Object
{
    FALCOR_OBJECT(cuda_utils::ExternalSemaphore)
public:
    ExternalSemaphore(ref<Fence> pFence) : mpFence(pFence.get())
    {
        FALCOR_CHECK(mpFence, "'fence' is null.");
        FALCOR_CHECK(mpFence->getDesc().shared, "'fence' must be created with shared=true.");
        mExternalSemaphore = importExternalSemaphore(mpFence);
    }

    ~ExternalSemaphore() { destroyExternalSemaphore(mExternalSemaphore); }

    void signal(uint64_t value, cudaStream_t stream = 0) { signalExternalSemaphore(mExternalSemaphore, value, stream); }

    void wait(uint64_t value, cudaStream_t stream = 0) { waitExternalSemaphore(mExternalSemaphore, value, stream); }

    void waitForCuda(CopyContext* pCopyContext, cudaStream_t stream = 0, uint64_t value = Fence::kAuto)
    {
        uint64_t signalValue = mpFence->updateSignaledValue(value);
        signal(signalValue, stream);
        pCopyContext->wait(mpFence, signalValue);
    }

    void waitForFalcor(CopyContext* pCopyContext, cudaStream_t stream = 0, uint64_t value = Fence::kAuto)
    {
        uint64_t signalValue = pCopyContext->signal(mpFence, value);
        wait(signalValue, stream);
    }

private:
    /// Keep a non-owning pointer to the fence.
    /// TODO: If available, we should use a weak_ref here.
    Fence* mpFence;
    cudaExternalSemaphore_t mExternalSemaphore;
};

} // namespace cuda_utils

/**
 * Structure to encapsulate DX <-> CUDA interop data for a buffer.
 */
struct InteropBuffer
{
    ref<Buffer> buffer;                     // Falcor buffer
    CUdeviceptr devicePtr = (CUdeviceptr)0; // CUDA pointer to buffer

    void free()
    {
        if (devicePtr)
        {
            cuda_utils::freeSharedDevicePtr((void*)devicePtr);
            devicePtr = (CUdeviceptr)0;
        }
    }
};

inline InteropBuffer createInteropBuffer(ref<Device> pDevice, size_t byteSize)
{
    InteropBuffer interop;

    // Create a new DX <-> CUDA shared buffer using the Falcor API to create, then find its CUDA pointer.
    interop.buffer =
        pDevice->createBuffer(byteSize, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared);
    interop.devicePtr =
        (CUdeviceptr)cuda_utils::getSharedDevicePtr(interop.buffer->getSharedApiHandle(), (uint32_t)interop.buffer->getSize());
    FALCOR_CHECK(interop.devicePtr != (CUdeviceptr)0, "Failed to create CUDA device ptr for buffer");

    return interop;
}
} // namespace Falcor
