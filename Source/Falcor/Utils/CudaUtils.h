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
#include "Core/API/Handles.h"
#include "Core/API/Buffer.h"
#include <vector>

// Note:  There's some CUDA / Falcor type conflicts.  This header includes Falcor-facing functions.
//        Including <cuda_runtime.h> or <cuda.h> was a pain we wanted to avoid, so CUDA-specific stuff lives in CudaUtils.cpp

/// CUDA device pointer
typedef unsigned long long CUdeviceptr;

/**
 * Get CUDA device pointer for shared resource.
 * This takes a Windows Handle (e.g., from Falcor::Resource::getSharedApiHandle()) on a resource
 * that has been declared "shared" [with Resource::BindFlags::Shared] plus a size of that
 * resource in bytes and returns a CUDA device pointer from that resource that can be
 * passed into OptiX or CUDA.  This pointer is a value returned from
 * cudaExternalMemoryGetMappedBuffer(), so should follow its rules (e.g., the docs claim
 * you are responsible for calling cudaFree() on this pointer).
 */
FALCOR_API void* getSharedDevicePtr(Falcor::SharedResourceApiHandle sharedHandle, uint32_t bytes);

/**
 * Calls cudaFree() on the provided pointer.
 */
FALCOR_API bool freeSharedDevicePtr(void* ptr);

FALCOR_API unsigned int initCuda(void);

FALCOR_API void setCUDAContext(void);

FALCOR_API void myCudaMemset(void* devPtr, int value, size_t count);

FALCOR_API void cudaCopyDeviceToDevice(void* dst, const void* src, size_t count);

FALCOR_API void syncCudaDevice();

FALCOR_API void cudaCopyHostToDevice(void* dst, const void* src, size_t count);

/**
 * Utility class for a GPU/device buffer for use with CUDA.
 * Adapted from Ingo Wald's SIGGRAPH 2019 tutorial code for OptiX 7.
 */
class FALCOR_API CudaBuffer
{
public:
    CudaBuffer() {}

    CUdeviceptr getDevicePtr() { return (CUdeviceptr)mpDevicePtr; }
    size_t getSize() { return mSizeBytes; }

    void allocate(size_t size);
    void resize(size_t size);
    void free();

    template<typename T>
    void allocAndUpload(const std::vector<T>& vt);

    template<typename T>
    bool download(T* t, size_t count);

    template<typename T>
    bool upload(const T* t, size_t count);

private:
    size_t mSizeBytes = 0;
    void* mpDevicePtr = nullptr;
};

namespace Falcor
{
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
            freeSharedDevicePtr((void*)devicePtr);
            devicePtr = (CUdeviceptr)0;
        }
    }
};

inline InteropBuffer createInteropBuffer(ref<Device> pDevice, size_t byteSize)
{
    InteropBuffer interop;

    // Create a new DX <-> CUDA shared buffer using the Falcor API to create, then find its CUDA pointer.
    interop.buffer = Buffer::create(
        pDevice, byteSize, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess | Resource::BindFlags::Shared
    );
    interop.devicePtr = (CUdeviceptr)getSharedDevicePtr(interop.buffer->getSharedApiHandle(), (uint32_t)interop.buffer->getSize());
    checkInvariant(interop.devicePtr != (CUdeviceptr)0, "Failed to create CUDA device ptr for buffer");

    return interop;
}
} // namespace Falcor
