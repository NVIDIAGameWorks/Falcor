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
#include "Core/Error.h"
#include "Core/API/Device.h"
#include "Core/API/Buffer.h"
#include "Core/API/Texture.h"
#include "Core/API/Fence.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <sstream>

namespace
{
uint32_t gNodeMask;
CUdevice gCudaDevice;
CUcontext gCudaContext;
CUstream gCudaStream;
} // namespace

namespace Falcor
{
namespace cuda_utils
{

void deviceSynchronize()
{
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    FALCOR_CHECK(error == cudaSuccess, "Failed to sync CUDA device: {}.", cudaGetErrorString(error));
}

void* mallocDevice(size_t size)
{
    void* devPtr;
    FALCOR_CUDA_CHECK(cudaMalloc(&devPtr, size));
    return devPtr;
}

void freeDevice(void* devPtr)
{
    if (!devPtr)
        return;
    FALCOR_CUDA_CHECK(cudaFree(devPtr));
}

void memcpyDeviceToDevice(void* dst, const void* src, size_t count)
{
    FALCOR_CUDA_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice));
}

void memcpyHostToDevice(void* dst, const void* src, size_t count)
{
    FALCOR_CUDA_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
}

void memcpyDeviceToHost(void* dst, const void* src, size_t count)
{
    FALCOR_CUDA_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
}

void memsetDevice(void* devPtr, int value, size_t count)
{
    FALCOR_CUDA_CHECK(cudaMemset(devPtr, value, count));
}

cudaExternalMemory_t importExternalMemory(const Buffer* buffer)
{
    FALCOR_CHECK(buffer, "'buffer' is nullptr.");
    FALCOR_CHECK(is_set(buffer->getBindFlags(), ResourceBindFlags::Shared), "Buffer must be created with ResourceBindFlags::Shared.");
    SharedResourceApiHandle sharedHandle = buffer->getSharedApiHandle();
    FALCOR_CHECK(sharedHandle, "Buffer shared handle creation failed.");

    cudaExternalMemoryHandleDesc desc = {};
    switch (buffer->getDevice()->getType())
    {
    case Device::Type::D3D12:
        desc.type = cudaExternalMemoryHandleTypeD3D12Resource;
        break;
#if FALCOR_WINDOWS
    case Device::Type::Vulkan:
        desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
        break;
#endif
    default:
        FALCOR_THROW("Unsupported device type '{}'.", buffer->getDevice()->getType());
    }
    desc.handle.win32.handle = sharedHandle;
    desc.size = buffer->getSize();
    desc.flags = cudaExternalMemoryDedicated;

    cudaExternalMemory_t extMem;
    FALCOR_CUDA_CHECK(cudaImportExternalMemory(&extMem, &desc));
    return extMem;
}

void destroyExternalMemory(cudaExternalMemory_t extMem)
{
    FALCOR_CUDA_CHECK(cudaDestroyExternalMemory(extMem));
}

void* externalMemoryGetMappedBuffer(cudaExternalMemory_t extMem, size_t offset, size_t size)
{
    cudaExternalMemoryBufferDesc desc = {};
    desc.offset = offset;
    desc.size = size;

    void* devPtr = nullptr;
    FALCOR_CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&devPtr, extMem, &desc));
    return devPtr;
}

cudaExternalSemaphore_t importExternalSemaphore(const Fence* fence)
{
    FALCOR_CHECK(fence, "'fence' is nullptr.");
    FALCOR_CHECK(fence->getDesc().shared, "'fence' must be created with shared=true.");
    SharedFenceApiHandle sharedHandle = fence->getSharedApiHandle();
    FALCOR_CHECK(sharedHandle, "Fence shared handle creation failed.");

    cudaExternalSemaphoreHandleDesc desc = {};
    switch (fence->getDevice()->getType())
    {
    case Device::Type::D3D12:
        desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
        break;
    default:
        FALCOR_THROW("Unsupported device type '{}'.", fence->getDevice()->getType());
    }
    desc.handle.win32.handle = (void*)sharedHandle;

    cudaExternalSemaphore_t extSem;
    FALCOR_CUDA_CHECK(cudaImportExternalSemaphore(&extSem, &desc));
    return extSem;
}

void destroyExternalSemaphore(cudaExternalSemaphore_t extSem)
{
    FALCOR_CUDA_CHECK(cudaDestroyExternalSemaphore(extSem));
}

void signalExternalSemaphore(cudaExternalSemaphore_t extSem, uint64_t value, cudaStream_t stream)
{
    cudaExternalSemaphoreSignalParams params = {};
    params.params.fence.value = value;
    FALCOR_CUDA_CHECK(cudaSignalExternalSemaphoresAsync(&extSem, &params, 1, stream));
}

void waitExternalSemaphore(cudaExternalSemaphore_t extSem, uint64_t value, cudaStream_t stream)
{
    cudaExternalSemaphoreWaitParams params = {};
    params.params.fence.value = value;
    FALCOR_CUDA_CHECK(cudaWaitExternalSemaphoresAsync(&extSem, &params, 1, stream));
}

void* getSharedDevicePtr(SharedResourceApiHandle sharedHandle, uint32_t bytes)
{
    // No handle?  No pointer!
    FALCOR_CHECK(sharedHandle, "Texture shared handle creation failed");

    // Create the descriptor of our shared memory buffer
    cudaExternalMemoryHandleDesc externalMemoryHandleDesc;
    memset(&externalMemoryHandleDesc, 0, sizeof(externalMemoryHandleDesc));
    externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    externalMemoryHandleDesc.handle.win32.handle = sharedHandle;
    externalMemoryHandleDesc.size = bytes;
    externalMemoryHandleDesc.flags = cudaExternalMemoryDedicated;

    // Get a handle to that memory
    cudaExternalMemory_t externalMemory;
    FALCOR_CUDA_CHECK(cudaImportExternalMemory(&externalMemory, &externalMemoryHandleDesc));

    // Create a descriptor for our shared buffer pointer
    cudaExternalMemoryBufferDesc bufDesc;
    memset(&bufDesc, 0, sizeof(bufDesc));
    bufDesc.size = bytes;

    // Actually map the buffer
    void* devPtr = nullptr;
    FALCOR_CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&devPtr, externalMemory, &bufDesc));

    return devPtr;
}

bool freeSharedDevicePtr(void* ptr)
{
    if (!ptr)
        return false;
    return cudaSuccess == cudaFree(ptr);
}

cudaMipmappedArray_t importTextureToMipmappedArray(ref<Texture> pTex, uint32_t cudaUsageFlags)
{
    SharedResourceApiHandle sharedHandle = pTex->getSharedApiHandle();
    FALCOR_CHECK(sharedHandle, "Texture shared handle creation failed");

    cudaExternalMemoryHandleDesc externalMemoryHandleDesc;
    memset(&externalMemoryHandleDesc, 0, sizeof(externalMemoryHandleDesc));

    externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    externalMemoryHandleDesc.handle.win32.handle = sharedHandle;
    externalMemoryHandleDesc.size = pTex->getTextureSizeInBytes();
    externalMemoryHandleDesc.flags = cudaExternalMemoryDedicated;

    cudaExternalMemory_t externalMemory;
    FALCOR_CUDA_CHECK(cudaImportExternalMemory(&externalMemory, &externalMemoryHandleDesc));

    // Map mipmapped array onto external memory
    cudaExternalMemoryMipmappedArrayDesc mipDesc;
    memset(&mipDesc, 0, sizeof(mipDesc));
    auto format = pTex->getFormat();
    mipDesc.formatDesc.x = getNumChannelBits(format, 0);
    mipDesc.formatDesc.y = getNumChannelBits(format, 1);
    mipDesc.formatDesc.z = getNumChannelBits(format, 2);
    mipDesc.formatDesc.w = getNumChannelBits(format, 3);
    mipDesc.formatDesc.f = (getFormatType(format) == FormatType::Float) ? cudaChannelFormatKindFloat : cudaChannelFormatKindUnsigned;
    mipDesc.extent.depth = 1;
    mipDesc.extent.width = pTex->getWidth();
    mipDesc.extent.height = pTex->getHeight();
    mipDesc.flags = cudaUsageFlags;
    mipDesc.numLevels = 1;

    cudaMipmappedArray_t mipmappedArray;
    FALCOR_CUDA_CHECK(cudaExternalMemoryGetMappedMipmappedArray(&mipmappedArray, externalMemory, &mipDesc));
    return mipmappedArray;
}

cudaSurfaceObject_t mapTextureToSurface(ref<Texture> pTex, uint32_t cudaUsageFlags)
{
    // Create a mipmapped array from the texture
    cudaMipmappedArray_t mipmap = importTextureToMipmappedArray(pTex, cudaUsageFlags);

    // Grab level 0
    cudaArray_t cudaArray;
    FALCOR_CUDA_CHECK(cudaGetMipmappedArrayLevel(&cudaArray, mipmap, 0));

    // Create cudaSurfObject_t from CUDA array
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.res.array.array = cudaArray;
    resDesc.resType = cudaResourceTypeArray;

    cudaSurfaceObject_t surface;
    FALCOR_CUDA_CHECK(cudaCreateSurfaceObject(&surface, &resDesc));
    return surface;
}

inline int32_t findDeviceByLUID(const std::vector<cudaDeviceProp>& devices, const AdapterLUID& luid)
{
    for (int32_t i = 0; i < devices.size(); ++i)
    {
#if FALCOR_WINDOWS
        // On Windows, we compare the 8-byte LUID. The LUID is the same for
        // D3D12, Vulkan and CUDA.
        static_assert(sizeof(cudaDeviceProp::luid) <= sizeof(AdapterLUID::luid));
        if (std::memcmp(devices[i].luid, luid.luid.data(), sizeof(cudaDeviceProp::luid)) == 0)
            return i;
#elif FALCOR_LINUX
        // On Linux, the LUID is not supported. Instead we compare the 16-byte
        // UUID which GFX conveniently returns in-place of the LUID.
        static_assert(sizeof(cudaDeviceProp::uuid) <= sizeof(AdapterLUID::luid));
        if (std::memcmp(&devices[i].uuid, luid.luid.data(), sizeof(cudaDeviceProp::uuid)) == 0)
            return i;
#endif
    }
    return -1;
}

inline int32_t findDeviceByName(const std::vector<cudaDeviceProp>& devices, std::string_view name)
{
    for (int32_t i = 0; i < devices.size(); ++i)
        if (devices[i].name == name)
            return i;
    return -1;
}

CudaDevice::CudaDevice(const Device* pDevice)
{
    FALCOR_CHECK(pDevice, "'pDevice' is nullptr.");
    FALCOR_CU_CHECK(cuInit(0));

    // Get a list of all available CUDA devices.
    int32_t deviceCount;
    FALCOR_CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    std::vector<cudaDeviceProp> devices(deviceCount);
    for (int32_t i = 0; i < deviceCount; ++i)
        FALCOR_CUDA_CHECK(cudaGetDeviceProperties(&devices[i], i));

    // First we try to find the matching CUDA device by LUID.
    int32_t selectedDevice = findDeviceByLUID(devices, pDevice->getInfo().adapterLUID);
    if (selectedDevice < 0)
    {
        logWarning("Failed to find CUDA device by LUID. Falling back to device name.");
        // Next we try to find the matching CUDA device by name.
        selectedDevice = findDeviceByName(devices, pDevice->getInfo().adapterName);
        if (selectedDevice < 0)
        {
            logWarning("Failed to find CUDA device by name. Falling back to first compatible device.");
            // Finally we try to find the first compatible CUDA device.
            for (int32_t i = 0; i < devices.size(); ++i)
            {
                if (devices[i].major >= 7)
                {
                    selectedDevice = i;
                    break;
                }
            }
        }
    }

    if (selectedDevice < 0)
        FALCOR_THROW("No compatible CUDA device found.");

    FALCOR_CUDA_CHECK(cudaSetDevice(selectedDevice));
    FALCOR_CU_CHECK(cuDeviceGet(&mCudaDevice, selectedDevice));
    FALCOR_CU_CHECK(cuDevicePrimaryCtxRetain(&mCudaContext, mCudaDevice));
    FALCOR_CU_CHECK(cuStreamCreate(&mCudaStream, CU_STREAM_DEFAULT));

    const auto& props = devices[selectedDevice];
    logInfo("Created CUDA device '{}' (architecture {}.{}).", props.name, props.major, props.minor);
}

CudaDevice::~CudaDevice()
{
    FALCOR_CU_CHECK(cuStreamDestroy(mCudaStream));
    FALCOR_CU_CHECK(cuDevicePrimaryCtxRelease(mCudaDevice));
}

} // namespace cuda_utils

} // namespace Falcor
