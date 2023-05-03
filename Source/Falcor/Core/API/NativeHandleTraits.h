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

#include "NativeHandle.h"

#if FALCOR_HAS_D3D12
#include <d3d12.h>
#endif

#if FALCOR_HAS_VULKAN
#include <vulkan/vulkan.h>
#endif

#include "fstd/bit.h"

namespace Falcor
{

template<typename T>
struct NativeHandleTrait
{};

#define FALCOR_NATIVE_HANDLE(T, TYPE)                \
    template<>                                       \
    struct NativeHandleTrait<T>                      \
    {                                                \
        static const NativeHandleType type = TYPE;   \
        static uint64_t pack(T native)               \
        {                                            \
            return fstd::bit_cast<uint64_t>(native); \
        }                                            \
        static T unpack(uint64_t value)              \
        {                                            \
            return fstd::bit_cast<T>(value);         \
        }                                            \
    };

#if FALCOR_HAS_D3D12
FALCOR_NATIVE_HANDLE(ID3D12Device*, NativeHandleType::ID3D12Device);
FALCOR_NATIVE_HANDLE(ID3D12Resource*, NativeHandleType::ID3D12Resource);
FALCOR_NATIVE_HANDLE(ID3D12PipelineState*, NativeHandleType::ID3D12PipelineState);
FALCOR_NATIVE_HANDLE(ID3D12Fence*, NativeHandleType::ID3D12Fence);
FALCOR_NATIVE_HANDLE(ID3D12CommandQueue*, NativeHandleType::ID3D12CommandQueue);
FALCOR_NATIVE_HANDLE(ID3D12GraphicsCommandList*, NativeHandleType::ID3D12GraphicsCommandList);
FALCOR_NATIVE_HANDLE(D3D12_CPU_DESCRIPTOR_HANDLE, NativeHandleType::D3D12_CPU_DESCRIPTOR_HANDLE);
#endif // FALCOR_HAS_D3D12

#if FALCOR_HAS_VULKAN
FALCOR_NATIVE_HANDLE(VkInstance, NativeHandleType::VkInstance);
FALCOR_NATIVE_HANDLE(VkPhysicalDevice, NativeHandleType::VkPhysicalDevice);
FALCOR_NATIVE_HANDLE(VkDevice, NativeHandleType::VkDevice);
FALCOR_NATIVE_HANDLE(VkImage, NativeHandleType::VkImage);
FALCOR_NATIVE_HANDLE(VkImageView, NativeHandleType::VkImageView);
FALCOR_NATIVE_HANDLE(VkBuffer, NativeHandleType::VkBuffer);
FALCOR_NATIVE_HANDLE(VkBufferView, NativeHandleType::VkBufferView);
FALCOR_NATIVE_HANDLE(VkPipeline, NativeHandleType::VkPipeline);
FALCOR_NATIVE_HANDLE(VkFence, NativeHandleType::VkFence);
FALCOR_NATIVE_HANDLE(VkQueue, NativeHandleType::VkQueue);
FALCOR_NATIVE_HANDLE(VkCommandBuffer, NativeHandleType::VkCommandBuffer);
FALCOR_NATIVE_HANDLE(VkSampler, NativeHandleType::VkSampler);
#endif // FALCOR_HAS_VULKAN

#undef FALCOR_NATIVE_HANDLE
} // namespace Falcor
