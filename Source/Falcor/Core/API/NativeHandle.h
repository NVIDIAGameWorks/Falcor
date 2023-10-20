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

#include "Core/Error.h"

#include <cstdint>

namespace Falcor
{

enum class NativeHandleType
{
    Unknown,

    ID3D12Device,
    ID3D12Resource,
    ID3D12PipelineState,
    ID3D12Fence,
    ID3D12CommandQueue,
    ID3D12GraphicsCommandList,
    D3D12_CPU_DESCRIPTOR_HANDLE,

    VkInstance,
    VkPhysicalDevice,
    VkDevice,
    VkImage,
    VkImageView,
    VkBuffer,
    VkBufferView,
    VkPipeline,
    VkFence,
    VkQueue,
    VkCommandBuffer,
    VkSampler,
};

template<typename T>
struct NativeHandleTrait;

/// Represents a native graphics API handle (e.g. D3D12 or Vulkan).
/// Native handles are expected to fit into 64 bits.
/// Type information and conversion from/to native handles is done
/// using type traits from NativeHandleTraits.h which needs to be
/// included when creating and accessing NativeHandle.
/// This separation is done so we don't expose the heavy D3D12/Vulkan
/// headers everywhere.
class NativeHandle
{
public:
    NativeHandle() = default;

    template<typename T>
    explicit NativeHandle(T native)
    {
        mType = NativeHandleTrait<T>::type;
        mValue = NativeHandleTrait<T>::pack(native);
    }

    NativeHandleType getType() const { return mType; }

    bool isValid() const { return mType != NativeHandleType::Unknown; }

    template<typename T>
    T as() const
    {
        FALCOR_ASSERT(mType == NativeHandleTrait<T>::type);
        return NativeHandleTrait<T>::unpack(mValue);
    }

private:
    NativeHandleType mType{NativeHandleType::Unknown};
    uint64_t mValue{0};
};

} // namespace Falcor
