/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/
#pragma once
#define NOMINMAX
#include "API/Formats.h"

#ifdef _WIN32
    #define VK_USE_PLATFORM_WIN32_KHR
#else
    #define VK_USE_PLATFORM_XLIB_KHR
#endif

#include <vulkan/vulkan.h>

// Remove defines from XLib.h (included by vulkan.h) that cause conflicts
#ifndef _WIN32
#undef None
#undef Status
#undef Bool
#undef Always
#endif

#ifdef _WIN32
    #pragma comment(lib, "vulkan-1.lib")
#endif

#include "API/Vulkan/VKSmartHandle.h"

namespace Falcor
{
    struct VkFormatDesc
    {
        ResourceFormat falcorFormat;
        VkFormat vkFormat;
    };

    extern const VkFormatDesc kVkFormatDesc[];

    inline VkFormat getVkFormat(ResourceFormat format)
    {
        assert(kVkFormatDesc[(uint32_t)format].falcorFormat == format);
        assert(kVkFormatDesc[(uint32_t)format].vkFormat != VK_FORMAT_UNDEFINED);
        return kVkFormatDesc[(uint32_t)format].vkFormat;
    }

    using HeapCpuHandle = void*;
    using HeapGpuHandle = void*;

    class DescriptorHeapEntry;

#ifdef _WIN32
    using WindowHandle = HWND;
#else
    struct WindowHandle
    {
        Display* pDisplay;
        Window window;
    };
#endif

    using DeviceHandle = VkDeviceData::SharedPtr;
    using CommandListHandle = VkCommandBuffer;
    using CommandQueueHandle = VkQueue;
    using ApiCommandQueueType = uint32_t;
    using CommandAllocatorHandle = VkHandle<VkCommandPool>::SharedPtr;
    using CommandSignatureHandle = void*;
    using FenceHandle = VkSemaphore;
    using ResourceHandle = VkResource<VkImage, VkBuffer>::SharedPtr;
    using RtvHandle = VkResource<VkImageView, VkBufferView>::SharedPtr;
    using DsvHandle = VkResource<VkImageView, VkBufferView>::SharedPtr;
    using SrvHandle = VkResource<VkImageView, VkBufferView>::SharedPtr;
    using UavHandle = VkResource<VkImageView, VkBufferView>::SharedPtr;
    using CbvHandle = VkResource<VkImageView, VkBufferView>::SharedPtr;
    using FboHandle = VkFbo::SharedPtr;
    using SamplerHandle = VkHandle<VkSampler>::SharedPtr;
    using GpuAddress = size_t;
    using DescriptorSetApiHandle = VkDescriptorSet;
    using QueryHeapHandle = VkHandle<VkQueryPool>::SharedPtr;

    using GraphicsStateHandle = VkHandle<VkPipeline>::SharedPtr;
    using ComputeStateHandle = VkHandle<VkPipeline>::SharedPtr;
    using ShaderHandle = VkHandle<VkShaderModule>::SharedPtr;
    using ShaderReflectionHandle = void*;
    using RootSignatureHandle = VkRootSignature::SharedPtr;
    using DescriptorHeapHandle = VkHandle<VkDescriptorPool>::SharedPtr;

    using VaoHandle = void*;
    using VertexShaderHandle = void*;
    using FragmentShaderHandle = void*;
    using DomainShaderHandle = void*;
    using HullShaderHandle = void*;
    using GeometryShaderHandle = void*;
    using ComputeShaderHandle = void*;
    using ProgramHandle = void*;
    using DepthStencilStateHandle = void*;
    using RasterizerStateHandle = void*;
    using BlendStateHandle = void*;

    static const uint32_t kDefaultSwapChainBuffers = 3;

    using ApiObjectHandle = VkBaseApiHandle::SharedPtr;

    uint32_t getMaxViewportCount();

#define appendShaderExtension(_a)  _a ".glsl"
}

#define DEFAULT_API_MAJOR_VERSION 1
#define DEFAULT_API_MINOR_VERSION 0

#define VK_FAILED(res) (res != VK_SUCCESS)

#ifdef _LOG_ENABLED
#define vk_call(a) {auto r = a; if(VK_FAILED(r)) { logError("Vulkan call failed.\n"#a); }}
#else
#define vk_call(a) a
#endif

#define UNSUPPORTED_IN_VULKAN(msg_) {logWarning(msg_ + std::string(" is not supported in Vulkan. Ignoring call."));}
