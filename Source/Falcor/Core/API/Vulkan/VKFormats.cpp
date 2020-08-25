/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "stdafx.h"
#include "API/Device.h"

namespace Falcor
{
    const VkFormatDesc kVkFormatDesc[] =
    {
        { ResourceFormat::Unknown,                       VK_FORMAT_UNDEFINED },
        { ResourceFormat::R8Unorm,                       VK_FORMAT_R8_UNORM },
        { ResourceFormat::R8Snorm,                       VK_FORMAT_R8_SNORM },
        { ResourceFormat::R16Unorm,                      VK_FORMAT_R16_UNORM },
        { ResourceFormat::R16Snorm,                      VK_FORMAT_R16_SNORM },
        { ResourceFormat::RG8Unorm,                      VK_FORMAT_R8G8_UNORM },
        { ResourceFormat::RG8Snorm,                      VK_FORMAT_R8G8_SNORM },
        { ResourceFormat::RG16Unorm,                     VK_FORMAT_R16G16_UNORM },
        { ResourceFormat::RG16Snorm,                     VK_FORMAT_R16G16_SNORM },
        { ResourceFormat::RGB16Unorm,                    VK_FORMAT_R16G16B16_UNORM },
        { ResourceFormat::RGB16Snorm,                    VK_FORMAT_R16G16B16_SNORM },
        { ResourceFormat::R24UnormX8,                    VK_FORMAT_UNDEFINED },
        { ResourceFormat::RGB5A1Unorm,                   VK_FORMAT_B5G5R5A1_UNORM_PACK16 }, // VK different component order?
        { ResourceFormat::RGBA8Unorm,                    VK_FORMAT_R8G8B8A8_UNORM },
        { ResourceFormat::RGBA8Snorm,                    VK_FORMAT_R8G8B8A8_SNORM },
        { ResourceFormat::RGB10A2Unorm,                  VK_FORMAT_A2R10G10B10_UNORM_PACK32 }, // VK different component order?
        { ResourceFormat::RGB10A2Uint,                   VK_FORMAT_A2R10G10B10_UINT_PACK32 }, // VK different component order?
        { ResourceFormat::RGBA16Unorm,                   VK_FORMAT_R16G16B16A16_UNORM },
        { ResourceFormat::RGBA8UnormSrgb,                VK_FORMAT_R8G8B8A8_SRGB },
        { ResourceFormat::R16Float,                      VK_FORMAT_R16_SFLOAT },
        { ResourceFormat::RG16Float,                     VK_FORMAT_R16G16_SFLOAT },
        { ResourceFormat::RGB16Float,                    VK_FORMAT_R16G16B16_SFLOAT },
        { ResourceFormat::RGBA16Float,                   VK_FORMAT_R16G16B16A16_SFLOAT },
        { ResourceFormat::R32Float,                      VK_FORMAT_R32_SFLOAT },
        { ResourceFormat::R32FloatX32,                   VK_FORMAT_UNDEFINED },
        { ResourceFormat::RG32Float,                     VK_FORMAT_R32G32_SFLOAT },
        { ResourceFormat::RGB32Float,                    VK_FORMAT_R32G32B32_SFLOAT },
        { ResourceFormat::RGBA32Float,                   VK_FORMAT_R32G32B32A32_SFLOAT },
        { ResourceFormat::R11G11B10Float,                VK_FORMAT_B10G11R11_UFLOAT_PACK32 }, // Unsigned in VK
        { ResourceFormat::RGB9E5Float,                   VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 }, // Unsigned in VK
        { ResourceFormat::R8Int,                         VK_FORMAT_R8_SINT },
        { ResourceFormat::R8Uint,                        VK_FORMAT_R8_UINT },
        { ResourceFormat::R16Int,                        VK_FORMAT_R16_SINT },
        { ResourceFormat::R16Uint,                       VK_FORMAT_R16_UINT },
        { ResourceFormat::R32Int,                        VK_FORMAT_R32_SINT },
        { ResourceFormat::R32Uint,                       VK_FORMAT_R32_UINT },
        { ResourceFormat::RG8Int,                        VK_FORMAT_R8G8_SINT },
        { ResourceFormat::RG8Uint,                       VK_FORMAT_R8G8_UINT },
        { ResourceFormat::RG16Int,                       VK_FORMAT_R16G16_SINT },
        { ResourceFormat::RG16Uint,                      VK_FORMAT_R16G16_UINT },
        { ResourceFormat::RG32Int,                       VK_FORMAT_R32G32_SINT },
        { ResourceFormat::RG32Uint,                      VK_FORMAT_R32G32_UINT },
        { ResourceFormat::RGB16Int,                      VK_FORMAT_R16G16B16_SINT },
        { ResourceFormat::RGB16Uint,                     VK_FORMAT_R16G16B16_UINT },
        { ResourceFormat::RGB32Int,                      VK_FORMAT_R32G32B32_SINT },
        { ResourceFormat::RGB32Uint,                     VK_FORMAT_R32G32B32_UINT },
        { ResourceFormat::RGBA8Int,                      VK_FORMAT_R8G8B8A8_SINT },
        { ResourceFormat::RGBA8Uint,                     VK_FORMAT_R8G8B8A8_UINT },
        { ResourceFormat::RGBA16Int,                     VK_FORMAT_R16G16B16A16_SINT },
        { ResourceFormat::RGBA16Uint,                    VK_FORMAT_R16G16B16A16_UINT },
        { ResourceFormat::RGBA32Int,                     VK_FORMAT_R32G32B32A32_SINT },
        { ResourceFormat::RGBA32Uint,                    VK_FORMAT_R32G32B32A32_UINT },
        { ResourceFormat::BGRA8Unorm,                    VK_FORMAT_B8G8R8A8_UNORM },
        { ResourceFormat::BGRA8UnormSrgb,                VK_FORMAT_B8G8R8A8_SRGB },
        { ResourceFormat::BGRX8Unorm,                    VK_FORMAT_B8G8R8A8_UNORM },
        { ResourceFormat::BGRX8UnormSrgb,                VK_FORMAT_B8G8R8A8_SRGB },
        { ResourceFormat::Alpha8Unorm,                   VK_FORMAT_UNDEFINED },
        { ResourceFormat::Alpha32Float,                  VK_FORMAT_UNDEFINED },
        { ResourceFormat::R5G6B5Unorm,                   VK_FORMAT_R5G6B5_UNORM_PACK16 },
        { ResourceFormat::D32Float,                      VK_FORMAT_D32_SFLOAT },
        { ResourceFormat::D16Unorm,                      VK_FORMAT_D16_UNORM },
        { ResourceFormat::D32FloatS8X24,                 VK_FORMAT_D32_SFLOAT_S8_UINT },
        { ResourceFormat::D24UnormS8,                    VK_FORMAT_D24_UNORM_S8_UINT },
        { ResourceFormat::BC1Unorm,                      VK_FORMAT_BC1_RGB_UNORM_BLOCK },
        { ResourceFormat::BC1UnormSrgb,                  VK_FORMAT_BC1_RGB_SRGB_BLOCK },
        { ResourceFormat::BC2Unorm,                      VK_FORMAT_BC2_UNORM_BLOCK },
        { ResourceFormat::BC2UnormSrgb,                  VK_FORMAT_BC2_SRGB_BLOCK },
        { ResourceFormat::BC3Unorm,                      VK_FORMAT_BC3_UNORM_BLOCK },
        { ResourceFormat::BC3UnormSrgb,                  VK_FORMAT_BC3_SRGB_BLOCK },
        { ResourceFormat::BC4Unorm,                      VK_FORMAT_BC4_UNORM_BLOCK },
        { ResourceFormat::BC4Snorm,                      VK_FORMAT_BC4_SNORM_BLOCK },
        { ResourceFormat::BC5Unorm,                      VK_FORMAT_BC5_UNORM_BLOCK },
        { ResourceFormat::BC5Snorm,                      VK_FORMAT_BC5_SNORM_BLOCK },
        { ResourceFormat::BC6HS16,                       VK_FORMAT_BC6H_SFLOAT_BLOCK },
        { ResourceFormat::BC6HU16,                       VK_FORMAT_BC6H_UFLOAT_BLOCK },
        { ResourceFormat::BC7Unorm,                      VK_FORMAT_BC7_UNORM_BLOCK },
        { ResourceFormat::BC7UnormSrgb,                  VK_FORMAT_BC7_SRGB_BLOCK },
    };

    ResourceBindFlags getFormatBindFlags(ResourceFormat format)
    {
        VkFormatProperties p;
        vkGetPhysicalDeviceFormatProperties(gpDevice->getApiHandle(), getVkFormat(format), &p);

        auto convertFlags = [](VkFormatFeatureFlags vk) -> ResourceBindFlags
        {
            ResourceBindFlags f = ResourceBindFlags::None;
            if (vk & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) f |= ResourceBindFlags::ShaderResource;
            if (vk & VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT) f |= ResourceBindFlags::ShaderResource;
            if (vk & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) f |= ResourceBindFlags::ShaderResource;
            if (vk & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) f |= ResourceBindFlags::UnorderedAccess;
            if (vk & VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT) f |= ResourceBindFlags::UnorderedAccess;
            if (vk & VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT) f |= ResourceBindFlags::UnorderedAccess;
            if (vk & VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT) f |= ResourceBindFlags::UnorderedAccess;
            if (vk & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT) f |= ResourceBindFlags::Vertex;
            if (vk & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) f |= ResourceBindFlags::RenderTarget;
            if (vk & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT) f |= ResourceBindFlags::RenderTarget;
            if (vk & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) f |= ResourceBindFlags::DepthStencil;

            return f;
        };
        
        ResourceBindFlags flags = ResourceBindFlags::None;
        flags |= convertFlags(p.bufferFeatures);
        flags |= convertFlags(p.linearTilingFeatures);
        flags |= convertFlags(p.optimalTilingFeatures);


        switch (format)
        {
        case ResourceFormat::R16Uint:
        case ResourceFormat::R32Uint:
            flags |= ResourceBindFlags::Index;
        }

        return flags;
    }
}
