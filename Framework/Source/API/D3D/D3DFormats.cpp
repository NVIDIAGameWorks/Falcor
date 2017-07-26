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
#include "Framework.h"

namespace Falcor
{
    const DxgiFormatDesc kDxgiFormatDesc[] = 
    {
        {ResourceFormat::Unknown,                       DXGI_FORMAT_UNKNOWN},
        {ResourceFormat::R8Unorm,                       DXGI_FORMAT_R8_UNORM},
        {ResourceFormat::R8Snorm,                       DXGI_FORMAT_R8_SNORM},
        {ResourceFormat::R16Unorm,                      DXGI_FORMAT_R16_UNORM},
        {ResourceFormat::R16Snorm,                      DXGI_FORMAT_R16_SNORM},
        {ResourceFormat::RG8Unorm,                      DXGI_FORMAT_R8G8_UNORM},
        {ResourceFormat::RG8Snorm,                      DXGI_FORMAT_R8G8_SNORM},
        {ResourceFormat::RG16Unorm,                     DXGI_FORMAT_R16G16_UNORM},
        {ResourceFormat::RG16Snorm,                     DXGI_FORMAT_R16G16_SNORM},
        {ResourceFormat::RGB16Unorm,                    DXGI_FORMAT_UNKNOWN},
        {ResourceFormat::RGB16Snorm,                    DXGI_FORMAT_UNKNOWN},
        {ResourceFormat::R24UnormX8,                    DXGI_FORMAT_R24_UNORM_X8_TYPELESS},
        {ResourceFormat::RGB5A1Unorm,                   DXGI_FORMAT_B5G5R5A1_UNORM},
        {ResourceFormat::RGBA8Unorm,                    DXGI_FORMAT_R8G8B8A8_UNORM},
        {ResourceFormat::RGBA8Snorm,                    DXGI_FORMAT_R8G8B8A8_SNORM},
        {ResourceFormat::RGB10A2Unorm,                  DXGI_FORMAT_R10G10B10A2_UNORM},
        {ResourceFormat::RGB10A2Uint,                   DXGI_FORMAT_R10G10B10A2_UINT},
        {ResourceFormat::RGBA16Unorm,                   DXGI_FORMAT_R16G16B16A16_UNORM},
        {ResourceFormat::RGBA8UnormSrgb,                DXGI_FORMAT_R8G8B8A8_UNORM_SRGB},
        {ResourceFormat::R16Float,                      DXGI_FORMAT_R16_FLOAT},
        {ResourceFormat::RG16Float,                     DXGI_FORMAT_R16G16_FLOAT},
        {ResourceFormat::RGB16Float,                    DXGI_FORMAT_UNKNOWN},
        {ResourceFormat::RGBA16Float,                   DXGI_FORMAT_R16G16B16A16_FLOAT},
        {ResourceFormat::R32Float,                      DXGI_FORMAT_R32_FLOAT},
        {ResourceFormat::R32FloatX32,                   DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS},
        {ResourceFormat::RG32Float,                     DXGI_FORMAT_R32G32_FLOAT},
        {ResourceFormat::RGB32Float,                    DXGI_FORMAT_R32G32B32_FLOAT},
        {ResourceFormat::RGBA32Float,                   DXGI_FORMAT_R32G32B32A32_FLOAT},
        {ResourceFormat::R11G11B10Float,                DXGI_FORMAT_R11G11B10_FLOAT},
        {ResourceFormat::RGB9E5Float,                   DXGI_FORMAT_R9G9B9E5_SHAREDEXP},
        {ResourceFormat::R8Int,                         DXGI_FORMAT_R8_SINT},
        {ResourceFormat::R8Uint,                        DXGI_FORMAT_R8_UINT},
        {ResourceFormat::R16Int,                        DXGI_FORMAT_R16_SINT},
        {ResourceFormat::R16Uint,                       DXGI_FORMAT_R16_UINT},
        {ResourceFormat::R32Int,                        DXGI_FORMAT_R32_SINT},
        {ResourceFormat::R32Uint,                       DXGI_FORMAT_R32_UINT},
        {ResourceFormat::RG8Int,                        DXGI_FORMAT_R8G8_SINT},
        {ResourceFormat::RG8Uint,                       DXGI_FORMAT_R8G8_UINT},
        {ResourceFormat::RG16Int,                       DXGI_FORMAT_R16G16_SINT},
        {ResourceFormat::RG16Uint,                      DXGI_FORMAT_R16G16_UINT},
        {ResourceFormat::RG32Int,                       DXGI_FORMAT_R32G32_SINT},
        {ResourceFormat::RG32Uint,                      DXGI_FORMAT_R32G32_UINT},
        {ResourceFormat::RGB16Int,                      DXGI_FORMAT_UNKNOWN},
        {ResourceFormat::RGB16Uint,                     DXGI_FORMAT_UNKNOWN},
        {ResourceFormat::RGB32Int,                      DXGI_FORMAT_R32G32B32_SINT},
        {ResourceFormat::RGB32Uint,                     DXGI_FORMAT_R32G32B32_UINT},
        {ResourceFormat::RGBA8Int,                      DXGI_FORMAT_R8G8B8A8_SINT},
        {ResourceFormat::RGBA8Uint,                     DXGI_FORMAT_R8G8B8A8_UINT},
        {ResourceFormat::RGBA16Int,                     DXGI_FORMAT_R16G16B16A16_SINT},
        {ResourceFormat::RGBA16Uint,                    DXGI_FORMAT_R16G16B16A16_UINT},
        {ResourceFormat::RGBA32Int,                     DXGI_FORMAT_R32G32B32A32_SINT},
        {ResourceFormat::RGBA32Uint,                    DXGI_FORMAT_R32G32B32A32_UINT},
        {ResourceFormat::BGRA8Unorm,                    DXGI_FORMAT_B8G8R8A8_UNORM},
        {ResourceFormat::BGRA8UnormSrgb,                DXGI_FORMAT_B8G8R8A8_UNORM_SRGB},
        {ResourceFormat::BGRX8Unorm,                    DXGI_FORMAT_B8G8R8X8_UNORM},
        {ResourceFormat::BGRX8UnormSrgb,                DXGI_FORMAT_B8G8R8X8_UNORM_SRGB},
        {ResourceFormat::Alpha8Unorm,                   DXGI_FORMAT_A8_UNORM},
        {ResourceFormat::Alpha32Float,                  DXGI_FORMAT_UNKNOWN},
        {ResourceFormat::R5G6B5Unorm,                   DXGI_FORMAT_B5G6R5_UNORM},
        {ResourceFormat::D32Float,                      DXGI_FORMAT_D32_FLOAT},
        {ResourceFormat::D16Unorm,                      DXGI_FORMAT_D16_UNORM},
        {ResourceFormat::D32FloatS8X24,                 DXGI_FORMAT_D32_FLOAT_S8X24_UINT},
        {ResourceFormat::D24UnormS8,                    DXGI_FORMAT_D24_UNORM_S8_UINT},
        {ResourceFormat::BC1Unorm,                      DXGI_FORMAT_BC1_UNORM},
        {ResourceFormat::BC1UnormSrgb,                  DXGI_FORMAT_BC1_UNORM_SRGB},
        {ResourceFormat::BC2Unorm,                      DXGI_FORMAT_BC2_UNORM},
        {ResourceFormat::BC2UnormSrgb,                  DXGI_FORMAT_BC2_UNORM_SRGB},
        {ResourceFormat::BC3Unorm,                      DXGI_FORMAT_BC3_UNORM},
        {ResourceFormat::BC3UnormSrgb,                  DXGI_FORMAT_BC3_UNORM_SRGB},
        {ResourceFormat::BC4Unorm,                      DXGI_FORMAT_BC4_UNORM},
        {ResourceFormat::BC4Snorm,                      DXGI_FORMAT_BC4_SNORM},
        {ResourceFormat::BC5Unorm,                      DXGI_FORMAT_BC5_UNORM},
        {ResourceFormat::BC5Snorm,                      DXGI_FORMAT_BC5_SNORM},
        {ResourceFormat::BC6HS16,                       DXGI_FORMAT_BC6H_SF16},
        {ResourceFormat::BC6HU16,                       DXGI_FORMAT_BC6H_UF16},
        {ResourceFormat::BC7Unorm,                      DXGI_FORMAT_BC7_UNORM},
        {ResourceFormat::BC7UnormSrgb,                  DXGI_FORMAT_BC7_UNORM_SRGB},
    };

    static_assert(arraysize(kDxgiFormatDesc) == (uint32_t)ResourceFormat::BC7UnormSrgb + 1, "DXGI format desc table has a wrong size");
}
