/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

namespace Falcor
{
    // Port/Copy of DX enums for portable DDS support

    enum DXResourceDimension
    {
        DX_RESOURCE_DIMENSION_UNKNOWN   = 0,
        DX_RESOURCE_DIMENSION_BUFFER    = 1,
        DX_RESOURCE_DIMENSION_TEXTURE1D = 2,
        DX_RESOURCE_DIMENSION_TEXTURE2D = 3,
        DX_RESOURCE_DIMENSION_TEXTURE3D = 4
    };

    enum DXFormat
    {
        DX_FORMAT_UNKNOWN                     = 0,
        DX_FORMAT_R32G32B32A32_TYPELESS       = 1,
        DX_FORMAT_R32G32B32A32_FLOAT          = 2,
        DX_FORMAT_R32G32B32A32_UINT           = 3,
        DX_FORMAT_R32G32B32A32_SINT           = 4,
        DX_FORMAT_R32G32B32_TYPELESS          = 5,
        DX_FORMAT_R32G32B32_FLOAT             = 6,
        DX_FORMAT_R32G32B32_UINT              = 7,
        DX_FORMAT_R32G32B32_SINT              = 8,
        DX_FORMAT_R16G16B16A16_TYPELESS       = 9,
        DX_FORMAT_R16G16B16A16_FLOAT          = 10,
        DX_FORMAT_R16G16B16A16_UNORM          = 11,
        DX_FORMAT_R16G16B16A16_UINT           = 12,
        DX_FORMAT_R16G16B16A16_SNORM          = 13,
        DX_FORMAT_R16G16B16A16_SINT           = 14,
        DX_FORMAT_R32G32_TYPELESS             = 15,
        DX_FORMAT_R32G32_FLOAT                = 16,
        DX_FORMAT_R32G32_UINT                 = 17,
        DX_FORMAT_R32G32_SINT                 = 18,
        DX_FORMAT_R32G8X24_TYPELESS           = 19,
        DX_FORMAT_D32_FLOAT_S8X24_UINT        = 20,
        DX_FORMAT_R32_FLOAT_X8X24_TYPELESS    = 21,
        DX_FORMAT_X32_TYPELESS_G8X24_UINT     = 22,
        DX_FORMAT_R10G10B10A2_TYPELESS        = 23,
        DX_FORMAT_R10G10B10A2_UNORM           = 24,
        DX_FORMAT_R10G10B10A2_UINT            = 25,
        DX_FORMAT_R11G11B10_FLOAT             = 26,
        DX_FORMAT_R8G8B8A8_TYPELESS           = 27,
        DX_FORMAT_R8G8B8A8_UNORM              = 28,
        DX_FORMAT_R8G8B8A8_UNORM_SRGB         = 29,
        DX_FORMAT_R8G8B8A8_UINT               = 30,
        DX_FORMAT_R8G8B8A8_SNORM              = 31,
        DX_FORMAT_R8G8B8A8_SINT               = 32,
        DX_FORMAT_R16G16_TYPELESS             = 33,
        DX_FORMAT_R16G16_FLOAT                = 34,
        DX_FORMAT_R16G16_UNORM                = 35,
        DX_FORMAT_R16G16_UINT                 = 36,
        DX_FORMAT_R16G16_SNORM                = 37,
        DX_FORMAT_R16G16_SINT                 = 38,
        DX_FORMAT_R32_TYPELESS                = 39,
        DX_FORMAT_D32_FLOAT                   = 40,
        DX_FORMAT_R32_FLOAT                   = 41,
        DX_FORMAT_R32_UINT                    = 42,
        DX_FORMAT_R32_SINT                    = 43,
        DX_FORMAT_R24G8_TYPELESS              = 44,
        DX_FORMAT_D24_UNORM_S8_UINT           = 45,
        DX_FORMAT_R24_UNORM_X8_TYPELESS       = 46,
        DX_FORMAT_X24_TYPELESS_G8_UINT        = 47,
        DX_FORMAT_R8G8_TYPELESS               = 48,
        DX_FORMAT_R8G8_UNORM                  = 49,
        DX_FORMAT_R8G8_UINT                   = 50,
        DX_FORMAT_R8G8_SNORM                  = 51,
        DX_FORMAT_R8G8_SINT                   = 52,
        DX_FORMAT_R16_TYPELESS                = 53,
        DX_FORMAT_R16_FLOAT                   = 54,
        DX_FORMAT_D16_UNORM                   = 55,
        DX_FORMAT_R16_UNORM                   = 56,
        DX_FORMAT_R16_UINT                    = 57,
        DX_FORMAT_R16_SNORM                   = 58,
        DX_FORMAT_R16_SINT                    = 59,
        DX_FORMAT_R8_TYPELESS                 = 60,
        DX_FORMAT_R8_UNORM                    = 61,
        DX_FORMAT_R8_UINT                     = 62,
        DX_FORMAT_R8_SNORM                    = 63,
        DX_FORMAT_R8_SINT                     = 64,
        DX_FORMAT_A8_UNORM                    = 65,
        DX_FORMAT_R1_UNORM                    = 66,
        DX_FORMAT_R9G9B9E5_SHAREDEXP          = 67,
        DX_FORMAT_R8G8_B8G8_UNORM             = 68,
        DX_FORMAT_G8R8_G8B8_UNORM             = 69,
        DX_FORMAT_BC1_TYPELESS                = 70,
        DX_FORMAT_BC1_UNORM                   = 71,
        DX_FORMAT_BC1_UNORM_SRGB              = 72,
        DX_FORMAT_BC2_TYPELESS                = 73,
        DX_FORMAT_BC2_UNORM                   = 74,
        DX_FORMAT_BC2_UNORM_SRGB              = 75,
        DX_FORMAT_BC3_TYPELESS                = 76,
        DX_FORMAT_BC3_UNORM                   = 77,
        DX_FORMAT_BC3_UNORM_SRGB              = 78,
        DX_FORMAT_BC4_TYPELESS                = 79,
        DX_FORMAT_BC4_UNORM                   = 80,
        DX_FORMAT_BC4_SNORM                   = 81,
        DX_FORMAT_BC5_TYPELESS                = 82,
        DX_FORMAT_BC5_UNORM                   = 83,
        DX_FORMAT_BC5_SNORM                   = 84,
        DX_FORMAT_B5G6R5_UNORM                = 85,
        DX_FORMAT_B5G5R5A1_UNORM              = 86,
        DX_FORMAT_B8G8R8A8_UNORM              = 87,
        DX_FORMAT_B8G8R8X8_UNORM              = 88,
        DX_FORMAT_R10G10B10_XR_BIAS_A2_UNORM  = 89,
        DX_FORMAT_B8G8R8A8_TYPELESS           = 90,
        DX_FORMAT_B8G8R8A8_UNORM_SRGB         = 91,
        DX_FORMAT_B8G8R8X8_TYPELESS           = 92,
        DX_FORMAT_B8G8R8X8_UNORM_SRGB         = 93,
        DX_FORMAT_BC6H_TYPELESS               = 94,
        DX_FORMAT_BC6H_UF16                   = 95,
        DX_FORMAT_BC6H_SF16                   = 96,
        DX_FORMAT_BC7_TYPELESS                = 97,
        DX_FORMAT_BC7_UNORM                   = 98,
        DX_FORMAT_BC7_UNORM_SRGB              = 99,
        DX_FORMAT_AYUV                        = 100,
        DX_FORMAT_Y410                        = 101,
        DX_FORMAT_Y416                        = 102,
        DX_FORMAT_NV12                        = 103,
        DX_FORMAT_P010                        = 104,
        DX_FORMAT_P016                        = 105,
        DX_FORMAT_420_OPAQUE                  = 106,
        DX_FORMAT_YUY2                        = 107,
        DX_FORMAT_Y210                        = 108,
        DX_FORMAT_Y216                        = 109,
        DX_FORMAT_NV11                        = 110,
        DX_FORMAT_AI44                        = 111,
        DX_FORMAT_IA44                        = 112,
        DX_FORMAT_P8                          = 113,
        DX_FORMAT_A8P8                        = 114,
        DX_FORMAT_B4G4R4A4_UNORM              = 115,
        DX_FORMAT_P208                        = 130,
        DX_FORMAT_V208                        = 131,
        DX_FORMAT_V408                        = 132,
        DX_FORMAT_FORCE_UINT                  = 0xffffffff
    };

}