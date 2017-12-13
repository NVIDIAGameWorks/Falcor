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
#ifdef _WIN32
#include "Utils/DXHeader.h"
#include <D3D11.h>

namespace Falcor
{
#define assert_enum_value(prefix, val) static_assert(val == prefix##val, #val " enum value differs from D3D!")

    assert_enum_value(D3D10_, RESOURCE_DIMENSION_UNKNOWN);
    assert_enum_value(D3D10_, RESOURCE_DIMENSION_BUFFER);
    assert_enum_value(D3D10_, RESOURCE_DIMENSION_TEXTURE1D);
    assert_enum_value(D3D10_, RESOURCE_DIMENSION_TEXTURE2D);
    assert_enum_value(D3D10_, RESOURCE_DIMENSION_TEXTURE3D);

    assert_enum_value(DXGI_, FORMAT_UNKNOWN);
    assert_enum_value(DXGI_, FORMAT_R32G32B32A32_TYPELESS);
    assert_enum_value(DXGI_, FORMAT_R32G32B32A32_FLOAT);
    assert_enum_value(DXGI_, FORMAT_R32G32B32A32_UINT);
    assert_enum_value(DXGI_, FORMAT_R32G32B32A32_SINT);
    assert_enum_value(DXGI_, FORMAT_R32G32B32_TYPELESS);
    assert_enum_value(DXGI_, FORMAT_R32G32B32_FLOAT);
    assert_enum_value(DXGI_, FORMAT_R32G32B32_UINT);
    assert_enum_value(DXGI_, FORMAT_R32G32B32_SINT);
    assert_enum_value(DXGI_, FORMAT_R16G16B16A16_TYPELESS);
    assert_enum_value(DXGI_, FORMAT_R16G16B16A16_FLOAT);
    assert_enum_value(DXGI_, FORMAT_R16G16B16A16_UNORM);
    assert_enum_value(DXGI_, FORMAT_R16G16B16A16_UINT);
    assert_enum_value(DXGI_, FORMAT_R16G16B16A16_SNORM);
    assert_enum_value(DXGI_, FORMAT_R16G16B16A16_SINT);
    assert_enum_value(DXGI_, FORMAT_R32G32_TYPELESS);
    assert_enum_value(DXGI_, FORMAT_R32G32_FLOAT);
    assert_enum_value(DXGI_, FORMAT_R32G32_UINT);
    assert_enum_value(DXGI_, FORMAT_R32G32_SINT);
    assert_enum_value(DXGI_, FORMAT_R32G8X24_TYPELESS);
    assert_enum_value(DXGI_, FORMAT_D32_FLOAT_S8X24_UINT);
    assert_enum_value(DXGI_, FORMAT_R32_FLOAT_X8X24_TYPELESS);
    assert_enum_value(DXGI_, FORMAT_X32_TYPELESS_G8X24_UINT);
    assert_enum_value(DXGI_, FORMAT_R10G10B10A2_TYPELESS);
    assert_enum_value(DXGI_, FORMAT_R10G10B10A2_UNORM);
    assert_enum_value(DXGI_, FORMAT_R10G10B10A2_UINT);
    assert_enum_value(DXGI_, FORMAT_R11G11B10_FLOAT);
    assert_enum_value(DXGI_, FORMAT_R8G8B8A8_TYPELESS);
    assert_enum_value(DXGI_, FORMAT_R8G8B8A8_UNORM);
    assert_enum_value(DXGI_, FORMAT_R8G8B8A8_UNORM_SRGB);
    assert_enum_value(DXGI_, FORMAT_R8G8B8A8_UINT);
    assert_enum_value(DXGI_, FORMAT_R8G8B8A8_SNORM);
    assert_enum_value(DXGI_, FORMAT_R8G8B8A8_SINT);
    assert_enum_value(DXGI_, FORMAT_R16G16_TYPELESS);
    assert_enum_value(DXGI_, FORMAT_R16G16_FLOAT);
    assert_enum_value(DXGI_, FORMAT_R16G16_UNORM);
    assert_enum_value(DXGI_, FORMAT_R16G16_UINT);
    assert_enum_value(DXGI_, FORMAT_R16G16_SNORM);
    assert_enum_value(DXGI_, FORMAT_R16G16_SINT);
    assert_enum_value(DXGI_, FORMAT_R32_TYPELESS);
    assert_enum_value(DXGI_, FORMAT_D32_FLOAT);
    assert_enum_value(DXGI_, FORMAT_R32_FLOAT);
    assert_enum_value(DXGI_, FORMAT_R32_UINT);
    assert_enum_value(DXGI_, FORMAT_R32_SINT);
    assert_enum_value(DXGI_, FORMAT_R24G8_TYPELESS);
    assert_enum_value(DXGI_, FORMAT_D24_UNORM_S8_UINT);
    assert_enum_value(DXGI_, FORMAT_R24_UNORM_X8_TYPELESS);
    assert_enum_value(DXGI_, FORMAT_X24_TYPELESS_G8_UINT);
    assert_enum_value(DXGI_, FORMAT_R8G8_TYPELESS);
    assert_enum_value(DXGI_, FORMAT_R8G8_UNORM);
    assert_enum_value(DXGI_, FORMAT_R8G8_UINT);
    assert_enum_value(DXGI_, FORMAT_R8G8_SNORM);
    assert_enum_value(DXGI_, FORMAT_R8G8_SINT);
    assert_enum_value(DXGI_, FORMAT_R16_TYPELESS);
    assert_enum_value(DXGI_, FORMAT_R16_FLOAT);
    assert_enum_value(DXGI_, FORMAT_D16_UNORM);
    assert_enum_value(DXGI_, FORMAT_R16_UNORM);
    assert_enum_value(DXGI_, FORMAT_R16_UINT);
    assert_enum_value(DXGI_, FORMAT_R16_SNORM);
    assert_enum_value(DXGI_, FORMAT_R16_SINT);
    assert_enum_value(DXGI_, FORMAT_R8_TYPELESS);
    assert_enum_value(DXGI_, FORMAT_R8_UNORM);
    assert_enum_value(DXGI_, FORMAT_R8_UINT);
    assert_enum_value(DXGI_, FORMAT_R8_SNORM);
    assert_enum_value(DXGI_, FORMAT_R8_SINT);
    assert_enum_value(DXGI_, FORMAT_A8_UNORM);
    assert_enum_value(DXGI_, FORMAT_R1_UNORM);
    assert_enum_value(DXGI_, FORMAT_R9G9B9E5_SHAREDEXP);
    assert_enum_value(DXGI_, FORMAT_R8G8_B8G8_UNORM);
    assert_enum_value(DXGI_, FORMAT_G8R8_G8B8_UNORM);
    assert_enum_value(DXGI_, FORMAT_BC1_TYPELESS);
    assert_enum_value(DXGI_, FORMAT_BC1_UNORM);
    assert_enum_value(DXGI_, FORMAT_BC1_UNORM_SRGB);
    assert_enum_value(DXGI_, FORMAT_BC2_TYPELESS);
    assert_enum_value(DXGI_, FORMAT_BC2_UNORM);
    assert_enum_value(DXGI_, FORMAT_BC2_UNORM_SRGB);
    assert_enum_value(DXGI_, FORMAT_BC3_TYPELESS);
    assert_enum_value(DXGI_, FORMAT_BC3_UNORM);
    assert_enum_value(DXGI_, FORMAT_BC3_UNORM_SRGB);
    assert_enum_value(DXGI_, FORMAT_BC4_TYPELESS);
    assert_enum_value(DXGI_, FORMAT_BC4_UNORM);
    assert_enum_value(DXGI_, FORMAT_BC4_SNORM);
    assert_enum_value(DXGI_, FORMAT_BC5_TYPELESS);
    assert_enum_value(DXGI_, FORMAT_BC5_UNORM);
    assert_enum_value(DXGI_, FORMAT_BC5_SNORM);
    assert_enum_value(DXGI_, FORMAT_B5G6R5_UNORM);
    assert_enum_value(DXGI_, FORMAT_B5G5R5A1_UNORM);
    assert_enum_value(DXGI_, FORMAT_B8G8R8A8_UNORM);
    assert_enum_value(DXGI_, FORMAT_B8G8R8X8_UNORM);
    assert_enum_value(DXGI_, FORMAT_R10G10B10_XR_BIAS_A2_UNORM);
    assert_enum_value(DXGI_, FORMAT_B8G8R8A8_TYPELESS);
    assert_enum_value(DXGI_, FORMAT_B8G8R8A8_UNORM_SRGB);
    assert_enum_value(DXGI_, FORMAT_B8G8R8X8_TYPELESS);
    assert_enum_value(DXGI_, FORMAT_B8G8R8X8_UNORM_SRGB);
    assert_enum_value(DXGI_, FORMAT_BC6H_TYPELESS);
    assert_enum_value(DXGI_, FORMAT_BC6H_UF16);
    assert_enum_value(DXGI_, FORMAT_BC6H_SF16);
    assert_enum_value(DXGI_, FORMAT_BC7_TYPELESS);
    assert_enum_value(DXGI_, FORMAT_BC7_UNORM);
    assert_enum_value(DXGI_, FORMAT_BC7_UNORM_SRGB);
    assert_enum_value(DXGI_, FORMAT_AYUV);
    assert_enum_value(DXGI_, FORMAT_Y410);
    assert_enum_value(DXGI_, FORMAT_Y416);
    assert_enum_value(DXGI_, FORMAT_NV12);
    assert_enum_value(DXGI_, FORMAT_P010);
    assert_enum_value(DXGI_, FORMAT_P016);
    assert_enum_value(DXGI_, FORMAT_420_OPAQUE);
    assert_enum_value(DXGI_, FORMAT_YUY2);
    assert_enum_value(DXGI_, FORMAT_Y210);
    assert_enum_value(DXGI_, FORMAT_Y216);
    assert_enum_value(DXGI_, FORMAT_NV11);
    assert_enum_value(DXGI_, FORMAT_AI44);
    assert_enum_value(DXGI_, FORMAT_IA44);
    assert_enum_value(DXGI_, FORMAT_P8);
    assert_enum_value(DXGI_, FORMAT_A8P8);
    assert_enum_value(DXGI_, FORMAT_B4G4R4A4_UNORM);
    assert_enum_value(DXGI_, FORMAT_P208);
    assert_enum_value(DXGI_, FORMAT_V208);
    assert_enum_value(DXGI_, FORMAT_V408);
    assert_enum_value(DXGI_, FORMAT_FORCE_UINT);
}

#endif // _WIN32
