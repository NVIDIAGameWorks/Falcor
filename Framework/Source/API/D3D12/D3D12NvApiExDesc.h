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
#if _ENABLE_NVAPI
    struct NvApiPsoExDesc
    {
        NV_PSO_EXTENSION psoExtension;
        NVAPI_D3D12_PSO_VERTEX_SHADER_DESC mVsExDesc;
        NVAPI_D3D12_PSO_HULL_SHADER_DESC   mHsExDesc;
        NVAPI_D3D12_PSO_DOMAIN_SHADER_DESC mDsExDesc;
        NVAPI_D3D12_PSO_GEOMETRY_SHADER_DESC mGsExDesc;
        NVAPI_D3D12_PSO_SET_SHADER_EXTENSION_SLOT_DESC mExtSlotDesc;
        std::vector<NV_CUSTOM_SEMANTIC> mCustomSemantics;
    };

    // TODO add these functions for Hs, Ds, Gs
    inline void createNvApiVsExDesc(NvApiPsoExDesc& ret)
    {
        ret.psoExtension = NV_PSO_VERTEX_SHADER_EXTENSION;

        auto& desc = ret.mVsExDesc;
        std::memset(&desc, 0, sizeof(desc));

        desc.psoExtension = NV_PSO_VERTEX_SHADER_EXTENSION;
        desc.version = NV_VERTEX_SHADER_PSO_EXTENSION_DESC_VER;
        desc.baseVersion = NV_PSO_EXTENSION_DESC_VER;
        desc.NumCustomSemantics = 2;

        ret.mCustomSemantics.resize(2);

        //desc.pCustomSemantics = (NV_CUSTOM_SEMANTIC *)malloc(2 * sizeof(NV_CUSTOM_SEMANTIC));
        memset(ret.mCustomSemantics.data(), 0, (2 * sizeof(NV_CUSTOM_SEMANTIC)));

        ret.mCustomSemantics[0].version = NV_CUSTOM_SEMANTIC_VERSION;
        ret.mCustomSemantics[0].NVCustomSemanticType = NV_X_RIGHT_SEMANTIC;
        strcpy_s(&(ret.mCustomSemantics[0].NVCustomSemanticNameString[0]), NVAPI_LONG_STRING_MAX, "NV_X_RIGHT");

        ret.mCustomSemantics[1].version = NV_CUSTOM_SEMANTIC_VERSION;
        ret.mCustomSemantics[1].NVCustomSemanticType = NV_VIEWPORT_MASK_SEMANTIC;
        strcpy_s(&(ret.mCustomSemantics[1].NVCustomSemanticNameString[0]), NVAPI_LONG_STRING_MAX, "NV_VIEWPORT_MASK");

        desc.pCustomSemantics = ret.mCustomSemantics.data();
    }

    inline void createNvApiUavSlotExDesc(NvApiPsoExDesc& ret, uint32_t uavSlot)
    {
        ret.psoExtension = NV_PSO_SET_SHADER_EXTNENSION_SLOT_AND_SPACE;

        auto& desc = ret.mExtSlotDesc;
        std::memset(&desc, 0, sizeof(desc));

        desc.psoExtension = NV_PSO_SET_SHADER_EXTNENSION_SLOT_AND_SPACE;
        desc.version = NV_SET_SHADER_EXTENSION_SLOT_DESC_VER;
        desc.baseVersion = NV_PSO_EXTENSION_DESC_VER;
        desc.uavSlot = uavSlot;
        desc.registerSpace = 0; // SM5.1+: If the "space" keyword is omitted, the default space index of 0 is implicitly assigned
    }

#else
    using NvApiPsoExDesc = uint32_t;
    inline void createNvApiVsExDesc(NvApiPsoExDesc& ret) { should_not_get_here(); }
    inline void createNvApiUavSlotExDesc(NvApiPsoExDesc& ret, uint32_t uavSlot) { should_not_get_here(); }
#endif
}
