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
#include "Scene/Material/BasicMaterial.h"

namespace Falcor
{
    /** This class implements a conductor material. This means the
        surface is purely reflective, and its albedo is dictated by
        the conductor Fresnel equations (i.e. by a complex IoR).
        The material can be act like a mirror (if roughness is 0),
        or a (potentially anisotropic) GGX microfacet.

        This class perfectly matches the PBRT "conductor" material.

        Texture channel layout:

            BaseColor
                - RGB - Complex Eta
                - A   - Opacity
            Transmission
                - RGB - Complex k
            Specular
                - R - X Roughness
                - G - Y Roughness
                - B - Unused
                - A - Unused
            Normal
                - 3-Channel standard normal map, or 2-Channel BC5 format

        See additional texture channels defined in BasicMaterial.
    */
    class FALCOR_API PBRTConductorMaterial : public BasicMaterial
    {
        FALCOR_OBJECT(PBRTConductorMaterial)
    public:
        static ref<PBRTConductorMaterial> create(ref<Device> pDevice, const std::string& name) { return make_ref<PBRTConductorMaterial>(pDevice, name); }

        PBRTConductorMaterial(ref<Device> pDevice, const std::string& name);

        ProgramDesc::ShaderModuleList getShaderModules() const override;
        TypeConformanceList getTypeConformances() const override;

        /** Set the roughness.
        */
        void setRoughness(float2 roughness);

        /** Get the roughness.
        */
        float2 getRoughness() const { return float2(mData.specular[0], mData.specular[1]); }

        const MaterialParamLayout& getParamLayout() const override;
        SerializedMaterialParams serializeParams() const override;
        void deserializeParams(const SerializedMaterialParams& params) override;

    protected:
        void renderSpecularUI(Gui::Widgets& widget) override;
    };
}
