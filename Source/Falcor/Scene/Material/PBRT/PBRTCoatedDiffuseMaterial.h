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
    /** This class implements a coated diffuse material, i.e. a
        dielectric coating on top of a Lamebrtian base.
        The coating can be smooth or rough, isotropic or anisotropic.
        The material simulates interreflection between the layers
        using a Monte Carlo random walk.

        This class perfectly matches the PBRT "coateddiffuse" material.

        Texture channel layout:

            BaseColor
                - RGB - Diffuse albedo
                - A   - Opacity
            Specular
                - R - Interface X Roughness
                - G - Interface Y Roughness
            Normal
                - 3-Channel standard normal map, or 2-Channel BC5 format

        See additional texture channels defined in BasicMaterial.
    */
    class FALCOR_API PBRTCoatedDiffuseMaterial : public BasicMaterial
    {
        FALCOR_OBJECT(PBRTCoatedDiffuseMaterial)
    public:
        static ref<PBRTCoatedDiffuseMaterial> create(ref<Device> pDevice, const std::string& name) { return make_ref<PBRTCoatedDiffuseMaterial>(pDevice, name); }

        PBRTCoatedDiffuseMaterial(ref<Device> pDevice, const std::string& name);

        ProgramDesc::ShaderModuleList getShaderModules() const override;
        TypeConformanceList getTypeConformances() const override;

        /** Set the roughness.
        */
        void setRoughness(float2 roughness);

        /** Get the roughness.
        */
        float2 getRoughness() const { return float2(mData.specular[0], mData.specular[1]); }

    protected:
        void renderSpecularUI(Gui::Widgets& widget) override;
    };
}
