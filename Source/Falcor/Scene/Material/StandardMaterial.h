/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "BasicMaterial.h"

namespace Falcor
{
    /** Class representing the standard material.

        Texture channel layout:

        ShadingModel::MetalRough
            BaseColor
                - RGB - Base Color
                - A   - Opacity
            Specular
                - R   - Unused
                - G   - Roughness
                - B   - Metallic
                - A   - Unused

        ShadingModel::SpecGloss
            BaseColor
                - RGB - Diffuse Color
                - A   - Opacity
            Specular
                - RGB - Specular Color
                - A   - Gloss

        In all shading models:

            Normal
                - 3-Channel standard normal map, or 2-Channel BC5 format
            Emissive
                - RGB - Emissive Color
                - A   - Unused
            Transmission
                - RGB - Transmission color
                - A   - Unused

        See additional texture channels defined in BasicMaterial.
    */
    class FALCOR_API StandardMaterial : public BasicMaterial
    {
    public:
        using SharedPtr = std::shared_ptr<StandardMaterial>;

        /** Create a new standard material.
            \param[in] name The material name.
            \param[in] model Shading model.
        */
        static SharedPtr create(const std::string& name = "", ShadingModel shadingModel = ShadingModel::MetalRough);

        /** Render the UI.
            \return True if the material was modified.
        */
        bool renderUI(Gui::Widgets& widget) override;

        Program::ShaderModuleList getShaderModules() const override;
        Program::TypeConformanceList getTypeConformances() const override;

        /** Get the shading model.
        */
        ShadingModel getShadingModel() const { return mData.getShadingModel(); }

        /** Set the roughness.
            Only available for metallic/roughness shading model.
        */
        void setRoughness(float roughness);

        /** Get the roughness.
            Only available for metallic/roughness shading model.
        */
        float getRoughness() const { return getShadingModel() == ShadingModel::MetalRough ? (float)mData.specular[1] : 0.f; }

        /** Set the metallic value.
            Only available for metallic/roughness shading model.
        */
        void setMetallic(float metallic);

        /** Get the metallic value.
            Only available for metallic/roughness shading model.
        */
        float getMetallic() const { return getShadingModel() == ShadingModel::MetalRough ? (float)mData.specular[2] : 0.f; }

        /** Set the emissive color.
        */
        void setEmissiveColor(const float3& color) override;

        /** Set the emissive factor.
        */
        void setEmissiveFactor(float factor);

        /** Get the emissive color.
        */
        float3 getEmissiveColor() const { return mData.emissive; }

        /** Get the emissive factor.
        */
        float getEmissiveFactor() const { return mData.emissiveFactor; }

        // DEMO21: The mesh will use the global IES profile (LightProfile) to modulate its emission
        void setLightProfileEnabled( bool enabled )
        {
            mHeader.setEnableLightProfile( enabled );
        }

    protected:
        StandardMaterial(const std::string& name, ShadingModel shadingModel);

        void updateDeltaSpecularFlag() override;

        void renderSpecularUI(Gui::Widgets& widget) override;
        void setShadingModel(ShadingModel model);
    };
}
