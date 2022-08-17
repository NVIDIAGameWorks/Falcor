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
    /** Class representing a cloth material.

        Texture channel layout:

            BaseColor
                - RGB - Base Color
                - A   - Opacity
            Specular
                - R - Unused
                - G - Roughness
                - B - Unused
                - A - Unused
            Normal
                - 3-Channel standard normal map, or 2-Channel BC5 format

        See additional texture channels defined in BasicMaterial.
    */
    class FALCOR_API ClothMaterial : public BasicMaterial
    {
    public:
        using SharedPtr = std::shared_ptr<ClothMaterial>;

        /** Create a new cloth material.
            \param[in] name The material name.
        */
        static SharedPtr create(const std::string& name = "");

        Program::ShaderModuleList getShaderModules() const override;
        Program::TypeConformanceList getTypeConformances() const override;

        /** Set the roughness.
        */
        void setRoughness(float roughness);

        /** Get the roughness.
        */
        float getRoughness() const { return (float)mData.specular[1]; }

    protected:
        ClothMaterial(const std::string& name);

        void renderSpecularUI(Gui::Widgets& widget) override;
    };
}
