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
#include "PBRTConductorMaterial.h"
#include "Utils/Scripting/ScriptBindings.h"

namespace Falcor
{
    namespace
    {
        const char kShaderFile[] = "Rendering/Materials/PBRT/PBRTConductorMaterial.slang";
    }

    PBRTConductorMaterial::SharedPtr PBRTConductorMaterial::create(const std::string& name)
    {
        return SharedPtr(new PBRTConductorMaterial(name));
    }

    PBRTConductorMaterial::PBRTConductorMaterial(const std::string& name)
        : BasicMaterial(name, MaterialType::PBRTConductor)
    {
        // Setup additional texture slots.
        mTextureSlotInfo[(uint32_t)TextureSlot::BaseColor] = { "baseColor", TextureChannelFlags::RGBA, false };
        mTextureSlotInfo[(uint32_t)TextureSlot::Transmission] = { "transmission", TextureChannelFlags::RGB, false };
        mTextureSlotInfo[(uint32_t)TextureSlot::Specular] = { "specular", TextureChannelFlags::Red | TextureChannelFlags::Green, false };
        mTextureSlotInfo[(uint32_t)TextureSlot::Normal] = { "normal", TextureChannelFlags::RGB, false };
    }

    Program::ShaderModuleList PBRTConductorMaterial::getShaderModules() const
    {
        return { Program::ShaderModule(kShaderFile) };
    }

    Program::TypeConformanceList PBRTConductorMaterial::getTypeConformances() const
    {
        return { {{"PBRTConductorMaterial", "IMaterial"}, (uint32_t)MaterialType::PBRTConductor} };
    }

    void PBRTConductorMaterial::renderSpecularUI(Gui::Widgets& widget)
    {
        float2 roughness = getRoughness();
        if (widget.var("X Roughness", roughness.x, 0.f, 1.f, 0.01f)) setRoughness(roughness);
        if (widget.var("Y Roughness", roughness.y, 0.f, 1.f, 0.01f)) setRoughness(roughness);
    }

    void PBRTConductorMaterial::setRoughness(float2 roughness)
    {
        if (mData.specular[0] != (float16_t)roughness.x || mData.specular[1] != (float16_t)roughness.y)
        {
            mData.specular[0] = (float16_t)roughness.x;
            mData.specular[1] = (float16_t)roughness.y;
            markUpdates(UpdateFlags::DataChanged);
        }
    }

    FALCOR_SCRIPT_BINDING(PBRTConductorMaterial)
    {
        using namespace pybind11::literals;

        FALCOR_SCRIPT_BINDING_DEPENDENCY(BasicMaterial)

        pybind11::class_<PBRTConductorMaterial, BasicMaterial, PBRTConductorMaterial::SharedPtr> material(m, "PBRTConductorMaterial");
        material.def(pybind11::init(&PBRTConductorMaterial::create), "name"_a = "");

        material.def_property("roughness", &PBRTConductorMaterial::getRoughness, &PBRTConductorMaterial::setRoughness);
    }
}
