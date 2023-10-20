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
#include "PBRTCoatedConductorMaterial.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "GlobalState.h"

namespace Falcor
{
    namespace
    {
        const char kShaderFile[] = "Rendering/Materials/PBRT/PBRTCoatedConductorMaterial.slang";
    }

    PBRTCoatedConductorMaterial::PBRTCoatedConductorMaterial(ref<Device> pDevice, const std::string& name)
        : BasicMaterial(pDevice, name, MaterialType::PBRTCoatedConductor)
    {
        // Setup additional texture slots.
        mTextureSlotInfo[(uint32_t)TextureSlot::BaseColor] = { "baseColor", TextureChannelFlags::RGBA, false };
        mTextureSlotInfo[(uint32_t)TextureSlot::Transmission] = { "transmission", TextureChannelFlags::RGB, false };
        mTextureSlotInfo[(uint32_t)TextureSlot::Specular] = { "specular", TextureChannelFlags::RGBA, false };
        mTextureSlotInfo[(uint32_t)TextureSlot::Normal] = { "normal", TextureChannelFlags::RGB, false };
    }

    ProgramDesc::ShaderModuleList PBRTCoatedConductorMaterial::getShaderModules() const
    {
        return { ProgramDesc::ShaderModule::fromFile(kShaderFile) };
    }

    TypeConformanceList PBRTCoatedConductorMaterial::getTypeConformances() const
    {
        return { {{"PBRTCoatedConductorMaterial", "IMaterial"}, (uint32_t)MaterialType::PBRTCoatedConductor} };
    }

    void PBRTCoatedConductorMaterial::renderSpecularUI(Gui::Widgets& widget)
    {
        float4 roughness = getRoughness();
        if (widget.var("Interface X Roughness", roughness.x, 0.f, 1.f, 0.01f)) setRoughness(roughness);
        if (widget.var("Interface Y Roughness", roughness.y, 0.f, 1.f, 0.01f)) setRoughness(roughness);
        if (widget.var("Conductor X Roughness", roughness.z, 0.f, 1.f, 0.01f)) setRoughness(roughness);
        if (widget.var("Conductor Y Roughness", roughness.w, 0.f, 1.f, 0.01f)) setRoughness(roughness);
    }

    void PBRTCoatedConductorMaterial::setRoughness(float4 roughness)
    {
        if (mData.specular[0] != (float16_t)roughness.x || mData.specular[1] != (float16_t)roughness.y ||
            mData.specular[2] != (float16_t)roughness.z || mData.specular[3] != (float16_t)roughness.w)
        {
            mData.specular[0] = (float16_t)roughness.x;
            mData.specular[1] = (float16_t)roughness.y;
            mData.specular[2] = (float16_t)roughness.z;
            mData.specular[3] = (float16_t)roughness.w;
            markUpdates(UpdateFlags::DataChanged);
        }
    }

    FALCOR_SCRIPT_BINDING(PBRTCoatedConductorMaterial)
    {
        using namespace pybind11::literals;

        FALCOR_SCRIPT_BINDING_DEPENDENCY(BasicMaterial)

        pybind11::class_<PBRTCoatedConductorMaterial, BasicMaterial, ref<PBRTCoatedConductorMaterial>> material(m, "PBRTCoatedConductorMaterial");
        auto create = [] (const std::string& name)
        {
            return PBRTCoatedConductorMaterial::create(accessActivePythonSceneBuilder().getDevice(), name);
        };
        material.def(pybind11::init(create), "name"_a = ""); // PYTHONDEPRECATED

        material.def_property("roughness", &PBRTCoatedConductorMaterial::getRoughness, &PBRTCoatedConductorMaterial::setRoughness);
    }
}
