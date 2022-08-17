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
#include "HairMaterial.h"
#include "Utils/Scripting/ScriptBindings.h"

namespace Falcor
{
    namespace
    {
        const char kShaderFile[] = "Rendering/Materials/HairMaterial.slang";
    }

    HairMaterial::SharedPtr HairMaterial::create(const std::string& name)
    {
        return SharedPtr(new HairMaterial(name));
    }

    HairMaterial::HairMaterial(const std::string& name)
        : BasicMaterial(name, MaterialType::Hair)
    {
        // Setup additional texture slots.
        mTextureSlotInfo[(uint32_t)TextureSlot::BaseColor] = { "baseColor", TextureChannelFlags::RGB, true }; // Note: No alpha support
        mTextureSlotInfo[(uint32_t)TextureSlot::Specular] = { "specular", TextureChannelFlags::RGB, false };
    }

    Program::ShaderModuleList HairMaterial::getShaderModules() const
    {
        return { Program::ShaderModule(kShaderFile) };
    }

    Program::TypeConformanceList HairMaterial::getTypeConformances() const
    {
        return { {{"HairMaterial", "IMaterial"}, (uint32_t)MaterialType::Hair} };
    }

    float3 HairMaterial::sigmaAFromConcentration(float ce, float cp)
    {
        float3 eumelaninSigmaA(0.419f, 0.697f, 1.37f);
        float3 pheomelaninSigmaA(0.187f, 0.4f, 1.05f);
        return ce * eumelaninSigmaA + cp * pheomelaninSigmaA;
    }

    float3 HairMaterial::sigmaAFromColor(float3 color, float betaN)
    {
        const float tmp = 5.969f - 0.215f * betaN + 2.532f * betaN * betaN - 10.73f * std::pow(betaN, 3) + 5.574f * std::pow(betaN, 4) + 0.245f * std::pow(betaN, 5);
        float3 sqrtSigmaA = log(max(color, 1e-4f)) / tmp;
        return sqrtSigmaA * sqrtSigmaA;
    }

    float3 HairMaterial::colorFromSigmaA(float3 sigmaA, float betaN)
    {
        const float tmp = 5.969f - 0.215f * betaN + 2.532f * betaN * betaN - 10.73f * std::pow(betaN, 3) + 5.574f * std::pow(betaN, 4) + 0.245f * std::pow(betaN, 5);
        return exp(sqrt(sigmaA) * tmp);
    }

    FALCOR_SCRIPT_BINDING(HairMaterial)
    {
        using namespace pybind11::literals;
        FALCOR_SCRIPT_BINDING_DEPENDENCY(BasicMaterial)

        pybind11::class_<HairMaterial, BasicMaterial, HairMaterial::SharedPtr> material(m, "HairMaterial");
        material.def(pybind11::init(&HairMaterial::create), "name"_a = "");
    }
}
