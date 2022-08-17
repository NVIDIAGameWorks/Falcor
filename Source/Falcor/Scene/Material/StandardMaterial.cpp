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
#include "StandardMaterial.h"
#include "Utils/Logger.h"
#include "Utils/Scripting/ScriptBindings.h"

namespace Falcor
{
    namespace
    {
        const char kShaderFile[] = "Rendering/Materials/StandardMaterial.slang";
    }

    StandardMaterial::SharedPtr StandardMaterial::create(const std::string& name, ShadingModel shadingModel)
    {
        return SharedPtr(new StandardMaterial(name, shadingModel));
    }

    StandardMaterial::StandardMaterial(const std::string& name, ShadingModel shadingModel)
        : BasicMaterial(name, MaterialType::Standard)
    {
        setShadingModel(shadingModel);
        bool specGloss = getShadingModel() == ShadingModel::SpecGloss;

        // Setup additional texture slots.
        mTextureSlotInfo[(uint32_t)TextureSlot::BaseColor] = { specGloss ? "diffuse" : "baseColor", TextureChannelFlags::RGBA, true };
        mTextureSlotInfo[(uint32_t)TextureSlot::Specular] = specGloss ? TextureSlotInfo{ "specular", TextureChannelFlags::RGBA, true } : TextureSlotInfo{ "spec", TextureChannelFlags::Green | TextureChannelFlags::Blue, false };
        mTextureSlotInfo[(uint32_t)TextureSlot::Normal] = { "normal", TextureChannelFlags::RGB, false };
        mTextureSlotInfo[(uint32_t)TextureSlot::Emissive] = { "emissive", TextureChannelFlags::RGB, true };
        mTextureSlotInfo[(uint32_t)TextureSlot::Transmission] = { "transmission", TextureChannelFlags::RGB, true };
    }

    bool StandardMaterial::renderUI(Gui::Widgets& widget)
    {
        widget.text("Shading model: " + to_string(getShadingModel()));

        // Render the base class UI first.
        bool changed = BasicMaterial::renderUI(widget);

        // We're re-using the material's update flags here to track changes.
        // Cache the previous flag so we can restore it before returning.
        UpdateFlags prevUpdates = mUpdates;
        mUpdates = UpdateFlags::None;

        if (auto pTexture = getEmissiveTexture())
        {
            widget.text("Emissive color: " + pTexture->getSourcePath().string());
            widget.text("Texture info: " + std::to_string(pTexture->getWidth()) + "x" + std::to_string(pTexture->getHeight()) + " (" + to_string(pTexture->getFormat()) + ")");
            widget.image("Emissive color", pTexture, float2(100.f));
            if (widget.button("Remove texture##Emissive")) setEmissiveTexture(nullptr);
        }
        else
        {
            float3 emissiveColor = getEmissiveColor();
            if (widget.var("Emissive color", emissiveColor, 0.f, 1.f, 0.01f)) setEmissiveColor(emissiveColor);
        }

        float emissiveFactor = getEmissiveFactor();
        if (widget.var("Emissive factor", emissiveFactor, 0.f, std::numeric_limits<float>::max(), 0.01f)) setEmissiveFactor(emissiveFactor);

        // Restore update flags.
        changed |= mUpdates != UpdateFlags::None;
        markUpdates(prevUpdates | mUpdates);

        return changed;
    }

    void StandardMaterial::updateDeltaSpecularFlag()
    {
        // Check if material has no diffuse lobe.
        bool isNonDiffuse = !hasTextureSlotData(TextureSlot::BaseColor) && float3(getBaseColor()) == float3(0.f) && getDiffuseTransmission() == 0.f;

        // Check if material is fully specular transmissive.
        bool isFullyTransmissive = getSpecularTransmission() >= 1.f;

        // Check if material only has delta reflection/transmission.
        bool isDelta = false;
        if (getShadingModel() == ShadingModel::MetalRough && !hasTextureSlotData(TextureSlot::Specular))
        {
            isDelta = getSpecularParams().g == 0.f; // Green component stores roughness in MetalRough mode.
            if (getSpecularParams().b >= 1.f) isNonDiffuse = true; // Blue component stores metallic in MetalRough mode. If 1.0 there is no diffuse lobe.
        }

        bool isDeltaSpecular = isDelta && (isNonDiffuse || isFullyTransmissive);

        if (mHeader.isDeltaSpecular() != isDeltaSpecular)
        {
            mHeader.setDeltaSpecular(isDeltaSpecular);
            markUpdates(UpdateFlags::DataChanged);
        }
    }

    void StandardMaterial::renderSpecularUI(Gui::Widgets& widget)
    {
        if (getShadingModel() == ShadingModel::MetalRough)
        {
            float roughness = getRoughness();
            if (widget.var("Roughness", roughness, 0.f, 1.f, 0.01f)) setRoughness(roughness);

            float metallic = getMetallic();
            if (widget.var("Metallic", metallic, 0.f, 1.f, 0.01f)) setMetallic(metallic);
        }
    }

    Program::ShaderModuleList StandardMaterial::getShaderModules() const
    {
        return { Program::ShaderModule(kShaderFile) };
    }

    Program::TypeConformanceList StandardMaterial::getTypeConformances() const
    {
        return { {{"StandardMaterial", "IMaterial"}, (uint32_t)MaterialType::Standard} };
    }

    void StandardMaterial::setShadingModel(ShadingModel model)
    {
        checkArgument(model == ShadingModel::MetalRough || model == ShadingModel::SpecGloss, "'model' must be MetalRough or SpecGloss");

        if (getShadingModel() != model)
        {
            mData.setShadingModel(model);
            markUpdates(UpdateFlags::DataChanged);
            updateDeltaSpecularFlag();
        }
    }

    void StandardMaterial::setRoughness(float roughness)
    {
        if (getShadingModel() != ShadingModel::MetalRough)
        {
            logWarning("Ignoring setRoughness(). Material '{}' does not use the metallic/roughness shading model.", mName);
            return;
        }

        if (mData.specular[1] != (float16_t)roughness)
        {
            mData.specular[1] = (float16_t)roughness;
            markUpdates(UpdateFlags::DataChanged);
            updateDeltaSpecularFlag();
        }
    }

    void StandardMaterial::setMetallic(float metallic)
    {
        if (getShadingModel() != ShadingModel::MetalRough)
        {
            logWarning("Ignoring setMetallic(). Material '{}' does not use the metallic/roughness shading model.", mName);
            return;
        }

        if (mData.specular[2] != (float16_t)metallic)
        {
            mData.specular[2] = (float16_t)metallic;
            markUpdates(UpdateFlags::DataChanged);
            updateDeltaSpecularFlag();
        }
    }

    void StandardMaterial::setEmissiveColor(const float3& color)
    {
        if (mData.emissive != color)
        {
            mData.emissive = color;
            markUpdates(UpdateFlags::DataChanged);
            updateEmissiveFlag();
        }
    }

    void StandardMaterial::setEmissiveFactor(float factor)
    {
        if (mData.emissiveFactor != factor)
        {
            mData.emissiveFactor = factor;
            markUpdates(UpdateFlags::DataChanged);
            updateEmissiveFlag();
        }
    }

    FALCOR_SCRIPT_BINDING(StandardMaterial)
    {
        using namespace pybind11::literals;

        FALCOR_SCRIPT_BINDING_DEPENDENCY(BasicMaterial)

        pybind11::enum_<ShadingModel> shadingModel(m, "ShadingModel");
        shadingModel.value("MetalRough", ShadingModel::MetalRough);
        shadingModel.value("SpecGloss", ShadingModel::SpecGloss);

        pybind11::class_<StandardMaterial, BasicMaterial, StandardMaterial::SharedPtr> material(m, "StandardMaterial");
        material.def(pybind11::init(&StandardMaterial::create), "name"_a = "", "model"_a = ShadingModel::MetalRough);

        material.def_property("roughness", &StandardMaterial::getRoughness, &StandardMaterial::setRoughness);
        material.def_property("metallic", &StandardMaterial::getMetallic, &StandardMaterial::setMetallic);
        material.def_property("emissiveColor", &StandardMaterial::getEmissiveColor, &StandardMaterial::setEmissiveColor);
        material.def_property("emissiveFactor", &StandardMaterial::getEmissiveFactor, &StandardMaterial::setEmissiveFactor);
        material.def_property_readonly("shadingModel", &StandardMaterial::getShadingModel);

        // Register alias Material -> StandardMaterial to allow deprecated script syntax.
        // TODO: Remove workaround when all scripts have been updated to create StandardMaterial directly.
        m.attr("Material") = m.attr("StandardMaterial"); // PYTHONDEPRECATED
    }
}
