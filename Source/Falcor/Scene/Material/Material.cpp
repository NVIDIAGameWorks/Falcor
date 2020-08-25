/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "stdafx.h"
#include "Material.h"
#include "Core/Program/GraphicsProgram.h"
#include "Core/Program/ProgramVars.h"
#include "Utils/Color/ColorHelpers.slang"

namespace Falcor
{
    static_assert(sizeof(MaterialData) % 16 == 0, "Material::MaterialData size should be a multiple of 16");

    Material::UpdateFlags Material::sGlobalUpdates = Material::UpdateFlags::None;

    Material::Material(const std::string& name) : mName(name)
    {
    }

    Material::SharedPtr Material::create(const std::string& name)
    {
        Material* pMaterial = new Material(name);
        return SharedPtr(pMaterial);
    }

    Material::~Material() = default;

    bool Material::renderUI(Gui::Widgets& widget)
    {
        // We're re-using the material's update flags here to track changes.
        // Cache the previous flag so we can restore it before returning.
        UpdateFlags prevUpdates = mUpdates;
        mUpdates = UpdateFlags::None;

        widget.text("Shading model:");
        if (getShadingModel() == ShadingModelMetalRough) widget.text("MetalRough", true);
        else if (getShadingModel() == ShadingModelSpecGloss) widget.text("SpecGloss", true);
        else should_not_get_here();

        if (const auto& tex = getBaseColorTexture(); tex != nullptr)
        {
            widget.text("Base color: " + tex->getSourceFilename());
            widget.text("Texture info: " + std::to_string(tex->getWidth()) + "x" + std::to_string(tex->getHeight()) + " (" + to_string(tex->getFormat()) + ")");
            widget.image("Base color", tex, float2(100.f));
            if (widget.button("Remove texture##BaseColor")) setBaseColorTexture(nullptr);
        }
        else
        {
            float4 baseColor = getBaseColor();
            if (widget.var("Base color", baseColor, 0.f, 1.f, 0.01f)) setBaseColor(baseColor);
        }

        if (const auto& tex = getSpecularTexture(); tex != nullptr)
        {
            widget.text("Specular params: " + tex->getSourceFilename());
            widget.text("Texture info: " + std::to_string(tex->getWidth()) + "x" + std::to_string(tex->getHeight()) + " (" + to_string(tex->getFormat()) + ")");
            widget.image("Specular params", tex, float2(100.f));
            if (widget.button("Remove texture##Specular")) setSpecularTexture(nullptr);
        }
        else
        {
            float4 specularParams = getSpecularParams();
            if (widget.var("Specular params", specularParams, 0.f, 1.f, 0.01f)) setSpecularParams(specularParams);
            widget.tooltip("The encoding depends on the shading model:\n\n"
                "MetalRough:\n"
                "    occlusion (R), roughness (G), metallic (B)\n\n"
                "SpecGloss:\n"
                "    specular color(RGB) and glossiness(A)", true);

            if (getShadingModel() == ShadingModelMetalRough)
            {
                float roughness = getRoughness();
                if (widget.var("Roughness", roughness, 0.f, 1.f, 0.01f)) setRoughness(roughness);

                float metallic = getMetallic();
                if (widget.var("Metallic", metallic, 0.f, 1.f, 0.01f)) setMetallic(metallic);
            }
        }

        if (const auto& tex = getNormalMap(); tex != nullptr)
        {
            widget.text("Normal map: " + tex->getSourceFilename());
            widget.text("Texture info: " + std::to_string(tex->getWidth()) + "x" + std::to_string(tex->getHeight()) + " (" + to_string(tex->getFormat()) + ")");
            widget.image("Normal map", tex, float2(100.f));
            if (widget.button("Remove texture##NormalMap")) setNormalMap(nullptr);
        }

        if (const auto& tex = getEmissiveTexture(); tex != nullptr)
        {
            widget.text("Emissive color: " + tex->getSourceFilename());
            widget.text("Texture info: " + std::to_string(tex->getWidth()) + "x" + std::to_string(tex->getHeight()) + " (" + to_string(tex->getFormat()) + ")");
            widget.image("Emissive color", tex, float2(100.f));
            if (widget.button("Remove texture##Emissive")) setEmissiveTexture(nullptr);
        }
        else
        {
            float3 emissiveColor = getEmissiveColor();
            if (widget.var("Emissive color", emissiveColor, 0.f, 1.f, 0.01f)) setEmissiveColor(emissiveColor);
        }

        float emissiveFactor = getEmissiveFactor();
        if (widget.var("Emissive factor", emissiveFactor, 0.f, std::numeric_limits<float>::max(), 0.01f)) setEmissiveFactor(emissiveFactor);

        if (const auto& tex = getSpecularTransmissionTexture(); tex != nullptr)
        {
            widget.text("Specular transmission: " + tex->getSourceFilename());
            widget.text("Texture info: " + std::to_string(tex->getWidth()) + "x" + std::to_string(tex->getHeight()) + " (" + to_string(tex->getFormat()) + ")");
            widget.image("Specular transmission", tex, float2(100.f));
            if (widget.button("Remove texture##Transmission")) setSpecularTransmissionTexture(nullptr);
        }
        else
        {
            float specTransmission = getSpecularTransmission();
            if (widget.var("Specular transmission", specTransmission, 0.f, 1.f, 0.01f)) setSpecularTransmission(specTransmission);
        }

        float IoR = getIndexOfRefraction();
        if (widget.var("Index of refraction", IoR, 1.f, std::numeric_limits<float>::max(), 0.01f)) setIndexOfRefraction(IoR);

        bool doubleSided = isDoubleSided();
        if (widget.checkbox("Double-sided", doubleSided)) setDoubleSided(doubleSided);

        // Restore update flags.
        bool changed = mUpdates != UpdateFlags::None;
        markUpdates(prevUpdates | mUpdates);

        return changed;
    }

    void Material::setShadingModel(uint32_t model)
    {
        setFlags(PACK_SHADING_MODEL(mData.flags, model));
    }

    void Material::setAlphaMode(uint32_t alphaMode)
    {
        setFlags(PACK_ALPHA_MODE(mData.flags, alphaMode));
    }

    void Material::setDoubleSided(bool doubleSided)
    {
        setFlags(PACK_DOUBLE_SIDED(mData.flags, doubleSided ? 1 : 0));
    }

    void Material::setAlphaThreshold(float alpha)
    {
        if (mData.alphaThreshold != alpha)
        {
            mData.alphaThreshold = alpha;
            markUpdates(UpdateFlags::DataChanged);
        }
    }

    void Material::setIndexOfRefraction(float IoR)
    {
        if (mData.IoR != IoR)
        {
            mData.IoR = IoR;
            markUpdates(UpdateFlags::DataChanged);
        }
    }

    void Material::setNestedPriority(uint32_t priority)
    {
        const uint32_t maxPriority = (1U << NESTED_PRIORITY_BITS) - 1;
        if (priority > maxPriority)
        {
            logWarning("Requested nested priority " + std::to_string(priority) + " for material '" + mName + "' is out of range. Clamping to " + std::to_string(maxPriority) + ".");
            priority = maxPriority;
        }
        setFlags(PACK_NESTED_PRIORITY(mData.flags, priority));
    }

    void Material::setSampler(Sampler::SharedPtr pSampler)
    {
        if (pSampler != mResources.samplerState)
        {
            mResources.samplerState = pSampler;
            markUpdates(UpdateFlags::ResourcesChanged);
        }
    }

    void Material::setTexture(TextureSlot slot, Texture::SharedPtr pTexture)
    {
        switch (slot)
        {
        case TextureSlot::BaseColor:
            setBaseColorTexture(pTexture);
            break;
        case TextureSlot::Specular:
            setSpecularTexture(pTexture);
            break;
        case TextureSlot::Emissive:
            setEmissiveTexture(pTexture);
            break;
        case TextureSlot::Normal:
            setNormalMap(pTexture);
            break;
        case TextureSlot::Occlusion:
            setOcclusionMap(pTexture);
            break;
        case TextureSlot::SpecularTransmission:
            setSpecularTransmissionTexture(pTexture);
            break;
        default:
            should_not_get_here();
        }
    }

    Texture::SharedPtr Material::getTexture(TextureSlot slot) const
    {
        switch (slot)
        {
        case TextureSlot::BaseColor:
            return getBaseColorTexture();
        case TextureSlot::Specular:
            return getSpecularTexture();
        case TextureSlot::Emissive:
            return getEmissiveTexture();
        case TextureSlot::Normal:
            return getNormalMap();
        case TextureSlot::Occlusion:
            return getOcclusionMap();
        case TextureSlot::SpecularTransmission:
            return getSpecularTransmissionTexture();
        default:
            should_not_get_here();
        }
        return nullptr;
    }

    uint2 Material::getMaxTextureDimensions() const
    {
        uint2 dim = uint2(0);
        for (uint32_t i = 0; i < (uint32_t)TextureSlot::Count; i++)
        {
            const auto& t = getTexture((TextureSlot)i);
            if (t) dim = max(dim, uint2(t->getWidth(), t->getHeight()));
        }
        return dim;
    }

    void Material::loadTexture(TextureSlot slot, const std::string& filename, bool useSrgb)
    {
        std::string fullpath;
        if (findFileInDataDirectories(filename, fullpath))
        {
            auto texture = Texture::createFromFile(fullpath, true, useSrgb && isSrgbTextureRequired(slot));
            if (texture)
            {
                setTexture(slot, texture);
                // Flush and sync in order to prevent the upload heap from growing too large. Doing so after
                // every texture creation is overly conservative, and will likely lead to performance issues
                // due to the forced CPU/GPU sync.
                gpDevice->flushAndSync();
            }
        }
    }

    void Material::clearTexture(TextureSlot slot)
    {
        setTexture(slot, nullptr);
    }

    bool Material::isSrgbTextureRequired(TextureSlot slot)
    {
        uint32_t shadingModel = getShadingModel();

        switch (slot)
        {
        case TextureSlot::Specular:
            return (shadingModel == ShadingModelSpecGloss);
        case TextureSlot::BaseColor:
        case TextureSlot::Emissive:
        case TextureSlot::Occlusion:
            return true;
        case TextureSlot::Normal:
            return false;
        default:
            should_not_get_here();
            return false;
        }
    }

    void Material::setBaseColorTexture(Texture::SharedPtr pBaseColor)
    {
        if (mResources.baseColor != pBaseColor)
        {
            mResources.baseColor = pBaseColor;
            markUpdates(UpdateFlags::ResourcesChanged);
            updateBaseColorType();
            bool hasAlpha = pBaseColor && doesFormatHasAlpha(pBaseColor->getFormat());
            setAlphaMode(hasAlpha ? AlphaModeMask : AlphaModeOpaque);
        }
    }

    void Material::setSpecularTexture(Texture::SharedPtr pSpecular)
    {
        if (mResources.specular != pSpecular)
        {
            mResources.specular = pSpecular;
            markUpdates(UpdateFlags::ResourcesChanged);
            updateSpecularType();
        }
    }

    void Material::setEmissiveTexture(const Texture::SharedPtr& pEmissive)
    {
        if (mResources.emissive != pEmissive)
        {
            mResources.emissive = pEmissive;
            markUpdates(UpdateFlags::ResourcesChanged);
            updateEmissiveType();
        }
    }

    void Material::setSpecularTransmissionTexture(const Texture::SharedPtr& pSpecularTransmission)
    {
        if (mResources.specularTransmission != pSpecularTransmission)
        {
            mResources.specularTransmission = pSpecularTransmission;
            markUpdates(UpdateFlags::ResourcesChanged);
            updateSpecularTransmissionType();
        }
    }


    void Material::setBaseColor(const float4& color)
    {
        if (mData.baseColor != color)
        {
            mData.baseColor = color;
            markUpdates(UpdateFlags::DataChanged);
            updateBaseColorType();
        }
    }

    void Material::setSpecularParams(const float4& color)
    {
        if (mData.specular != color)
        {
            mData.specular = color;
            markUpdates(UpdateFlags::DataChanged);
            updateSpecularType();
        }
    }

    void Material::setRoughness(float roughness)
    {
        if (getShadingModel() != ShadingModelMetalRough)
        {
            logWarning("Ignoring setRoughness(). Material '" + mName + "' does not use the metallic/roughness shading model.");
            return;
        }

        if (mData.specular.g != roughness)
        {
            mData.specular.g = roughness;
            markUpdates(UpdateFlags::DataChanged);
            updateSpecularType();
        }
    }

    void Material::setMetallic(float metallic)
    {
        if (getShadingModel() != ShadingModelMetalRough)
        {
            logWarning("Ignoring setMetallic(). Material '" + mName + "' does not use the metallic/roughness shading model.");
            return;
        }

        if (mData.specular.b != metallic)
        {
            mData.specular.b = metallic;
            markUpdates(UpdateFlags::DataChanged);
            updateSpecularType();
        }
    }

    void Material::setSpecularTransmission(float specularTransmission)
    {
        if (mData.specularTransmission != specularTransmission)
        {
            mData.specularTransmission = specularTransmission;
            markUpdates(UpdateFlags::DataChanged);
            updateSpecularTransmissionType();
        }
    }

    void Material::setVolumeAbsorption(const float3& volumeAbsorption)
    {
        if (mData.volumeAbsorption != volumeAbsorption)
        {
            mData.volumeAbsorption = volumeAbsorption;
            markUpdates(UpdateFlags::DataChanged);
        }
    }

    void Material::setEmissiveColor(const float3& color)
    {
        if (mData.emissive != color)
        {
            mData.emissive = color;
            markUpdates(UpdateFlags::DataChanged);
            updateEmissiveType();
        }
    }

    void Material::setEmissiveFactor(float factor)
    {
        if (mData.emissiveFactor != factor)
        {
            mData.emissiveFactor = factor;
            markUpdates(UpdateFlags::DataChanged);
            updateEmissiveType();
        }
    }

    void Material::setNormalMap(Texture::SharedPtr pNormalMap)
    {
        if (mResources.normalMap != pNormalMap)
        {
            mResources.normalMap = pNormalMap;
            markUpdates(UpdateFlags::ResourcesChanged);
            uint32_t normalMode = NormalMapUnused;
            if (pNormalMap)
            {
                switch(getFormatChannelCount(pNormalMap->getFormat()))
                {
                case 2:
                    normalMode = NormalMapRG;
                    break;
                case 3:
                case 4: // Some texture formats don't support RGB, only RGBA. We have no use for the alpha channel in the normal map.
                    normalMode = NormalMapRGB;
                    break;
                default:
                    should_not_get_here();
                    logWarning("Unsupported normal map format for material " + mName);
                }
            }
            setFlags(PACK_NORMAL_MAP_TYPE(mData.flags, normalMode));
        }
    }

    void Material::setOcclusionMap(Texture::SharedPtr pOcclusionMap)
    {
        if (mResources.occlusionMap != pOcclusionMap)
        {
            mResources.occlusionMap = pOcclusionMap;
            markUpdates(UpdateFlags::ResourcesChanged);
            updateOcclusionFlag();
        }
    }

    bool Material::operator==(const Material& other) const
    {
#define compare_field(_a) if (mData._a != other.mData._a) return false
        compare_field(baseColor);
        compare_field(specular);
        compare_field(emissive);
        compare_field(emissiveFactor);
        compare_field(alphaThreshold);
        compare_field(IoR);
        compare_field(specularTransmission);
        compare_field(flags);
        compare_field(volumeAbsorption);
#undef compare_field

#define compare_texture(_a) if (mResources._a != other.mResources._a) return false
        compare_texture(baseColor);
        compare_texture(specular);
        compare_texture(emissive);
        compare_texture(normalMap);
        compare_texture(occlusionMap);
        compare_texture(specularTransmission);
#undef compare_texture
        if (mResources.samplerState != other.mResources.samplerState) return false;
        return true;
    }

    void Material::markUpdates(UpdateFlags updates)
    {
        mUpdates |= updates;
        sGlobalUpdates |= updates;
    }

    void Material::setFlags(uint32_t flags)
    {
        if (mData.flags != flags)
        {
            mData.flags = flags;
            markUpdates(UpdateFlags::DataChanged);
        }
    }

    template<typename vec>
    static uint32_t getChannelMode(bool hasTexture, const vec& color)
    {
        if (hasTexture) return ChannelTypeTexture;
        if (luminance(color) == 0) return ChannelTypeUnused;
        return ChannelTypeConst;
    }

    void Material::updateBaseColorType()
    {
        setFlags(PACK_DIFFUSE_TYPE(mData.flags, getChannelMode(mResources.baseColor != nullptr, mData.baseColor)));
    }

    void Material::updateSpecularType()
    {
        setFlags(PACK_SPECULAR_TYPE(mData.flags, getChannelMode(mResources.specular != nullptr, mData.specular)));
    }

    void Material::updateEmissiveType()
    {
        setFlags(PACK_EMISSIVE_TYPE(mData.flags, getChannelMode(mResources.emissive != nullptr, mData.emissive * mData.emissiveFactor)));
    }

    void Material::updateSpecularTransmissionType()
    {
        setFlags(PACK_SPEC_TRANS_TYPE(mData.flags, getChannelMode(mResources.specularTransmission != nullptr, mData.specularTransmission)));
    }

    void Material::updateOcclusionFlag()
    {
        bool hasMap = false;
        switch (EXTRACT_SHADING_MODEL(mData.flags))
        {
        case ShadingModelMetalRough:
            hasMap = (mResources.specular != nullptr);
            break;
        case ShadingModelSpecGloss:
            hasMap = (mResources.occlusionMap != nullptr);
            break;
        default:
            should_not_get_here();
        }
        bool shouldEnable = mOcclusionMapEnabled && hasMap;
        setFlags(PACK_OCCLUSION_MAP(mData.flags, shouldEnable ? 1 : 0));
    }

    SCRIPT_BINDING(Material)
    {
        pybind11::enum_<Material::TextureSlot> textureSlot(m, "MaterialTextureSlot");
        textureSlot.value("BaseColor", Material::TextureSlot::BaseColor);
        textureSlot.value("Specular", Material::TextureSlot::Specular);
        textureSlot.value("Emissive", Material::TextureSlot::Emissive);
        textureSlot.value("Normal", Material::TextureSlot::Normal);
        textureSlot.value("Occlusion", Material::TextureSlot::Occlusion);
        textureSlot.value("SpecularTransmission", Material::TextureSlot::SpecularTransmission);

        pybind11::class_<Material, Material::SharedPtr> material(m, "Material");
        material.def_property_readonly("name", &Material::getName);
        material.def_property("baseColor", &Material::getBaseColor, &Material::setBaseColor);
        material.def_property("specularParams", &Material::getSpecularParams, &Material::setSpecularParams);
        material.def_property("roughness", &Material::getRoughness, &Material::setRoughness);
        material.def_property("metallic", &Material::getMetallic, &Material::setMetallic);
        material.def_property("specularTransmission", &Material::getSpecularTransmission, &Material::setSpecularTransmission);
        material.def_property("volumeAbsorption", &Material::getVolumeAbsorption, &Material::setVolumeAbsorption);
        material.def_property("indexOfRefraction", &Material::getIndexOfRefraction, &Material::setIndexOfRefraction);
        material.def_property("emissiveColor", &Material::getEmissiveColor, &Material::setEmissiveColor);
        material.def_property("emissiveFactor", &Material::getEmissiveFactor, &Material::setEmissiveFactor);
        material.def_property("alphaMode", &Material::getAlphaMode, &Material::setAlphaMode);
        material.def_property("alphaThreshold", &Material::getAlphaThreshold, &Material::setAlphaThreshold);
        material.def_property("doubleSided", &Material::isDoubleSided, &Material::setDoubleSided);
        material.def_property("nestedPriority", &Material::getNestedPriority, &Material::setNestedPriority);

        material.def("loadTexture", &Material::loadTexture, "slot"_a, "filename"_a, "useSrgb"_a = true);
        material.def("clearTexture", &Material::clearTexture, "slot"_a);
    }
}
