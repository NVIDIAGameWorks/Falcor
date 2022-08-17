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
#include "BasicMaterial.h"
#include "MaterialSystem.h"
#include "Core/API/Device.h"
#include "Core/API/RenderContext.h"
#include "Core/Program/GraphicsProgram.h"
#include "Core/Program/ProgramVars.h"
#include "Utils/Logger.h"
#include "Utils/Math/Common.h"
#include "Utils/Color/ColorHelpers.slang"
#include "Utils/Scripting/ScriptBindings.h"
#include <sstream>

namespace Falcor
{
    namespace
    {
        static_assert((sizeof(MaterialHeader) + sizeof(BasicMaterialData)) <= sizeof(MaterialDataBlob), "Total material data size is too large");
        static_assert(static_cast<uint32_t>(ShadingModel::Count) <= (1u << BasicMaterialData::kShadingModelBits), "ShadingModel bit count exceeds the maximum");
        static_assert(static_cast<uint32_t>(NormalMapType::Count) <= (1u << BasicMaterialData::kNormalMapTypeBits), "NormalMapType bit count exceeds the maximum");
        static_assert(BasicMaterialData::kTotalFlagsBits <= 32, "BasicMaterialData flags bit count exceeds the maximum");

        // Constants.
        const float kMaxVolumeAnisotropy = 0.99f;
    }

    BasicMaterial::BasicMaterial(const std::string& name, MaterialType type)
        : Material(name, type)
    {
        mHeader.setIsBasicMaterial(true);

        // Setup common texture slots.
        mTextureSlotInfo[(uint32_t)TextureSlot::Displacement] = { "displacement", TextureChannelFlags::RGB, false };

        // Call update functions to ensure a valid initial state based on default material parameters.
        updateAlphaMode();
        updateNormalMapType();
        updateEmissiveFlag();
        updateDeltaSpecularFlag();
    }

    bool BasicMaterial::renderUI(Gui::Widgets& widget)
    {
        // Render the base class UI first.
        bool changed = Material::renderUI(widget);

        // We're re-using the material's update flags here to track changes.
        // Cache the previous flag so we can restore it before returning.
        UpdateFlags prevUpdates = mUpdates;
        mUpdates = UpdateFlags::None;

        if (auto pTexture = getBaseColorTexture())
        {
            bool hasAlpha = isAlphaSupported() && doesFormatHaveAlpha(pTexture->getFormat());
            bool alphaConst = mIsTexturedAlphaConstant && hasAlpha;
            bool colorConst = mIsTexturedBaseColorConstant;

            std::ostringstream oss;
            oss << "Texture info: " << pTexture->getWidth() << "x" << pTexture->getHeight()
                << " (" << to_string(pTexture->getFormat()) << ")";
            if (colorConst && !alphaConst) oss << " (color constant)";
            else if (!colorConst && alphaConst) oss << " (alpha constant)";
            else if (colorConst && alphaConst) oss << " (color and alpha constant)"; // Shouldn't happen

            widget.text("Base color: " + pTexture->getSourcePath().string());
            widget.text(oss.str());

            if (colorConst || alphaConst)
            {
                float4 baseColor = getBaseColor();
                if (widget.var("Base color", baseColor, 0.f, 1.f, 0.01f)) setBaseColor(baseColor);
            }

            widget.image("Base color", pTexture, float2(100.f));
            if (widget.button("Remove texture##BaseColor")) setBaseColorTexture(nullptr);
        }
        else
        {
            float4 baseColor = getBaseColor();
            if (widget.var("Base color", baseColor, 0.f, 1.f, 0.01f)) setBaseColor(baseColor);
        }

        if (auto pTexture = getSpecularTexture())
        {
            widget.text("Specular params: " + pTexture->getSourcePath().string());
            widget.text("Texture info: " + std::to_string(pTexture->getWidth()) + "x" + std::to_string(pTexture->getHeight()) + " (" + to_string(pTexture->getFormat()) + ")");
            widget.image("Specular params", pTexture, float2(100.f));
            if (widget.button("Remove texture##Specular")) setSpecularTexture(nullptr);
        }
        else
        {
            float4 specularParams = getSpecularParams();
            if (widget.var("Specular params", specularParams, 0.f, 1.f, 0.01f)) setSpecularParams(specularParams);
            widget.tooltip("The encoding depends on the material type");

            renderSpecularUI(widget); // Let derived classes draw additional UI elements.
        }

        if (auto pTexture = getNormalMap())
        {
            widget.text("Normal map: " + pTexture->getSourcePath().string());
            widget.text("Texture info: " + std::to_string(pTexture->getWidth()) + "x" + std::to_string(pTexture->getHeight()) + " (" + to_string(pTexture->getFormat()) + ")");
            widget.image("Normal map", pTexture, float2(100.f));
            if (widget.button("Remove texture##NormalMap")) setNormalMap(nullptr);
        }

        if (auto pTexture = getDisplacementMap())
        {
            widget.text("Displacement map: " + pTexture->getSourcePath().string());
            widget.text("Texture info: " + std::to_string(pTexture->getWidth()) + "x" + std::to_string(pTexture->getHeight()) + " (" + to_string(pTexture->getFormat()) + ")");
            widget.image("Displacement map", pTexture, float2(100.f));
            if (widget.button("Remove texture##DisplacementMap")) setDisplacementMap(nullptr);

            float scale = getDisplacementScale();
            if (widget.var("Displacement scale", scale)) setDisplacementScale(scale);

            float offset = getDisplacementOffset();
            if (widget.var("Displacement offset", offset)) setDisplacementOffset(offset);
        }

        if (auto pTexture = getTransmissionTexture())
        {
            widget.text("Transmission color: " + pTexture->getSourcePath().string());
            widget.text("Texture info: " + std::to_string(pTexture->getWidth()) + "x" + std::to_string(pTexture->getHeight()) + " (" + to_string(pTexture->getFormat()) + ")");
            widget.image("Transmission color", pTexture, float2(100.f));
            if (widget.button("Remove texture##Transmission")) setTransmissionTexture(nullptr);
        }
        else
        {
            float3 transmissionColor = getTransmissionColor();
            if (widget.var("Transmission", transmissionColor, 0.f, 1.f, 0.01f)) setTransmissionColor(transmissionColor);
        }

        float diffuseTransmission = getDiffuseTransmission();
        if (widget.var("Diffuse transmission", diffuseTransmission, 0.f, 1.f, 0.01f)) setDiffuseTransmission(diffuseTransmission);

        float specularTransmission = getSpecularTransmission();
        if (widget.var("Specular transmission", specularTransmission, 0.f, 1.f, 0.01f)) setSpecularTransmission(specularTransmission);

        float IoR = getIndexOfRefraction();
        if (widget.var("Index of refraction", IoR, 1.f, std::numeric_limits<float>::max(), 0.01f)) setIndexOfRefraction(IoR);

        float3 volumeAbsorption = getVolumeAbsorption();
        if (widget.var("Absorption coefficient", volumeAbsorption, 0.f, std::numeric_limits<float>::max(), 0.01f)) setVolumeAbsorption(volumeAbsorption);

        float3 volumeScattering = getVolumeScattering();
        if (widget.var("Scattering coefficient", volumeScattering, 0.f, std::numeric_limits<float>::max(), 0.01f)) setVolumeScattering(volumeScattering);

        float volumeAnisotropy = getVolumeAnisotropy();
        if (widget.var("Anisotropy (g)", volumeAnisotropy, -1.f, 1.f, 0.01f)) setVolumeAnisotropy(volumeAnisotropy);

        if (isAlphaSupported())
        {
            // Show alpha range. This is not directly editable.
            float2 alphaRange = mAlphaRange;
            widget.var("Alpha range", alphaRange);
        }

        // Restore update flags.
        changed |= mUpdates != UpdateFlags::None;
        markUpdates(prevUpdates | mUpdates);

        return changed;
    }

    Material::UpdateFlags BasicMaterial::update(MaterialSystem* pOwner)
    {
        FALCOR_ASSERT(pOwner);

        auto flags = Material::UpdateFlags::None;
        if (mUpdates != Material::UpdateFlags::None)
        {
            // Adjust material sidedness based on current parameters.
            // TODO: Remove when single-sided transmissive materials are supported.
            adjustDoubleSidedFlag();

            // Prepare displacement maps for rendering.
            prepareDisplacementMapForRendering();

            // Update texture handles.
            updateTextureHandle(pOwner, TextureSlot::BaseColor, mData.texBaseColor);
            updateTextureHandle(pOwner, TextureSlot::Specular, mData.texSpecular);
            updateTextureHandle(pOwner, TextureSlot::Emissive, mData.texEmissive);
            updateTextureHandle(pOwner, TextureSlot::Transmission, mData.texTransmission);
            updateTextureHandle(pOwner, TextureSlot::Normal, mData.texNormalMap);
            updateTextureHandle(pOwner, TextureSlot::Displacement, mData.texDisplacementMap);

            // Update default sampler.
            updateDefaultTextureSamplerID(pOwner, mpDefaultSampler);

            // Update displacement samplers.
            uint prevFlags = mData.flags;
            mData.setDisplacementMinSamplerID(pOwner->addTextureSampler(mpDisplacementMinSampler));
            mData.setDisplacementMaxSamplerID(pOwner->addTextureSampler(mpDisplacementMaxSampler));
            if (mData.flags != prevFlags) mUpdates |= Material::UpdateFlags::DataChanged;

            flags |= mUpdates;
            mUpdates = Material::UpdateFlags::None;
        }

        return flags;
    }

    bool BasicMaterial::isDisplaced() const
    {
        return hasTextureSlotData(Material::TextureSlot::Displacement);
    }

    void BasicMaterial::setAlphaMode(AlphaMode alphaMode)
    {
        if (!isAlphaSupported())
        {
            FALCOR_ASSERT(getAlphaMode() == AlphaMode::Opaque);
            logWarning("Alpha is not supported by material type '{}'. Ignoring call to setAlphaMode() for material '{}'.", to_string(getType()), getName());
            return;
        }
        if (mHeader.getAlphaMode() != alphaMode)
        {
            mHeader.setAlphaMode(alphaMode);
            markUpdates(UpdateFlags::DataChanged);
        }
    }

    void BasicMaterial::setAlphaThreshold(float alphaThreshold)
    {
        if (!isAlphaSupported())
        {
            logWarning("Alpha is not supported by material type '{}'. Ignoring call to setAlphaThreshold() for material '{}'.", to_string(getType()), getName());
            return;
        }
        if (mHeader.getAlphaThreshold() != (float16_t)alphaThreshold)
        {
            mHeader.setAlphaThreshold((float16_t)alphaThreshold);
            markUpdates(UpdateFlags::DataChanged);
            updateAlphaMode();
        }
    }

    void BasicMaterial::setIndexOfRefraction(float IoR)
    {
        if (mData.IoR != (float16_t)IoR)
        {
            mData.IoR = (float16_t)IoR;
            markUpdates(UpdateFlags::DataChanged);
        }
    }

    void BasicMaterial::setDefaultTextureSampler(const Sampler::SharedPtr& pSampler)
    {
        if (pSampler != mpDefaultSampler)
        {
            mpDefaultSampler = pSampler;

            // Create derived samplers for displacement Min/Max filtering.
            Sampler::Desc desc = pSampler->getDesc();
            desc.setMaxAnisotropy(16); // Set 16x anisotropic filtering for improved min/max precision per triangle.
            desc.setReductionMode(Sampler::ReductionMode::Min);
            mpDisplacementMinSampler = Sampler::create(desc);
            desc.setReductionMode(Sampler::ReductionMode::Max);
            mpDisplacementMaxSampler = Sampler::create(desc);

            markUpdates(UpdateFlags::ResourcesChanged);
        }
    }

    bool BasicMaterial::setTexture(const TextureSlot slot, const Texture::SharedPtr& pTexture)
    {
        if (!Material::setTexture(slot, pTexture)) return false;

        // Update additional metadata about texture usage.
        switch (slot)
        {
        case TextureSlot::BaseColor:
            if (pTexture)
            {
                // Assume the texture is non-constant and has full alpha range.
                // This may be changed later by optimizeTexture().
                mAlphaRange = float2(0.f, 1.f);
                mIsTexturedBaseColorConstant = mIsTexturedAlphaConstant = false;
            }
            updateAlphaMode();
            updateDeltaSpecularFlag();
            break;
        case TextureSlot::Specular:
            updateDeltaSpecularFlag();
            break;
        case TextureSlot::Normal:
            updateNormalMapType();
            break;
        case TextureSlot::Emissive:
            updateEmissiveFlag();
            break;
        case TextureSlot::Displacement:
            mDisplacementMapChanged = true;
            markUpdates(UpdateFlags::DisplacementChanged);
            break;
        default:
            break;
        }

        return true;
    }

    void BasicMaterial::optimizeTexture(const TextureSlot slot, const TextureAnalyzer::Result& texInfo, TextureOptimizationStats& stats)
    {
        FALCOR_ASSERT(getTexture(slot) != nullptr);
        TextureChannelFlags channelMask = getTextureSlotInfo(slot).mask;

        switch (slot)
        {
        case TextureSlot::BaseColor:
        {
            bool previouslyOpaque = isOpaque();

            auto pBaseColor = getBaseColorTexture();
            bool hasAlpha = isAlphaSupported() && pBaseColor && doesFormatHaveAlpha(pBaseColor->getFormat());
            bool isColorConstant = texInfo.isConstant(TextureChannelFlags::RGB);
            bool isAlphaConstant = texInfo.isConstant(TextureChannelFlags::Alpha);

            // Update the alpha range.
            if (hasAlpha) mAlphaRange = float2(texInfo.minValue.a, texInfo.maxValue.a);

            // Update base color parameter and texture.
            float4 baseColor = getBaseColor();
            if (isColorConstant)
            {
                baseColor = float4(texInfo.value.rgb, baseColor.a);
                mIsTexturedBaseColorConstant = true;
            }
            if (hasAlpha && isAlphaConstant)
            {
                baseColor = float4(baseColor.rgb, texInfo.value.a);
                mIsTexturedAlphaConstant = true;
            }
            setBaseColor(baseColor);

            if (isColorConstant && (!hasAlpha || isAlphaConstant))
            {
                clearTexture(Material::TextureSlot::BaseColor);
                stats.texturesRemoved[(size_t)slot]++;
            }
            else if (isColorConstant)
            {
                // We don't have a way to specify constant base color with non-constant alpha since they share a texture slot.
                // Count number of cases and issue a perf warning later instead.
                stats.constantBaseColor++;
            }

            updateAlphaMode();

            if (!previouslyOpaque && isOpaque()) stats.disabledAlpha++;

            break;
        }
        case TextureSlot::Specular:
        {
            if (texInfo.isConstant(channelMask))
            {
                clearTexture(Material::TextureSlot::Specular);
                setSpecularParams(texInfo.value);
                stats.texturesRemoved[(size_t)slot]++;
            }
            break;
        }
        case TextureSlot::Emissive:
        {
            if (texInfo.isConstant(channelMask))
            {
                clearTexture(Material::TextureSlot::Emissive);
                setEmissiveColor(texInfo.value.rgb);
                stats.texturesRemoved[(size_t)slot]++;
            }
            break;
        }
        case TextureSlot::Normal:
        {
            // Determine which channels of the normal map are used.
            switch (getNormalMapType())
            {
            case NormalMapType::RG:
                channelMask = TextureChannelFlags::Red | TextureChannelFlags::Green;
                break;
            case NormalMapType::RGB:
                channelMask = TextureChannelFlags::RGB;
                break;
            default:
                logWarning("BasicMaterial::optimizeTexture() - Unsupported normal map mode");
                channelMask = TextureChannelFlags::RGBA;
                break;
            }

            if (texInfo.isConstant(channelMask))
            {
                // We don't have a way to specify constant normal map value.
                // Count number of cases and issue a perf warning later instead.
                stats.constantNormalMaps++;
            }
            break;
        }
        case TextureSlot::Transmission:
        {
            if (texInfo.isConstant(channelMask))
            {
                clearTexture(Material::TextureSlot::Transmission);
                setTransmissionColor(texInfo.value.rgb);
                stats.texturesRemoved[(size_t)slot]++;
            }
            break;
        }
        case TextureSlot::Displacement:
        {
            // Nothing to do here, displacement texture is prepared when calling prepareDisplacementMap().
            break;
        }
        default:
            throw ArgumentError("'slot' refers to unexpected texture slot {}", (uint32_t)slot);
        }
    }

    bool BasicMaterial::isAlphaSupported() const
    {
        return getTextureSlotInfo(TextureSlot::BaseColor).hasChannel(TextureChannelFlags::Alpha);
    }

    void BasicMaterial::prepareDisplacementMapForRendering()
    {
        if (auto pDisplacementMap = getDisplacementMap(); pDisplacementMap && mDisplacementMapChanged)
        {
            // Creates RGBA texture with MIP pyramid containing average, min, max values.
            Falcor::ResourceFormat oldFormat = pDisplacementMap->getFormat();

            // Replace texture with a 4 component one if necessary.
            if (getFormatChannelCount(oldFormat) < 4)
            {
                Falcor::ResourceFormat newFormat = ResourceFormat::RGBA16Float;
                Resource::BindFlags bf = pDisplacementMap->getBindFlags() | Resource::BindFlags::UnorderedAccess | Resource::BindFlags::RenderTarget;
                Texture::SharedPtr newDisplacementTex = Texture::create2D(pDisplacementMap->getWidth(), pDisplacementMap->getHeight(), newFormat, pDisplacementMap->getArraySize(), Resource::kMaxPossible, nullptr, bf);

                // Copy base level.
                RenderContext* pContext = gpDevice->getRenderContext();
                uint32_t arraySize = pDisplacementMap->getArraySize();
                for (uint32_t a = 0; a < arraySize; a++)
                {
                    auto srv = pDisplacementMap->getSRV(0, 1, a, 1);
                    auto rtv = newDisplacementTex->getRTV(0, a, 1);
                    const Sampler::ReductionMode redModes[] = { Sampler::ReductionMode::Standard, Sampler::ReductionMode::Standard, Sampler::ReductionMode::Standard, Sampler::ReductionMode::Standard };
                    const float4 componentsTransform[] = { float4(1.0f, 0.0f, 0.0f, 0.0f), float4(1.0f, 0.0f, 0.0f, 0.0f), float4(1.0f, 0.0f, 0.0f, 0.0f), float4(1.0f, 0.0f, 0.0f, 0.0f) };
                    pContext->blit(srv, rtv, RenderContext::kMaxRect, RenderContext::kMaxRect, Sampler::Filter::Linear, redModes, componentsTransform);
                }

                pDisplacementMap = newDisplacementTex;
                setDisplacementMap(newDisplacementTex);
            }

            // Build min/max MIPS.
            pDisplacementMap->generateMips(gpDevice->getRenderContext(), true);
        }
        mDisplacementMapChanged = false;
    }

    void BasicMaterial::setDisplacementScale(float scale)
    {
        if (mData.displacementScale != scale)
        {
            mData.displacementScale = scale;
            markUpdates(UpdateFlags::DataChanged | UpdateFlags::DisplacementChanged);
        }
    }

    void BasicMaterial::setDisplacementOffset(float offset)
    {
        if (mData.displacementOffset != offset)
        {
            mData.displacementOffset = offset;
            markUpdates(UpdateFlags::DataChanged | UpdateFlags::DisplacementChanged);
        }
    }

    void BasicMaterial::setBaseColor(const float4& color)
    {
        if (mData.baseColor != (float16_t4)color)
        {
            mData.baseColor = (float16_t4)color;
            markUpdates(UpdateFlags::DataChanged);
            updateAlphaMode();
            updateDeltaSpecularFlag();
        }
    }

    void BasicMaterial::setSpecularParams(const float4& color)
    {
        if (mData.specular != (float16_t4)color)
        {
            mData.specular = (float16_t4)color;
            markUpdates(UpdateFlags::DataChanged);
            updateDeltaSpecularFlag();
        }
    }

    void BasicMaterial::setTransmissionColor(const float3& transmissionColor)
    {
        if (mData.transmission != (float16_t3)transmissionColor)
        {
            mData.transmission = (float16_t3)transmissionColor;
            markUpdates(UpdateFlags::DataChanged);
        }
    }

    void BasicMaterial::setDiffuseTransmission(float diffuseTransmission)
    {
        if (mData.diffuseTransmission != (float16_t)diffuseTransmission)
        {
            mData.diffuseTransmission = (float16_t)diffuseTransmission;
            markUpdates(UpdateFlags::DataChanged);
            updateDeltaSpecularFlag();
        }
    }

    void BasicMaterial::setSpecularTransmission(float specularTransmission)
    {
        if (mData.specularTransmission != (float16_t)specularTransmission)
        {
            mData.specularTransmission = (float16_t)specularTransmission;
            markUpdates(UpdateFlags::DataChanged);
            updateDeltaSpecularFlag();
        }
    }

    void BasicMaterial::setVolumeAbsorption(const float3& volumeAbsorption)
    {
        if (mData.volumeAbsorption != (float16_t3)volumeAbsorption)
        {
            mData.volumeAbsorption = (float16_t3)volumeAbsorption;
            markUpdates(UpdateFlags::DataChanged);
        }
    }

    void BasicMaterial::setVolumeScattering(const float3& volumeScattering)
    {
        if (mData.volumeScattering != (float16_t3)volumeScattering)
        {
            mData.volumeScattering = (float16_t3)volumeScattering;
            markUpdates(UpdateFlags::DataChanged);
        }
    }

    void BasicMaterial::setVolumeAnisotropy(float volumeAnisotropy)
    {
        auto clampedAnisotropy = clamp(volumeAnisotropy, -kMaxVolumeAnisotropy, kMaxVolumeAnisotropy);
        if (mData.volumeAnisotropy != (float16_t)clampedAnisotropy)
        {
            mData.volumeAnisotropy = (float16_t)clampedAnisotropy;
            markUpdates(UpdateFlags::DataChanged);
        }
    }

    bool BasicMaterial::isEqual(const Material::SharedPtr& pOther) const
    {
        auto other = std::dynamic_pointer_cast<BasicMaterial>(pOther);
        if (!other) return false;

        return (*this) == (*other);
    }

    bool BasicMaterial::operator==(const BasicMaterial& other) const
    {
        if (!isBaseEqual(other)) return false;

#define compare_field(_a) if (mData._a != other.mData._a) return false
        compare_field(flags);
        compare_field(displacementScale);
        compare_field(displacementOffset);
        compare_field(baseColor);
        compare_field(specular);
        compare_field(emissive);
        compare_field(emissiveFactor);
        compare_field(IoR);
        compare_field(diffuseTransmission);
        compare_field(specularTransmission);
        compare_field(transmission);
        compare_field(volumeAbsorption);
        compare_field(volumeAnisotropy);
        compare_field(volumeScattering);
#undef compare_field

        // Compare the sampler descs directly to identify functional differences.
        if (mpDefaultSampler->getDesc() != other.mpDefaultSampler->getDesc()) return false;
        if (mpDisplacementMinSampler->getDesc() != other.mpDisplacementMinSampler->getDesc()) return false;
        if (mpDisplacementMaxSampler->getDesc() != other.mpDisplacementMaxSampler->getDesc()) return false;

        return true;
    }

    void BasicMaterial::updateAlphaMode()
    {
        if (!isAlphaSupported())
        {
            FALCOR_ASSERT(getAlphaMode() == AlphaMode::Opaque);
            return;
        }

        // Set alpha range to the constant alpha value if non-textured.
        bool hasAlpha = getBaseColorTexture() && doesFormatHaveAlpha(getBaseColorTexture()->getFormat());
        float alpha = ((float4)mData.baseColor).a;
        if (!hasAlpha) mAlphaRange = float2(alpha);

        // Decide if we need to run the alpha test.
        // This is derived from the current alpha threshold and conservative alpha range.
        // If the test will never fail we disable it. This optimization assumes basic alpha thresholding.
        // We could also optimize for the case of the test always failing by adding a 'Transparent' mode.
        // This is however expected to be rare and probably not worth the runtime cost of an extra branch.
        // TODO: Check if optimizing for always-fail is worth it.
        // TODO: Update the logic if other alpha modes are added.
        bool useAlpha = mAlphaRange.x < getAlphaThreshold();
        setAlphaMode(useAlpha ? AlphaMode::Mask : AlphaMode::Opaque);
    }

    void BasicMaterial::updateNormalMapType()
    {
        NormalMapType type = NormalMapType::None;

        if (auto pNormalMap = getNormalMap())
        {
            switch (getFormatChannelCount(pNormalMap->getFormat()))
            {
            case 2:
                type = NormalMapType::RG;
                break;
            case 3:
            case 4: // Some texture formats don't support RGB, only RGBA. We have no use for the alpha channel in the normal map.
                type = NormalMapType::RGB;
                break;
            default:
                FALCOR_UNREACHABLE();
                logWarning("Unsupported normal map format for material '{}'.", mName);
            }
        }

        if (mData.getNormalMapType() != type)
        {
            mData.setNormalMapType(type);
            markUpdates(UpdateFlags::DataChanged);
        }
    }

    void BasicMaterial::updateEmissiveFlag()
    {
        bool isEmissive = false;
        if (mData.emissiveFactor > 0.f)
        {
            isEmissive = hasTextureSlotData(Material::TextureSlot::Emissive) || mData.emissive != float3(0.f);
        }
        if (mHeader.isEmissive() != isEmissive)
        {
            mHeader.setEmissive(isEmissive);
            markUpdates(UpdateFlags::DataChanged);
        }
    }

    void BasicMaterial::adjustDoubleSidedFlag()
    {
        bool doubleSided = isDoubleSided();

        // Make double sided if diffuse or specular transmission is used.
        // Note this convention will eventually change to allow single-sided transmissive materials.
        if ((float)mData.diffuseTransmission > 0.f || (float)mData.specularTransmission > 0.f) doubleSided = true;

        // Make double sided if displaced since backfacing surfaces can become frontfacing.
        if (isDisplaced()) doubleSided = true;

        setDoubleSided(doubleSided);
    }

    FALCOR_SCRIPT_BINDING(BasicMaterial)
    {
        FALCOR_SCRIPT_BINDING_DEPENDENCY(Material)

        pybind11::class_<BasicMaterial, Material, BasicMaterial::SharedPtr> material(m, "BasicMaterial");
        material.def_property("baseColor", &BasicMaterial::getBaseColor, &BasicMaterial::setBaseColor);
        material.def_property("specularParams", &BasicMaterial::getSpecularParams, &BasicMaterial::setSpecularParams);
        material.def_property("transmissionColor", &BasicMaterial::getTransmissionColor, &BasicMaterial::setTransmissionColor);
        material.def_property("diffuseTransmission", &BasicMaterial::getDiffuseTransmission, &BasicMaterial::setDiffuseTransmission);
        material.def_property("specularTransmission", &BasicMaterial::getSpecularTransmission, &BasicMaterial::setSpecularTransmission);
        material.def_property("volumeAbsorption", &BasicMaterial::getVolumeAbsorption, &BasicMaterial::setVolumeAbsorption);
        material.def_property("volumeScattering", &BasicMaterial::getVolumeScattering, &BasicMaterial::setVolumeScattering);
        material.def_property("volumeAnisotropy", &BasicMaterial::getVolumeAnisotropy, &BasicMaterial::setVolumeAnisotropy);
        material.def_property("indexOfRefraction", &BasicMaterial::getIndexOfRefraction, &BasicMaterial::setIndexOfRefraction);
        material.def_property("displacementScale", &BasicMaterial::getDisplacementScale, &BasicMaterial::setDisplacementScale);
        material.def_property("displacementOffset", &BasicMaterial::getDisplacementOffset, &BasicMaterial::setDisplacementOffset);
    }
}
