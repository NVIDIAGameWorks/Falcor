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
#include "Material.h"
#include "BasicMaterial.h"
#include "MaterialSystem.h"
#include "MaterialTypeRegistry.h"
#include "GlobalState.h"
#include "Core/API/Device.h"
#include "Utils/Logger.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "Rendering/Materials/LobeType.slang"

namespace Falcor
{
    namespace
    {
        static_assert(sizeof(TextureHandle) == 4);
        static_assert(sizeof(MaterialHeader) == 16);
        static_assert(sizeof(MaterialPayload) == 112);
        static_assert(sizeof(MaterialDataBlob) == 128);
        static_assert(static_cast<uint32_t>(MaterialType::BuiltinCount) <= (1u << MaterialHeader::kMaterialTypeBits), "MaterialType count exceeds the maximum");
        static_assert(static_cast<uint32_t>(AlphaMode::Count) <= (1u << MaterialHeader::kAlphaModeBits), "AlphaMode bit count exceeds the maximum");
        static_assert(static_cast<uint32_t>(LobeType::All) < (1u << MaterialHeader::kLobeTypeBits), "LobeType bit count exceeds the maximum");
        static_assert(static_cast<uint32_t>(TextureHandle::Mode::Count) <= (1u << TextureHandle::kModeBits), "TextureHandle::Mode bit count exceeds the maximum");
        static_assert(MaterialHeader::kTotalHeaderBitsX <= 32, "MaterialHeader bit count x exceeds the maximum");
        static_assert(MaterialHeader::kTotalHeaderBitsY <= 32, "MaterialHeader bit count y exceeds the maximum");
        static_assert(MaterialHeader::kTotalHeaderBitsZ <= 32, "MaterialHeader bit count z exceeds the maximum");
        static_assert(MaterialHeader::kTotalHeaderBitsW <= 32, "MaterialHeader bit count w exceeds the maximum");
        static_assert(MaterialHeader::kAlphaThresholdBits == 16, "MaterialHeader alpha threshold bit count must be 16");
    }

    bool operator==(const MaterialHeader& lhs, const MaterialHeader& rhs)
    {
        return all(lhs.packedData == rhs.packedData);
    }

    Material::Material(ref<Device> pDevice, const std::string& name, MaterialType type)
        : mpDevice(pDevice)
        , mName(name)
    {
        mHeader.setMaterialType(type);
        mHeader.setAlphaMode(AlphaMode::Opaque);
        mHeader.setAlphaThreshold(0.5h);
        mHeader.setActiveLobes(static_cast<uint32_t>(LobeType::All));
        mHeader.setIoR(1.h);
    }

    bool Material::renderUI(Gui::Widgets& widget)
    {
        // We're re-using the material's update flags here to track changes.
        // Cache the previous flag so we can restore it before returning.
        UpdateFlags prevUpdates = mUpdates;
        mUpdates = UpdateFlags::None;

        widget.text("Type: " + to_string(getType()));

        widget.text("Flags:");
        {
            bool doubleSided = isDoubleSided();
            if (widget.checkbox("Double-sided", doubleSided)) setDoubleSided(doubleSided);

            bool thinSurface = isThinSurface();
            if (widget.checkbox("Thin surface", thinSurface)) setThinSurface(thinSurface);
        }

        bool alphaTest = !isOpaque();
        widget.checkbox("Alpha test", alphaTest);
        if (alphaTest)
        {
            // Show alpha threshold. This is not directly editable.
            float alphaThreshold = getAlphaThreshold();
            widget.var("Alpha threshold", alphaThreshold);
        }

        uint32_t nestedPriority = getNestedPriority();
        if (widget.var("Nested priority", nestedPriority, 0u, (1u << MaterialHeader::kNestedPriorityBits) - 1)) setNestedPriority(nestedPriority);

        // Restore update flags.
        bool changed = mUpdates != UpdateFlags::None;
        markUpdates(prevUpdates | mUpdates);

        return changed;
    }

    void Material::setDoubleSided(bool doubleSided)
    {
        if (mHeader.isDoubleSided() != doubleSided)
        {
            mHeader.setDoubleSided(doubleSided);
            markUpdates(UpdateFlags::DataChanged);
        }
    }

    void Material::setThinSurface(bool thinSurface)
    {
        if (mHeader.isThinSurface() != thinSurface)
        {
            mHeader.setThinSurface(thinSurface);
            markUpdates(UpdateFlags::DataChanged);
        }
    }

    void Material::setAlphaMode(AlphaMode alphaMode)
    {
        logWarning("Material '{}' of type '{}' does not support alpha. Ignoring call to setAlphaMode().", getName(), to_string(getType()));
    }

    void Material::setAlphaThreshold(float alphaThreshold)
    {
        logWarning("Material '{}' of type '{}' does not support alpha. Ignoring call to setAlphaThreshold().", getName(), to_string(getType()));
    }

    void Material::setNestedPriority(uint32_t priority)
    {
        const uint32_t maxPriority = (1u << MaterialHeader::kNestedPriorityBits) - 1;
        if (priority > maxPriority)
        {
            logWarning("Requested nested priority {} for material '{}' is out of range. Clamping to {}.", priority, mName, maxPriority);
            priority = maxPriority;
        }
        if (mHeader.getNestedPriority() != priority)
        {
            mHeader.setNestedPriority(priority);
            markUpdates(UpdateFlags::DataChanged);
        }
    }

    void Material::setIndexOfRefraction(float IoR)
    {
        if (mHeader.getIoR() != (float16_t)IoR)
        {
            mHeader.setIoR((float16_t)IoR);
            markUpdates(UpdateFlags::DataChanged);
        }
    }

    const Material::TextureSlotInfo& Material::getTextureSlotInfo(const TextureSlot slot) const
    {
        FALCOR_ASSERT((size_t)slot < mTextureSlotInfo.size());
        return mTextureSlotInfo[(size_t)slot];
    }

    bool Material::hasTextureSlotData(const TextureSlot slot) const
    {
        FALCOR_ASSERT((size_t)slot < mTextureSlotInfo.size());
        return mTextureSlotData[(size_t)slot].pTexture != nullptr;
    }

    bool Material::setTexture(const TextureSlot slot, const ref<Texture>& pTexture)
    {
        if (!hasTextureSlot(slot))
        {
            logWarning("Material '{}' does not have texture slot '{}'. Ignoring call to setTexture().", getName(), to_string(slot));
            return false;
        }

        if (pTexture == getTexture(slot)) return false;

        FALCOR_ASSERT((size_t)slot < mTextureSlotInfo.size());
        mTextureSlotData[(size_t)slot].pTexture = pTexture;

        markUpdates(UpdateFlags::ResourcesChanged);
        if (slot == TextureSlot::Emissive)
            markUpdates(UpdateFlags::EmissiveChanged);

        return true;
    }

    ref<Texture> Material::getTexture(const TextureSlot slot) const
    {
        if (!hasTextureSlot(slot)) return nullptr;

        FALCOR_ASSERT((size_t)slot < mTextureSlotInfo.size());
        return mTextureSlotData[(size_t)slot].pTexture;
    }

    bool Material::loadTexture(TextureSlot slot, const std::filesystem::path& path, bool useSrgb)
    {
        if (!hasTextureSlot(slot))
        {
            logWarning("Material '{}' does not have texture slot '{}'. Ignoring call to loadTexture().", getName(), to_string(slot));
            return false;
        }

        auto texture = Texture::createFromFile(mpDevice, path, true, useSrgb && getTextureSlotInfo(slot).srgb);
        if (texture)
        {
            setTexture(slot, texture);
            // Wait for GPU to finish to prevent the upload heap from growing too large. Doing so after
            // every texture creation is overly conservative, and will likely lead to performance issues
            // due to the forced CPU/GPU sync.
            mpDevice->wait();
            return true;
        }

        return false;
    }

    uint2 Material::getMaxTextureDimensions() const
    {
        uint2 dim = uint2(0);
        for (uint32_t i = 0; i < (uint32_t)TextureSlot::Count; i++)
        {
            auto pTexture = getTexture((TextureSlot)i);
            if (pTexture) dim = max(dim, uint2(pTexture->getWidth(), pTexture->getHeight()));
        }
        return dim;
    }

    void Material::setTextureTransform(const Transform& textureTransform)
    {
        mTextureTransform = textureTransform;
    }

    ref<BasicMaterial> Material::toBasicMaterial()
    {
        if (mHeader.isBasicMaterial())
        {
            FALCOR_ASSERT(dynamic_ref_cast<BasicMaterial>(ref<Material>(this)));
            return static_ref_cast<BasicMaterial>(ref<Material>(this));
        }
        return nullptr;
    }

    void Material::markUpdates(UpdateFlags updates)
    {
        // Mark updates locally in this material.
        mUpdates |= updates;

        // Mark updates globally across all materials.
        if (mUpdateCallback) mUpdateCallback(updates);
    }

    void Material::updateTextureHandle(MaterialSystem* pOwner, const ref<Texture>& pTexture, TextureHandle& handle)
    {
        TextureHandle prevHandle = handle;

        // Update the given texture handle.
        if (pTexture)
        {
            auto h = pOwner->getTextureManager().addTexture(pTexture);
            FALCOR_ASSERT(h);
            handle = h.toGpuHandle();
        }
        else
        {
            handle.setMode(TextureHandle::Mode::Uniform);
            handle.setUdimEnabled(false);
        }
        FALCOR_ASSERT(!handle.getUdimEnabled());

        if (handle != prevHandle) mUpdates |= Material::UpdateFlags::DataChanged;
    }

    void Material::updateTextureHandle(MaterialSystem* pOwner, const TextureSlot slot, TextureHandle& handle)
    {
        auto pTexture = getTexture(slot);
        updateTextureHandle(pOwner, pTexture, handle);

        // The base color texture potentially contains the alpha mask in it's alpha channel.
        // Set it as the alpha texture handle in the material header.
        if (slot == TextureSlot::BaseColor)
        {
            mHeader.setAlphaTextureHandle(handle);
            mUpdates |= Material::UpdateFlags::DataChanged;
        }
    }

    void Material::updateDefaultTextureSamplerID(MaterialSystem* pOwner, const ref<Sampler>& pSampler)
    {
        const uint32_t samplerID = pOwner->addTextureSampler(pSampler);

        if (mHeader.getDefaultTextureSamplerID() != samplerID)
        {
            mHeader.setDefaultTextureSamplerID(samplerID);
            mUpdates |= Material::UpdateFlags::DataChanged;
        }
    }

    bool Material::isBaseEqual(const Material& other) const
    {
        // This function compares all data in the base class between two materials *except* the name.
        // It's a separate helper to ensure isEqual() is pure virtual and must be implemented in all derived classes.

        if (mHeader != other.mHeader) return false;
        if (mTextureTransform != other.mTextureTransform) return false;

        FALCOR_ASSERT(mTextureSlotInfo.size() == mTextureSlotData.size());
        for (size_t i = 0; i < mTextureSlotInfo.size(); i++)
        {
            // Compare texture slots.
            // These checks are a bit redundant since identical material types are currently
            // guaranteed to have the same set of slots, but this is future-proof if that changes.
            auto slot = (TextureSlot)i;
            if (hasTextureSlot(slot) != other.hasTextureSlot(slot)) return false;
            if (hasTextureSlot(slot))
            {
                if (mTextureSlotInfo[i] != other.mTextureSlotInfo[i]) return false;
                if (mTextureSlotData[i] != other.mTextureSlotData[i]) return false;
            }
        }

        return true;
    }

    NormalMapType Material::detectNormalMapType(const ref<Texture>& pNormalMap)
    {
        NormalMapType type = NormalMapType::None;
        if (pNormalMap != nullptr)
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
                logWarning("Unsupported normal map format: ", to_string(pNormalMap->getFormat()));
            }
        }
        return type;
    }

    FALCOR_SCRIPT_BINDING(Material)
    {
        using namespace pybind11::literals;

        FALCOR_SCRIPT_BINDING_DEPENDENCY(Transform)
        FALCOR_SCRIPT_BINDING_DEPENDENCY(Texture)

        pybind11::enum_<MaterialType> materialType(m, "MaterialType");
        materialType.value("Standard", MaterialType::Standard);
        materialType.value("Cloth", MaterialType::Cloth);
        materialType.value("Hair", MaterialType::Hair);
        materialType.value("MERL", MaterialType::MERL);
        materialType.value("MERLMix", MaterialType::MERLMix);
        materialType.value("PBRTDiffuse", MaterialType::PBRTDiffuse);
        materialType.value("PBRTDiffuseTransmission", MaterialType::PBRTDiffuseTransmission);
        materialType.value("PBRTConductor", MaterialType::PBRTConductor);
        materialType.value("PBRTDielectric", MaterialType::PBRTDielectric);
        materialType.value("PBRTCoatedConductor", MaterialType::PBRTCoatedConductor);
        materialType.value("PBRTCoatedDiffuse", MaterialType::PBRTCoatedDiffuse);
        materialType.value("RGL", MaterialType::RGL);

        pybind11::enum_<AlphaMode> alphaMode(m, "AlphaMode");
        alphaMode.value("Opaque", AlphaMode::Opaque);
        alphaMode.value("Mask", AlphaMode::Mask);

        pybind11::enum_<Material::TextureSlot> textureSlot(m, "MaterialTextureSlot");
        textureSlot.value("BaseColor", Material::TextureSlot::BaseColor);
        textureSlot.value("Specular", Material::TextureSlot::Specular);
        textureSlot.value("Emissive", Material::TextureSlot::Emissive);
        textureSlot.value("Normal", Material::TextureSlot::Normal);
        textureSlot.value("Transmission", Material::TextureSlot::Transmission);
        textureSlot.value("Displacement", Material::TextureSlot::Displacement);
        textureSlot.value("Index", Material::TextureSlot::Index);

        // Register Material base class as IMaterial in python to allow deprecated script syntax.
        // TODO: Remove workaround when all scripts have been updated to create derived Material classes.
        pybind11::class_<Material, ref<Material>> material(m, "IMaterial"); // PYTHONDEPRECATED
        material.def_property_readonly("type", &Material::getType);
        material.def_property("name", &Material::getName, &Material::setName);
        material.def_property("doubleSided", &Material::isDoubleSided, &Material::setDoubleSided);
        material.def_property("thinSurface", &Material::isThinSurface, &Material::setThinSurface);
        material.def_property_readonly("emissive", &Material::isEmissive);
        material.def_property("alphaMode", &Material::getAlphaMode, &Material::setAlphaMode);
        material.def_property("alphaThreshold", &Material::getAlphaThreshold, &Material::setAlphaThreshold);
        material.def_property("nestedPriority", &Material::getNestedPriority, &Material::setNestedPriority);
        material.def_property("textureTransform", pybind11::overload_cast<>(&Material::getTextureTransform, pybind11::const_), &Material::setTextureTransform);

        material.def("setTexture", &Material::setTexture, "slot"_a, "texture"_a);
        material.def("getTexture", &Material::getTexture, "slot"_a);
        auto loadTexture = [&](Material& self, Material::TextureSlot slot, const std::filesystem::path& path, bool useSrgb) {
            return self.loadTexture(slot, getActiveAssetResolver().resolvePath(path), useSrgb);
        };
        material.def("loadTexture", loadTexture, "slot"_a, "path"_a, "useSrgb"_a = true); // PYTHONDEPRECATED
        material.def("load_texture", loadTexture, "slot"_a, "path"_a, "use_srgb"_a = true); // PYTHONDEPRECATED
        material.def("clearTexture", &Material::clearTexture, "slot"_a);

        auto getMaterialParamLayoutDict = [&](MaterialType type) -> pybind11::dict {
            MaterialParamLayout layout = getMaterialParamLayout(type);
            pybind11::dict dict;
            for (const auto& entry : layout)
                dict[entry.pythonName] = pybind11::dict("offset"_a = entry.offset, "size"_a = entry.size);
            return dict;
        };

        auto getMaterialParamLayoutsDict = [getMaterialParamLayoutDict]() -> pybind11::dict {
            pybind11::dict dict;
            for (uint32_t i = 0; i < uint32_t(MaterialType::BuiltinCount); ++i)
            {
                auto type = static_cast<MaterialType>(i);
                auto name = to_string(type);
                dict[name.c_str()] = getMaterialParamLayoutDict(type);
            }
            return dict;
        };

        material.attr("PARAM_COUNT") = SerializedMaterialParams::kParamCount;

        m.def("get_material_param_layout", getMaterialParamLayoutDict, "type"_a);
        m.attr("MATERIAL_PARAM_LAYOUTS") = getMaterialParamLayoutsDict();
    }
}
