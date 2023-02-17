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
#include "MERLMixMaterial.h"
#include "Core/API/Device.h"
#include "Utils/Logger.h"
#include "Utils/BufferAllocator.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "Scene/SceneBuilderAccess.h"
#include "Scene/Material/MERLFile.h"
#include "Scene/Material/MaterialSystem.h"
#include "Scene/Material/DiffuseSpecularUtils.h"
#include <fstream>

namespace Falcor
{
    namespace
    {
        static_assert((sizeof(MaterialHeader) + sizeof(MERLMixMaterialData)) <= sizeof(MaterialDataBlob), "MERLMixMaterialData is too large");
        static_assert(static_cast<uint32_t>(NormalMapType::Count) <= (1u << MERLMixMaterialData::kNormalMapTypeBits), "NormalMapType bit count exceeds the maximum");

        const char kShaderFile[] = "Rendering/Materials/MERLMixMaterial.slang";
    }

    MERLMixMaterial::SharedPtr MERLMixMaterial::create(std::shared_ptr<Device> pDevice, const std::string& name, const std::vector<std::filesystem::path>& paths)
    {
        return SharedPtr(new MERLMixMaterial(pDevice, name, paths));
    }

    MERLMixMaterial::MERLMixMaterial(std::shared_ptr<Device> pDevice, const std::string& name, const std::vector<std::filesystem::path>& paths)
        : Material(pDevice, name, MaterialType::MERLMix)
    {
        checkArgument(!paths.empty(), "MERLMixMaterial: Expected at least one path.");

        // Setup texture slots.
        mTextureSlotInfo[(uint32_t)TextureSlot::Normal] = { "normal", TextureChannelFlags::RGB, false };
        mTextureSlotInfo[(uint32_t)TextureSlot::Index] = { "index", TextureChannelFlags::Red, false };

        // Load all BRDFs.
        mBRDFs.resize(paths.size());
        std::vector<DiffuseSpecularData> extraData(paths.size());
        std::vector<float4> albedoLut;
        BufferAllocator buffer(128, 0 /* raw buffer */, 128, ResourceBindFlags::ShaderResource);
        MERLFile merlFile;

        for (size_t i = 0; i < paths.size(); i++)
        {
            if (!merlFile.loadBRDF(paths[i]))
                throw RuntimeError("MERLMixMaterial: Failed to load BRDF from '{}'.", paths[i]);

            auto& desc = mBRDFs[i];
            desc.path = merlFile.getDesc().path;
            desc.name = merlFile.getDesc().name;
            extraData[i] = merlFile.getDesc().extraData;

            // Copy BRDF samples into shared data buffer.
            const auto& brdf = merlFile.getData();
            checkInvariant(!brdf.empty() && sizeof(brdf[0]) == sizeof(float3), "Expected BRDF data in float3 format.");
            desc.byteSize = brdf.size() * sizeof(brdf[0]);
            desc.byteOffset = buffer.allocate(desc.byteSize);
            buffer.setBlob(brdf.data(), desc.byteOffset, desc.byteSize);

            // Copy albedo LUT into shared table.
            const auto& lut = merlFile.prepareAlbedoLUT(mpDevice);
            checkInvariant(lut.size() == MERLMixMaterialData::kAlbedoLUTSize, "MERLMixMaterial: Unexpected albedo LUT size.");
            albedoLut.insert(albedoLut.end(), lut.begin(), lut.end());
        }

        mData.brdfCount = static_cast<uint32_t>(mBRDFs.size());
        mData.byteStride = mBRDFs.size() > 1 ? mBRDFs[1].byteOffset : 0;
        for (size_t i = 0; i < mBRDFs.size(); i++)
        {
            checkInvariant(mBRDFs[i].byteOffset == i * mData.byteStride, "MERLMixMaterial: Unexpected stride.");
        }

        // Upload extra data for sampling.
        {
            mData.extraDataStride = (uint32_t)sizeof(DiffuseSpecularData);
            size_t byteSize = extraData.size() * mData.extraDataStride;
            mData.extraDataOffset = buffer.allocate(byteSize);
            buffer.setBlob(extraData.data(), mData.extraDataOffset, byteSize);
        }

        // Create GPU data buffer.
        mpBRDFData = buffer.getGPUBuffer(mpDevice.get());

        // Create albedo LUT as 2D texture parameterization over (cosTehta, brdfIndex).
        mpAlbedoLUT = Texture::create2D(mpDevice.get(), MERLMixMaterialData::kAlbedoLUTSize, mData.brdfCount, MERLFile::kAlbedoLUTFormat, 1, 1, albedoLut.data(), ResourceBindFlags::ShaderResource);

        // Create sampler for albedo LUT.
        {
            Sampler::Desc desc;
            desc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Point, Sampler::Filter::Point);
            desc.setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
            desc.setMaxAnisotropy(1);
            mpLUTSampler = Sampler::create(mpDevice.get(), desc);
        }

        // Create sampler for index map. Using point sampling as indices are not interpolatable.
        {
            Sampler::Desc desc;
            desc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point);
            desc.setAddressingMode(Sampler::AddressMode::Wrap, Sampler::AddressMode::Wrap, Sampler::AddressMode::Wrap);
            desc.setMaxAnisotropy(1);
            mpIndexSampler = Sampler::create(mpDevice.get(), desc);
        }

        updateNormalMapType();
        updateIndexMapType();

        markUpdates(Material::UpdateFlags::ResourcesChanged);
    }

    bool MERLMixMaterial::renderUI(Gui::Widgets& widget)
    {
        bool changed = Material::renderUI(widget);

        // We're re-using the material's update flags here to track changes.
        // Cache the previous flag so we can restore it before returning.
        UpdateFlags prevUpdates = mUpdates;
        mUpdates = UpdateFlags::None;

        // Display BRDF info.
        widget.text(fmt::format("Loaded MERL BRDFs: {}", mBRDFs.size()));
        if (auto g = widget.group("BRDFs"))
        {
            for (size_t i = 0; i < mBRDFs.size(); i++)
            {
                g.text(fmt::format("ID {}: {}", i, mBRDFs[i].name));
            }
        }

        // Display texture info.
        if (auto pTexture = getNormalMap())
        {
            widget.text("Normal map: " + pTexture->getSourcePath().string());
            widget.text("Texture info: " + std::to_string(pTexture->getWidth()) + "x" + std::to_string(pTexture->getHeight()) + " (" + to_string(pTexture->getFormat()) + ")");
            widget.image("Normal map", pTexture, float2(100.f));
            if (widget.button("Remove texture##NormalMap")) setNormalMap(nullptr);
        }
        else
        {
            widget.text("Normal map: N/A");
        }

        if (auto pTexture = getTexture(TextureSlot::Index))
        {
            widget.text("Index map: " + pTexture->getSourcePath().string());
            widget.text("Texture info: " + std::to_string(pTexture->getWidth()) + "x" + std::to_string(pTexture->getHeight()) + " (" + to_string(pTexture->getFormat()) + ")");
            if (widget.button("Remove texture##IndexMap")) setTexture(TextureSlot::Index, nullptr);
        }
        else
        {
            widget.text("Index map: N/A");
        }

        // Restore update flags.
        changed |= mUpdates != UpdateFlags::None;
        markUpdates(prevUpdates | mUpdates);

        return changed;
    }

    Material::UpdateFlags MERLMixMaterial::update(MaterialSystem* pOwner)
    {
        FALCOR_ASSERT(pOwner);

        auto flags = Material::UpdateFlags::None;
        if (mUpdates != Material::UpdateFlags::None)
        {
            // Update buffers.
            uint32_t bufferID = pOwner->addBuffer(mpBRDFData);
            if (mData.bufferID != bufferID)
                mUpdates |= Material::UpdateFlags::DataChanged;
            mData.bufferID = bufferID;

            // Update texture handles.
            updateTextureHandle(pOwner, TextureSlot::Normal, mData.texNormalMap);
            updateTextureHandle(pOwner, TextureSlot::Index, mData.texIndexMap);
            updateTextureHandle(pOwner, mpAlbedoLUT, mData.texAlbedoLUT);

            // Update samplers.
            uint prevFlags = mData.flags;
            mData.setLUTSamplerID(pOwner->addTextureSampler(mpLUTSampler));
            mData.setIndexSamplerID(pOwner->addTextureSampler(mpIndexSampler));
            if (mData.flags != prevFlags)
                mUpdates |= Material::UpdateFlags::DataChanged;

            updateDefaultTextureSamplerID(pOwner, mpDefaultSampler);

            flags |= mUpdates;
            mUpdates = Material::UpdateFlags::None;
        }

        return flags;
    }

    bool MERLMixMaterial::isEqual(const Material::SharedPtr& pOther) const
    {
        auto other = std::dynamic_pointer_cast<MERLMixMaterial>(pOther);
        if (!other) return false;

        if (!isBaseEqual(*other)) return false;

        // Check if the list loaded BRDFs is identical.
        if (mBRDFs.size() != other->mBRDFs.size()) return false;
        for (size_t i = 0; i < mBRDFs.size(); i++)
        {
            if (!(mBRDFs[i] == other->mBRDFs[i]))
                return false;
        }

        // Compare samplers.
        if (mpDefaultSampler->getDesc() != other->mpDefaultSampler->getDesc()) return false;

        return true;
    }

    Program::ShaderModuleList MERLMixMaterial::getShaderModules() const
    {
        return { Program::ShaderModule(kShaderFile) };
    }

    Program::TypeConformanceList MERLMixMaterial::getTypeConformances() const
    {
        return { {{"MERLMixMaterial", "IMaterial"}, (uint32_t)MaterialType::MERLMix} };
    }

    bool MERLMixMaterial::setTexture(const TextureSlot slot, const Texture::SharedPtr& pTexture)
    {
        if (!Material::setTexture(slot, pTexture)) return false;

        // Update additional metadata about texture usage.
        switch (slot)
        {
        case TextureSlot::Normal:
            updateNormalMapType();
            break;
        case TextureSlot::Index:
            updateIndexMapType();
            break;
        default:
            break;
        }

        return true;
    }

    void MERLMixMaterial::setDefaultTextureSampler(const Sampler::SharedPtr& pSampler)
    {
        if (pSampler != mpDefaultSampler)
        {
            mpDefaultSampler = pSampler;
            markUpdates(UpdateFlags::ResourcesChanged);
        }
    }

    void MERLMixMaterial::updateNormalMapType()
    {
        NormalMapType type = detectNormalMapType(getNormalMap());
        if (mData.getNormalMapType() != type)
        {
            mData.setNormalMapType(type);
            markUpdates(UpdateFlags::DataChanged);
        }
    }

    void MERLMixMaterial::updateIndexMapType()
    {
        auto tex = getTexture(TextureSlot::Index);
        if (tex)
        {
            // Verify that index map is in uncompressed 8-bit unorm format.
            // The shader requires this because the BRDF index is computed by scaling the unorm value by 255.
            ResourceFormat format = tex->getFormat();
            checkInvariant(!isSrgbFormat(format), "MERLMixMaterial: Index map must not be in SRGB format.");

            switch (format)
            {
            case ResourceFormat::R8Unorm:
            case ResourceFormat::RG8Unorm:
            case ResourceFormat::RGBA8Unorm:
            case ResourceFormat::BGRA8Unorm:
            case ResourceFormat::BGRX8Unorm:
                break;
            default:
                throw RuntimeError(fmt::format("MERLMixMaterial: Index map unsupported format ({}).", to_string(format)));
            }
        }
    }

    FALCOR_SCRIPT_BINDING(MERLMixMaterial)
    {
        using namespace pybind11::literals;

        FALCOR_SCRIPT_BINDING_DEPENDENCY(Material)

        pybind11::class_<MERLMixMaterial, Material, MERLMixMaterial::SharedPtr> material(m, "MERLMixMaterial");
        auto create = [](const std::string& name, const std::vector<std::filesystem::path>& paths)
        {
            return MERLMixMaterial::create(getActivePythonSceneBuilder().getDevice(), name, paths);
        };
        material.def(pybind11::init(create), "name"_a, "paths"_a); // PYTHONDEPRECATED
    }
}
