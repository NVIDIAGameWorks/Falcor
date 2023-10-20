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
#include "MaterialSystem.h"
#include "StandardMaterial.h"
#include "Core/API/Device.h"
#include "Utils/Logger.h"
#include "Utils/StringUtils.h"
#include "MaterialTypeRegistry.h"
#include <numeric>

namespace Falcor
{
    namespace
    {
        const std::string kShaderFilename = "Scene/Material/MaterialSystem.slang";
        const std::string kMaterialDataName = "materialData";
        const std::string kMaterialSamplersName = "materialSamplers";
        const std::string kMaterialTexturesName = "materialTextures";
        const std::string kMaterialBuffersName = "materialBuffers";
        const std::string kMaterialTextures3DName = "materialTextures3D";

        const size_t kMaxSamplerCount = 1ull << MaterialHeader::kSamplerIDBits;
        const size_t kMaxTextureCount = 1ull << TextureHandle::kTextureIDBits;
        const size_t kMaxBufferCountPerMaterial = 1; // This is a conservative estimation of how many buffer descriptors to allocate per material. Most materials don't use any auxiliary data buffers.

        // Helper to check if a material is a standard material using the SpecGloss shading model.
        // We keep track of these as an optimization because most scenes do not use this shading model.
        bool isSpecGloss(const ref<Material>& pMaterial)
        {
            if (pMaterial->getType() == MaterialType::Standard)
            {
                return static_ref_cast<StandardMaterial>(pMaterial)->getShadingModel() == ShadingModel::SpecGloss;
            }
            return false;
        }
    }

    MaterialSystem::MaterialSystem(ref<Device> pDevice)
        : mpDevice(pDevice)
    {
        FALCOR_ASSERT(kMaxSamplerCount <= mpDevice->getLimits().maxShaderVisibleSamplers);

        mpFence = mpDevice->createFence();
        mpTextureManager = std::make_unique<TextureManager>(mpDevice, kMaxTextureCount);

        // Create a default texture sampler.
        Sampler::Desc desc;
        desc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
        desc.setMaxAnisotropy(8);
        mpDefaultTextureSampler = mpDevice->createSampler(desc);
    }

    void MaterialSystem::renderUI(Gui::Widgets& widget)
    {
        auto showMaterial = [&](uint32_t materialID, const std::string& label) {
            const auto& pMaterial = mMaterials[materialID];
            if (auto materialGroup = widget.group(label))
            {
                if (pMaterial->renderUI(materialGroup)) uploadMaterial(materialID);
            }
        };

        widget.checkbox("Sort by name", mSortMaterialsByName);
        if (mSortMaterialsByName)
        {
            for (uint32_t materialID : mSortedMaterialIndices)
            {
                auto label = mMaterials[materialID]->getName() + " (#" + std::to_string(materialID) + ")";
                showMaterial(materialID, label);
            }
        }
        else
        {
            uint32_t materialID = 0;
            for (auto& material : mMaterials)
            {
                auto label = std::to_string(materialID) + ": " + material->getName();
                showMaterial(materialID, label);
                materialID++;
            }
        }
    }

    void MaterialSystem::updateUI()
    {
        // Construct a vector of indices that sort the materials by case-insensitive name.
        // TODO: Material could have changed names since last sort. This currently isn't detected.
        mSortedMaterialIndices.resize(mMaterials.size());
        std::iota(mSortedMaterialIndices.begin(), mSortedMaterialIndices.end(), 0);
        std::sort(mSortedMaterialIndices.begin(), mSortedMaterialIndices.end(), [this](uint32_t a, uint32_t b) {
            const std::string& astr = mMaterials[a]->getName();
            const std::string& bstr = mMaterials[b]->getName();
            const auto r = std::mismatch(astr.begin(), astr.end(), bstr.begin(), bstr.end(), [](uint8_t l, uint8_t r) { return tolower(l) == tolower(r); });
            return r.second != bstr.end() && (r.first == astr.end() || tolower(*r.first) < tolower(*r.second));
            });
    }

    void MaterialSystem::setDefaultTextureSampler(const ref<Sampler>& pSampler)
    {
        mpDefaultTextureSampler = pSampler;
        for (const auto& pMaterial : mMaterials)
        {
            pMaterial->setDefaultTextureSampler(pSampler);
        }
    }

    uint32_t MaterialSystem::addTextureSampler(const ref<Sampler>& pSampler)
    {
        FALCOR_ASSERT(pSampler);
        auto isEqual = [&pSampler](const ref<Sampler>& pOther) {
            return pSampler->getDesc() == pOther->getDesc();
        };

        // Reuse previously added samplers. We compare by sampler desc.
        if (auto it = std::find_if(mTextureSamplers.begin(), mTextureSamplers.end(), isEqual); it != mTextureSamplers.end())
        {
            return (uint32_t)std::distance(mTextureSamplers.begin(), it);
        }

        // Add sampler.
        if (mTextureSamplers.size() >= kMaxSamplerCount)
        {
            FALCOR_THROW("Too many samplers");
        }
        const uint32_t samplerID = static_cast<uint32_t>(mTextureSamplers.size());

        mTextureSamplers.push_back(pSampler);
        mSamplersChanged = true;

        return samplerID;
    }

    uint32_t MaterialSystem::addBuffer(const ref<Buffer>& pBuffer)
    {
        FALCOR_ASSERT(pBuffer);

        // Reuse previously added buffers. We compare by pointer as the contents of the buffers is unknown.
        if (auto it = std::find_if(mBuffers.begin(), mBuffers.end(), [&](auto pOther) { return pBuffer == pOther; }); it != mBuffers.end())
        {
            return (uint32_t)std::distance(mBuffers.begin(), it);
        }

        // Add buffer.
        FALCOR_ASSERT(!mMaterialsChanged);
        if (mBuffers.size() >= mBufferDescCount)
        {
            FALCOR_THROW("Too many buffers");
        }
        const uint32_t bufferID = static_cast<uint32_t>(mBuffers.size());

        mBuffers.push_back(pBuffer);
        mBuffersChanged = true;

        return bufferID;
    }

    void MaterialSystem::replaceBuffer(uint32_t id, const ref<Buffer>& pBuffer)
    {
        FALCOR_ASSERT(pBuffer);
        FALCOR_CHECK(id < mBuffers.size(), "'id' is out of bounds.");

        mBuffers[id] = pBuffer;
        mBuffersChanged = true;
    }

    uint32_t MaterialSystem::addTexture3D(const ref<Texture>& pTexture)
    {
        FALCOR_ASSERT(pTexture && pTexture->getType() == Texture::Type::Texture3D && pTexture->getSampleCount() == 1);

        // Reuse previously added texture.
        if (auto it = std::find_if(mTextures3D.begin(), mTextures3D.end(), [&](auto pOther) { return pTexture == pOther; }); it != mTextures3D.end())
            return (uint32_t)std::distance(mTextures3D.begin(), it);

        if (mTextures3D.size() >= mTexture3DDescCount)
            FALCOR_THROW("Too many 3D textures");

        const uint32_t textureID = static_cast<uint32_t>(mTextures3D.size());

        mTextures3D.push_back(pTexture);
        mTextures3DChanged = true;

        return textureID;
    }

    MaterialID MaterialSystem::addMaterial(const ref<Material>& pMaterial)
    {
        FALCOR_CHECK(pMaterial != nullptr, "'pMaterial' is missing");

        // Reuse previously added materials.
        if (auto it = std::find(mMaterials.begin(), mMaterials.end(), pMaterial); it != mMaterials.end())
        {
            return MaterialID{(size_t)std::distance(mMaterials.begin(), it) };
        }

        // Add material.
        if (mMaterials.size() >= std::numeric_limits<uint32_t>::max())
        {
            FALCOR_THROW("Too many materials");
        }
        const MaterialID materialID{ mMaterials.size() };

        if (pMaterial->getDefaultTextureSampler() == nullptr)
        {
            pMaterial->setDefaultTextureSampler(mpDefaultTextureSampler);
        }

        pMaterial->registerUpdateCallback([this](auto flags) { mMaterialUpdates |= flags; });
        mMaterials.push_back(pMaterial);
        mMaterialsChanged = true;

        return materialID;
    }

    void MaterialSystem::replaceMaterial(const MaterialID materialID, const ref<Material>& pReplacement)
    {
        FALCOR_CHECK(materialID.isValid() && materialID.get() < mMaterials.size(), "Material ID is invalid.");
        FALCOR_CHECK(pReplacement != nullptr, "Replacement material is missing.");

        // Mark descriptors used by the deleted material as reserved.
        // This is a workaround until we have dynamically sized descriptor arrays and proper resource tracking.
        const auto& prevMtl = mMaterials[materialID.get()];
        mReservedTextureDescCount += prevMtl->getMaxTextureCount();
        mReservedBufferDescCount += prevMtl->getMaxBufferCount();
        mReservedTexture3DDescCount += prevMtl->getMaxTexture3DCount();

        // Prepare replacement material.
        if (pReplacement->getDefaultTextureSampler() == nullptr)
        {
            pReplacement->setDefaultTextureSampler(mpDefaultTextureSampler);
        }
        pReplacement->registerUpdateCallback([this](auto flags) { mMaterialUpdates |= flags; });

        // Replace the material.
        mMaterials[materialID.get()] = pReplacement;
        mMaterialsChanged = true;
    }

    void MaterialSystem::replaceMaterial(const ref<Material>& pMaterial, const ref<Material>& pReplacement)
    {
        FALCOR_CHECK(pMaterial != nullptr, "'pMaterial' is missing");

        // Find material to replace.
        if (auto it = std::find(mMaterials.begin(), mMaterials.end(), pMaterial); it != mMaterials.end())
        {
            auto materialID = MaterialID{ (size_t)std::distance(mMaterials.begin(), it) };
            replaceMaterial(materialID, pReplacement);
        }
        else
        {
            FALCOR_THROW("Material does not exist");
        }
    }

    uint32_t MaterialSystem::getMaterialCountByType(const MaterialType type) const
    {
        FALCOR_CHECK(!mMaterialsChanged, "Materials have changed. Call update() first.");
        size_t index = (size_t)type;
        return (index < mMaterialCountByType.size()) ? mMaterialCountByType[index] : 0;
    }

    std::set<MaterialType> MaterialSystem::getMaterialTypes() const
    {
        FALCOR_CHECK(!mMaterialsChanged, "Materials have changed. Call update() first.");
        return mMaterialTypes;
    }

    bool MaterialSystem::hasMaterialType(MaterialType type) const
    {
        FALCOR_CHECK(!mMaterialsChanged, "Materials have changed. Call update() first.");
        return mMaterialTypes.find(type) != mMaterialTypes.end();
    }

    bool MaterialSystem::hasMaterial(const MaterialID materialID) const
    {
        // The materials are sequentially indexed without gaps for now. Check if the ID is within range.
        return materialID.isValid() ? materialID.get() < mMaterials.size() : false;
    }

    const ref<Material>& MaterialSystem::getMaterial(const MaterialID materialID) const
    {
        FALCOR_CHECK(materialID.get() < mMaterials.size(), "MaterialID is out of range.");
        return mMaterials[materialID.get()];
    }

    ref<Material> MaterialSystem::getMaterialByName(const std::string& name) const
    {
        for (const auto& pMaterial : mMaterials)
        {
            if (pMaterial->getName() == name) return pMaterial;
        }
        return nullptr;
    }

    size_t MaterialSystem::removeDuplicateMaterials(std::vector<MaterialID>& idMap)
    {
        std::vector<ref<Material>> uniqueMaterials;
        idMap.resize(mMaterials.size());

        // Find unique set of materials.
        for (MaterialID id{ 0 }; id.get() < mMaterials.size(); ++id)
        {
            const auto& pMaterial = mMaterials[id.get()];
            auto it = std::find_if(uniqueMaterials.begin(), uniqueMaterials.end(), [&pMaterial](const auto& m) { return m->isEqual(pMaterial); });
            if (it == uniqueMaterials.end())
            {
                idMap[id.get()] = MaterialID{ uniqueMaterials.size() };
                uniqueMaterials.push_back(pMaterial);
            }
            else
            {
                logInfo("Removing duplicate material '{}' (duplicate of '{}').", pMaterial->getName(), (*it)->getName());
                idMap[id.get()] = MaterialID{ (size_t)std::distance(uniqueMaterials.begin(), it) };
            }
        }

        size_t removed = mMaterials.size() - uniqueMaterials.size();
        if (removed > 0)
        {
            mMaterials = uniqueMaterials;
            mMaterialsChanged = true;
        }

        return removed;
    }

    void MaterialSystem::optimizeMaterials()
    {
        // Gather a list of all textures to analyze.
        std::vector<std::pair<ref<Material>, Material::TextureSlot>> materialSlots;
        std::vector<ref<Texture>> textures;
        size_t maxCount = mMaterials.size() * (size_t)Material::TextureSlot::Count;
        materialSlots.reserve(maxCount);
        textures.reserve(maxCount);

        for (const auto& pMaterial : mMaterials)
        {
            for (uint32_t i = 0; i < (uint32_t)Material::TextureSlot::Count; i++)
            {
                auto slot = (Material::TextureSlot)i;
                if (auto pTexture = pMaterial->getTexture(slot))
                {
                    materialSlots.push_back({ pMaterial, slot });
                    textures.push_back(pTexture);
                }
            }
        }

        if (textures.empty()) return;

        // Analyze the textures.
        logInfo("Analyzing {} material textures.", textures.size());

        RenderContext* pRenderContext = mpDevice->getRenderContext();

        TextureAnalyzer analyzer(mpDevice);
        auto pResults = mpDevice->createBuffer(textures.size() * TextureAnalyzer::getResultSize(), ResourceBindFlags::UnorderedAccess);
        analyzer.analyze(pRenderContext, textures, pResults);

        // Copy result to staging buffer for readback.
        // This is mostly to avoid a full flush and the associated perf warning.
        // We do not have any other useful GPU work, but unrelated GPU tasks can be in flight.
        auto pResultsStaging = mpDevice->createBuffer(textures.size() * TextureAnalyzer::getResultSize(), ResourceBindFlags::None, MemoryType::ReadBack);
        pRenderContext->copyResource(pResultsStaging.get(), pResults.get());
        pRenderContext->submit(false);
        pRenderContext->signal(mpFence.get());

        // Wait for results to become available. Then optimize the materials.
        mpFence->wait();
        const TextureAnalyzer::Result* results = static_cast<const TextureAnalyzer::Result*>(pResultsStaging->map(Buffer::MapType::Read));
        Material::TextureOptimizationStats stats = {};

        for (size_t i = 0; i < textures.size(); i++)
        {
            materialSlots[i].first->optimizeTexture(materialSlots[i].second, results[i], stats);
        }

        pResultsStaging->unmap();

        // Log optimization stats.
        if (size_t totalRemoved = std::accumulate(stats.texturesRemoved.begin(), stats.texturesRemoved.end(), 0ull); totalRemoved > 0)
        {
            logInfo("Optimized materials by removing {} constant textures.", totalRemoved);
            for (size_t slot = 0; slot < (size_t)Material::TextureSlot::Count; slot++)
            {
                logInfo(padStringToLength("  " + to_string((Material::TextureSlot)slot) + ":", 26) + std::to_string(stats.texturesRemoved[slot]));
            }
        }

        if (stats.disabledAlpha > 0) logInfo("Optimized materials by disabling alpha test for {} materials.", stats.disabledAlpha);
        if (stats.constantBaseColor > 0) logWarning("Materials have {} base color maps of constant value with non-constant alpha channel.", stats.constantBaseColor);
        if (stats.constantNormalMaps > 0) logWarning("Materials have {} normal maps of constant value. Please update the asset to optimize performance.", stats.constantNormalMaps);
    }

    Material::UpdateFlags MaterialSystem::update(bool forceUpdate)
    {
        // If materials were added/removed since last update, we update all metadata
        // and trigger re-creation of the parameter block and update of all materials.
        if (forceUpdate || mMaterialsChanged)
        {
            updateMetadata();
            updateUI();

            mpMaterialsBlock = nullptr;
            mMaterialsChanged = false;
            forceUpdate = true;
        }

        // Update all materials.
        // Do either a full update of all materials with deferred texture loading, or an update of just the dynamic materials.
        // We track per-material update flags along with the combined update flags across all materials.
        // Note that materials can record updates in between calls to update() and/or return flags from their update() calls.
        Material::UpdateFlags updateFlags = Material::UpdateFlags::None;
        mMaterialsUpdateFlags.resize(mMaterials.size());
        std::fill(mMaterialsUpdateFlags.begin(), mMaterialsUpdateFlags.end(), Material::UpdateFlags::None);

        auto updateMaterial = [&](const MaterialID materialID) {
            auto& pMaterial = getMaterial(materialID);
            if (pMaterial->mpDevice != mpDevice)
                FALCOR_THROW("Material '{}' was created with a different device than the MaterialSystem.", pMaterial->getName());
            Material::UpdateFlags flags = pMaterial->update(this);
            // Record update flags.
            mMaterialsUpdateFlags[materialID.get()] = flags;
            updateFlags |= flags;
        };

        if (forceUpdate || mMaterialUpdates != Material::UpdateFlags::None)
        {
            mpTextureManager->beginDeferredLoading();

            for (size_t materialIdx = 0; materialIdx < mMaterials.size(); ++materialIdx)
                updateMaterial(MaterialID{ materialIdx });

            mpTextureManager->endDeferredLoading();
        }
        else
        {
            for (const auto& materialID : mDynamicMaterialIDs)
                updateMaterial(materialID);
        }

        // Include updates recorded since last update.
        // After this point no more material changes are expected.
        updateFlags |= mMaterialUpdates;
        mMaterialUpdates = Material::UpdateFlags::None;

        // Create parameter block if needed.
        if (!mpMaterialsBlock)
        {
            createParameterBlock();

            // Set update flags if parameter block changes.
            // TODO: We may want to introduce MaterialSystem::UpdateFlags instead of re-using the material flags.
            updateFlags |= Material::UpdateFlags::DataChanged | Material::UpdateFlags::ResourcesChanged;

            forceUpdate = true; // Trigger full upload of all materials.
        }

        // Upload all modified materials.
        if (forceUpdate || is_set(updateFlags, Material::UpdateFlags::DataChanged))
        {
            for (uint32_t materialID = 0; materialID < (uint32_t)mMaterials.size(); ++materialID)
            {
                if (forceUpdate || is_set(mMaterialsUpdateFlags[materialID], Material::UpdateFlags::DataChanged))
                {
                    uploadMaterial(materialID);
                }
            }
        }

        auto blockVar = mpMaterialsBlock->getRootVar();

        // Update samplers.
        if (forceUpdate || mSamplersChanged)
        {
            auto var = blockVar[kMaterialSamplersName];
            for (size_t i = 0; i < mTextureSamplers.size(); i++)
            {
                var[i] = mTextureSamplers[i];
            }
            mSamplersChanged = false;
        }

        // Update textures.
        if (forceUpdate || is_set(updateFlags, Material::UpdateFlags::ResourcesChanged))
        {
            FALCOR_ASSERT(!mMaterialsChanged);
            mpTextureManager->bindShaderData(blockVar[kMaterialTexturesName], mTextureDescCount,
                blockVar["udimIndirection"]);
        }

        // Update buffers.
        if (forceUpdate || mBuffersChanged)
        {
            auto var = blockVar[kMaterialBuffersName];
            for (size_t i = 0; i < mBuffers.size(); i++)
            {
                var[i] = mBuffers[i];
            }
            mBuffersChanged = false;
        }

        // Update 3D textures.
        if (forceUpdate || mTextures3DChanged)
        {
            auto var = blockVar[kMaterialTextures3DName];
            for (size_t i = 0; i < mTextures3D.size(); i++)
            {
                var[i] = mTextures3D[i];
            }
            mTextures3DChanged = false;
        }


        // Update shader modules and type conformances.
        // This is done by iterating over all materials to query their properties.
        // We de-duplicate the result by material type to store the unique set of shader modules and type conformances.
        // Note that this means the shader code for all materials of the same type is assumed to be identical.
        if (forceUpdate || is_set(updateFlags, Material::UpdateFlags::CodeChanged))
        {
            mShaderModules.clear();
            mTypeConformances.clear();

            for (const auto& pMaterial : mMaterials)
            {
                const auto materialType = pMaterial->getType();
                if (mTypeConformances.find(materialType) == mTypeConformances.end())
                {
                    auto modules = pMaterial->getShaderModules();
                    mShaderModules.insert(mShaderModules.end(), modules.begin(), modules.end());
                    mTypeConformances[materialType] = pMaterial->getTypeConformances();
                }
            }
        }

        FALCOR_CHECK(mMaterialUpdates == Material::UpdateFlags::None, "Unexpected material updates.");
        return updateFlags;
    }

    void MaterialSystem::updateMetadata()
    {
        mTextureDescCount = mReservedTextureDescCount;
        mBufferDescCount = mReservedBufferDescCount;
        mTexture3DDescCount = mReservedTexture3DDescCount;

        mMaterialCountByType.resize(getMaterialTypeCount());
        std::fill(mMaterialCountByType.begin(), mMaterialCountByType.end(), 0);
        mMaterialTypes.clear();
        mHasSpecGlossStandardMaterial = false;
        mDynamicMaterialIDs.clear();

        for (size_t materialIdx = 0; materialIdx < mMaterials.size(); materialIdx++)
        {
            const auto& pMaterial = mMaterials[materialIdx];
            const MaterialID materialID{ materialIdx };

            if (pMaterial->isDynamic())
                mDynamicMaterialIDs.push_back(materialID);

            // Update descriptor counts. These counts will be reported by getDefines().
            // TODO: Remove this when unbounded descriptor arrays are supported (#1321).
            mTextureDescCount += pMaterial->getMaxTextureCount();
            mBufferDescCount += pMaterial->getMaxBufferCount();
            mTexture3DDescCount += pMaterial->getMaxTexture3DCount();

            // Update material type info.
            size_t index = (size_t)pMaterial->getType();
            FALCOR_ASSERT(index < mMaterialCountByType.size());
            mMaterialCountByType[index]++;
            mMaterialTypes.insert(pMaterial->getType());
            if (isSpecGloss(pMaterial)) mHasSpecGlossStandardMaterial = true;
        }

        FALCOR_CHECK(mMaterialTypes.find(MaterialType::Unknown) == mMaterialTypes.end(), "Unknown material type found. Make sure all material types are registered.");
    }

    MaterialSystem::MaterialStats MaterialSystem::getStats() const
    {
        FALCOR_CHECK(!mMaterialsChanged, "Materials have changed. Call update() first.");

        MaterialStats s = {};

        s.materialTypeCount = mMaterialTypes.size();
        s.materialCount = mMaterials.size();
        s.materialOpaqueCount = 0;
        s.materialMemoryInBytes += mpMaterialDataBuffer ? mpMaterialDataBuffer->getSize() : 0;

        for (const auto& pMaterial : mMaterials)
        {
            if (pMaterial->isOpaque()) s.materialOpaqueCount++;
        }

        const auto textureStats = mpTextureManager->getStats();
        s.textureCount = textureStats.textureCount;
        s.textureCompressedCount = textureStats.textureCompressedCount;
        s.textureTexelCount = textureStats.textureTexelCount;
        s.textureTexelChannelCount = textureStats.textureTexelChannelCount;
        s.textureMemoryInBytes = textureStats.textureMemoryInBytes;

        return s;
    }

    void MaterialSystem::getDefines(DefineList& defines) const
    {
        FALCOR_CHECK(!mMaterialsChanged, "Materials have changed. Call update() first.");

        size_t materialInstanceByteSize = 0;
        for (auto& it : mMaterials)
            materialInstanceByteSize = std::max(materialInstanceByteSize, it->getMaterialInstanceByteSize());

        defines.add("MATERIAL_SYSTEM_SAMPLER_DESC_COUNT", std::to_string(kMaxSamplerCount));
        defines.add("MATERIAL_SYSTEM_TEXTURE_DESC_COUNT", std::to_string(mTextureDescCount));
        defines.add("MATERIAL_SYSTEM_BUFFER_DESC_COUNT", std::to_string(mBufferDescCount));
        defines.add("MATERIAL_SYSTEM_TEXTURE_3D_DESC_COUNT", std::to_string(mTexture3DDescCount));
        defines.add("MATERIAL_SYSTEM_UDIM_INDIRECTION_ENABLED", mpTextureManager->getUdimIndirectionCount() > 0 ? "1" : "0");
        defines.add("MATERIAL_SYSTEM_HAS_SPEC_GLOSS_MATERIALS", mHasSpecGlossStandardMaterial ? "1" : "0");
        defines.add("FALCOR_MATERIAL_INSTANCE_SIZE", std::to_string(materialInstanceByteSize));

        // Add defines specified by the materials.
        // We ensure that two materials cannot set the same define to mismatching values.
        for (const auto& material : mMaterials)
        {
            DefineList materialDefines = material->getDefines();
            for (const auto& [name, value] : materialDefines)
            {
                if (auto it = defines.find(name); it != defines.end())
                {
                    FALCOR_CHECK(it->second == value, "Mismatching values '{}' and '{}' for material define '{}'.", name, it->second, value);
                }
                else
                {
                    defines.add(name, value);
                }
            }
        }
    }

    void MaterialSystem::getTypeConformances(TypeConformanceList& typeConformances) const
    {
        for (const auto& it : mTypeConformances)
        {
            typeConformances.add(it.second);
        }
        typeConformances.add("NullPhaseFunction", "IPhaseFunction", 0);
        typeConformances.add("IsotropicPhaseFunction", "IPhaseFunction", 1);
        typeConformances.add("HenyeyGreensteinPhaseFunction", "IPhaseFunction", 2);
        typeConformances.add("DualHenyeyGreensteinPhaseFunction", "IPhaseFunction", 3);
    }

    TypeConformanceList MaterialSystem::getTypeConformances(const MaterialType type) const
    {
        auto it = mTypeConformances.find(type);
        if (it == mTypeConformances.end())
        {
            FALCOR_THROW(fmt::format("No type conformances for material type '{}'.", to_string(type)));
        }
        return it->second;
    }

    ProgramDesc::ShaderModuleList MaterialSystem::getShaderModules() const
    {
        FALCOR_CHECK(!mMaterialsChanged, "Materials have changed. Call update() first.");
        return mShaderModules;
    }

    void MaterialSystem::getShaderModules(ProgramDesc::ShaderModuleList& shaderModuleList) const
    {
        FALCOR_CHECK(!mMaterialsChanged, "Materials have changed. Call update() first.");
        shaderModuleList.insert(shaderModuleList.end(), mShaderModules.begin(), mShaderModules.end());
    }

    void MaterialSystem::bindShaderData(const ShaderVar& var) const
    {
        FALCOR_CHECK(mpMaterialsBlock != nullptr && !mMaterialsChanged, "Parameter block is not ready. Call update() first.");
        var = mpMaterialsBlock;
    }

    void MaterialSystem::createParameterBlock()
    {
        // Create parameter block.
        DefineList defines = getDefines();
        defines.add("MATERIAL_SYSTEM_PARAMETER_BLOCK");
        auto pPass = ComputePass::create(mpDevice, kShaderFilename, "main", defines);
        auto pReflector = pPass->getProgram()->getReflector()->getParameterBlock("gMaterialsBlock");
        FALCOR_ASSERT(pReflector);

        mpMaterialsBlock = ParameterBlock::create(mpDevice, pReflector);
        FALCOR_ASSERT(mpMaterialsBlock);

        // Verify that the material data struct size on the GPU matches the host-side size.
        auto reflVar = mpMaterialsBlock->getReflection()->findMember(kMaterialDataName);
        FALCOR_ASSERT(reflVar);
        auto reflResType = reflVar->getType()->asResourceType();
        FALCOR_ASSERT(reflResType && reflResType->getType() == ReflectionResourceType::Type::StructuredBuffer);
        auto byteSize = reflResType->getStructType()->getByteSize();
        if (byteSize != sizeof(MaterialDataBlob))
        {
            FALCOR_THROW("MaterialSystem material data buffer has unexpected struct size");
        }

        auto blockVar = mpMaterialsBlock->getRootVar();

        // Create materials data buffer.
        if (!mMaterials.empty() && (!mpMaterialDataBuffer || mpMaterialDataBuffer->getElementCount() < mMaterials.size()))
        {
            mpMaterialDataBuffer = mpDevice->createStructuredBuffer(blockVar[kMaterialDataName], (uint32_t)mMaterials.size(), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, nullptr, false);
            mpMaterialDataBuffer->setName("MaterialSystem::mpMaterialDataBuffer");
        }

        // Bind resources to parameter block.
        blockVar[kMaterialDataName] = !mMaterials.empty() ? mpMaterialDataBuffer : nullptr;
        blockVar["materialCount"] = getMaterialCount();
    }

    void MaterialSystem::uploadMaterial(const uint32_t materialID)
    {
        FALCOR_ASSERT(materialID < mMaterials.size());
        const auto& pMaterial = mMaterials[materialID];
        FALCOR_ASSERT(pMaterial);

        // TODO: On initial upload of materials, we could improve this by not having separate calls to setElement()
        // but instead prepare a buffer containing all data.
        FALCOR_ASSERT(mpMaterialDataBuffer);
        mpMaterialDataBuffer->setElement(materialID, pMaterial->getDataBlob());
    }
}
