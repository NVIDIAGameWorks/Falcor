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
#include "MaterialSystem.h"
#include "StandardMaterial.h"
#include "Core/API/Device.h"
#include "Utils/Logger.h"
#include "Utils/StringUtils.h"
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

        const size_t kMaxSamplerCount = 1ull << MaterialHeader::kSamplerIDBits;
        const size_t kMaxTextureCount = 1ull << TextureHandle::kTextureIDBits;
        const size_t kMaxBufferCountPerMaterial = 1; // This is a conservative estimation of how many buffer descriptors to allocate per material. Most materials don't use any auxiliary data buffers.

        // Helper to check if a material is a standard material using the SpecGloss shading model.
        // We keep track of these as an optimization because most scenes do not use this shading model.
        bool isSpecGloss(const Material::SharedPtr& pMaterial)
        {
            if (pMaterial->getType() == MaterialType::Standard)
            {
                return std::static_pointer_cast<StandardMaterial>(pMaterial)->getShadingModel() == ShadingModel::SpecGloss;
            }
            return false;
        }
    }

    MaterialSystem::SharedPtr MaterialSystem::create()
    {
        return SharedPtr(new MaterialSystem());
    }

    MaterialSystem::MaterialSystem()
    {
#ifdef FALCOR_D3D12
        FALCOR_ASSERT(kMaxSamplerCount <= D3D12DescriptorPool::getMaxShaderVisibleSamplerHeapSize());
#endif // FALCOR_D3D12

        mpFence = GpuFence::create();
        mpTextureManager = TextureManager::create(kMaxTextureCount);
        mMaterialCountByType.resize((size_t)MaterialType::Count, 0);

        // Create a default texture sampler.
        Sampler::Desc desc;
        desc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
        desc.setMaxAnisotropy(8);
        mpDefaultTextureSampler = Sampler::create(desc);
    }

    void MaterialSystem::finalize()
    {
        // Pre-allocate texture and buffer descriptors based on final material count. This count will be reported by getDefines().
        // This is needed just because Falcor currently has no mechanism for reporting to user code that scene defines have changed.
        // Note we allocate more descriptors here than is typically needed as many materials do not use all texture slots,
        // so there is some room for adding materials at runtime after scene creation until running into this limit.
        // TODO: Remove this when unbounded descriptor arrays are supported (#1321).
        mTextureDescCount = getMaterialCount() * (size_t)Material::TextureSlot::Count;

        mBufferDescCount = 0;
        for (const auto& material : mMaterials)
        {
            mBufferDescCount += material->getBufferCount();
        }
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

    void MaterialSystem::setDefaultTextureSampler(const Sampler::SharedPtr& pSampler)
    {
        mpDefaultTextureSampler = pSampler;
        for (const auto& pMaterial : mMaterials)
        {
            pMaterial->setDefaultTextureSampler(pSampler);
        }
    }

    uint32_t MaterialSystem::addTextureSampler(const Sampler::SharedPtr& pSampler)
    {
        FALCOR_ASSERT(pSampler);
        auto isEqual = [&pSampler](const Sampler::SharedPtr& pOther) {
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
            throw RuntimeError("Too many samplers");
        }
        const uint32_t samplerID = static_cast<uint32_t>(mTextureSamplers.size());

        mTextureSamplers.push_back(pSampler);
        mSamplersChanged = true;

        return samplerID;
    }

    uint32_t MaterialSystem::addBuffer(const Buffer::SharedPtr& pBuffer)
    {
        FALCOR_ASSERT(pBuffer);

        // Reuse previously added buffers. We compare by pointer as the contents of the buffers is unknown.
        if (auto it = std::find_if(mBuffers.begin(), mBuffers.end(), [&](auto pOther) { return pBuffer == pOther; }); it != mBuffers.end())
        {
            return (uint32_t)std::distance(mBuffers.begin(), it);
        }

        // Add buffer.
        if (mBuffers.size() >= mBufferDescCount)
        {
            throw RuntimeError("Too many buffers");
        }
        const uint32_t bufferID = static_cast<uint32_t>(mBuffers.size());

        mBuffers.push_back(pBuffer);
        mBuffersChanged = true;

        return bufferID;
    }

    void MaterialSystem::replaceBuffer(uint32_t id, const Buffer::SharedPtr& pBuffer)
    {
        FALCOR_ASSERT(pBuffer);
        checkArgument(id < mBuffers.size(), "'id' is out of bounds.");

        mBuffers[id] = pBuffer;
        mBuffersChanged = true;
    }

    MaterialID MaterialSystem::addMaterial(const Material::SharedPtr& pMaterial)
    {
        FALCOR_ASSERT(pMaterial);

        // Reuse previously added materials.
        if (auto it = std::find(mMaterials.begin(), mMaterials.end(), pMaterial); it != mMaterials.end())
        {
            return MaterialID{(size_t)std::distance(mMaterials.begin(), it) };
        }

        // Add material.
        if (mMaterials.size() >= std::numeric_limits<uint32_t>::max())
        {
            throw RuntimeError("Too many materials");
        }
        const MaterialID materialID{ mMaterials.size() };

        if (pMaterial->getDefaultTextureSampler() == nullptr)
        {
            pMaterial->setDefaultTextureSampler(mpDefaultTextureSampler);
        }

        pMaterial->registerUpdateCallback([this](auto flags) { mMaterialUpdates |= flags; });
        mMaterials.push_back(pMaterial);
        mMaterialsChanged = true;

        // Update metadata.
        mMaterialTypes.insert(pMaterial->getType());
        if (isSpecGloss(pMaterial)) mSpecGlossMaterialCount++;

        return materialID;
    }

    uint32_t MaterialSystem::getMaterialCountByType(const MaterialType type) const
    {
        size_t index = (size_t)type;
        FALCOR_ASSERT(index < mMaterialCountByType.size());
        return mMaterialCountByType[index];
    }

    const Material::SharedPtr& MaterialSystem::getMaterial(const MaterialID materialID) const
    {
        FALCOR_ASSERT_LT(materialID.get(), mMaterials.size());
        return mMaterials[materialID.get()];
    }

    Material::SharedPtr MaterialSystem::getMaterialByName(const std::string& name) const
    {
        for (const auto& pMaterial : mMaterials)
        {
            if (pMaterial->getName() == name) return pMaterial;
        }
        return nullptr;
    }

    size_t MaterialSystem::removeDuplicateMaterials(std::vector<MaterialID>& idMap)
    {
        std::vector<Material::SharedPtr> uniqueMaterials;
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

                // Update metadata.
                if (isSpecGloss(pMaterial)) mSpecGlossMaterialCount--;
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
        std::vector<std::pair<Material::SharedPtr, Material::TextureSlot>> materialSlots;
        std::vector<Texture::SharedPtr> textures;
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

        TextureAnalyzer::SharedPtr pAnalyzer = TextureAnalyzer::create();
        auto pResults = Buffer::create(textures.size() * TextureAnalyzer::getResultSize(), ResourceBindFlags::UnorderedAccess);
        pAnalyzer->analyze(gpDevice->getRenderContext(), textures, pResults);

        // Copy result to staging buffer for readback.
        // This is mostly to avoid a full flush and the associated perf warning.
        // We do not have any other useful GPU work, but unrelated GPU tasks can be in flight.
        auto pResultsStaging = Buffer::create(textures.size() * TextureAnalyzer::getResultSize(), ResourceBindFlags::None, Buffer::CpuAccess::Read);
        gpDevice->getRenderContext()->copyResource(pResultsStaging.get(), pResults.get());
        gpDevice->getRenderContext()->flush(false);
        mpFence->gpuSignal(gpDevice->getRenderContext()->getLowLevelData()->getCommandQueue());

        // Wait for results to become available. Then optimize the materials.
        mpFence->syncCpu();
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
        if (stats.constantBaseColor > 0) logWarning("Scene has {} base color maps of constant value with non-constant alpha channel.", stats.constantBaseColor);
        if (stats.constantNormalMaps > 0) logWarning("Scene has {} normal maps of constant value. Please update the asset to optimize performance.", stats.constantNormalMaps);
    }

    Material::UpdateFlags MaterialSystem::update(bool forceUpdate)
    {
        Material::UpdateFlags flags = Material::UpdateFlags::None;

        // Update metadata if materials changed.
        if (mMaterialsChanged)
        {
            std::fill(mMaterialCountByType.begin(), mMaterialCountByType.end(), 0);
            for (const auto& pMaterial : mMaterials)
            {
                size_t index = (size_t)pMaterial->getType();
                FALCOR_ASSERT(index < mMaterialCountByType.size());
                mMaterialCountByType[index]++;
            }

            updateUI();
        }

        // Create parameter block if needed.
        if (!mpMaterialsBlock || mMaterialsChanged)
        {
            createParameterBlock();

            // Set update flags if parameter block changes.
            // TODO: We may want to introduce MaterialSystem::UpdateFlags instead of re-using the material flags.
            flags |= Material::UpdateFlags::DataChanged | Material::UpdateFlags::ResourcesChanged;

            forceUpdate = true; // Trigger full upload of all materials
        }

        // Update all materials.
        if (forceUpdate || mMaterialUpdates != Material::UpdateFlags::None)
        {
            for (uint32_t materialID = 0; materialID < (uint32_t)mMaterials.size(); ++materialID)
            {
                auto& pMaterial = mMaterials[materialID];
                const auto materialUpdates = pMaterial->update(this);

                if (forceUpdate || materialUpdates != Material::UpdateFlags::None)
                {
                    uploadMaterial(materialID);

                    flags |= materialUpdates;
                }
            }
        }

        // Update samplers.
        if (forceUpdate || mSamplersChanged)
        {
            auto var = mpMaterialsBlock[kMaterialSamplersName];
            for (size_t i = 0; i < mTextureSamplers.size(); i++) var[i] = mTextureSamplers[i];
        }

        // Update textures.
        if (forceUpdate || is_set(flags, Material::UpdateFlags::ResourcesChanged))
        {
            mpTextureManager->setShaderData(mpMaterialsBlock[kMaterialTexturesName], mTextureDescCount);
        }

        // Update buffers.
        if (forceUpdate || mBuffersChanged)
        {
            auto var = mpMaterialsBlock[kMaterialBuffersName];
            for (size_t i = 0; i < mBuffers.size(); i++) var[i] = mBuffers[i];
        }

        // Update shader modules and type conformances.
        // This is done by iterating over all materials to query their properties.
        // We de-duplicate the result to store the unique set of shader modules and type conformances needed.
        if (forceUpdate || is_set(flags, Material::UpdateFlags::CodeChanged))
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

        mSamplersChanged = false;
        mBuffersChanged = false;
        mMaterialsChanged = false;
        mMaterialUpdates = Material::UpdateFlags::None;

        return flags;
    }

    MaterialSystem::MaterialStats MaterialSystem::getStats() const
    {
        MaterialStats s = {};

        s.materialTypeCount = mMaterialTypes.size();
        s.materialCount = mMaterials.size();
        s.materialOpaqueCount = 0;
        s.materialMemoryInBytes += mpMaterialDataBuffer ? mpMaterialDataBuffer->getSize() : 0;

        std::set<Texture::SharedPtr> textures;
        for (const auto& pMaterial : mMaterials)
        {
            for (uint32_t i = 0; i < (uint32_t)Material::TextureSlot::Count; i++)
            {
                auto pTexture = pMaterial->getTexture((Material::TextureSlot)i);
                if (pTexture) textures.insert(pTexture);
            }

            if (pMaterial->isOpaque()) s.materialOpaqueCount++;
        }

        s.textureCount = textures.size();
        s.textureCompressedCount = 0;
        s.textureTexelCount = 0;
        s.textureMemoryInBytes = 0;

        for (const auto& t : textures)
        {
            s.textureTexelCount += t->getTexelCount();
            s.textureMemoryInBytes += t->getTextureSizeInBytes();
            if (isCompressedFormat(t->getFormat())) s.textureCompressedCount++;
        }

        return s;
    }

    Shader::DefineList MaterialSystem::getDefaultDefines()
    {
        Shader::DefineList defines;
        defines.add("MATERIAL_SYSTEM_SAMPLER_DESC_COUNT", std::to_string(kMaxSamplerCount));
        defines.add("MATERIAL_SYSTEM_TEXTURE_DESC_COUNT", "0");
        defines.add("MATERIAL_SYSTEM_BUFFER_DESC_COUNT", "0");
        defines.add("MATERIAL_SYSTEM_HAS_SPEC_GLOSS_MATERIALS", "0");

        return defines;
    }

    Shader::DefineList MaterialSystem::getDefines() const
    {
        Shader::DefineList defines;
        defines.add("MATERIAL_SYSTEM_SAMPLER_DESC_COUNT", std::to_string(kMaxSamplerCount));
        defines.add("MATERIAL_SYSTEM_TEXTURE_DESC_COUNT", std::to_string(mTextureDescCount));
        defines.add("MATERIAL_SYSTEM_BUFFER_DESC_COUNT", std::to_string(mBufferDescCount));
        defines.add("MATERIAL_SYSTEM_HAS_SPEC_GLOSS_MATERIALS", mSpecGlossMaterialCount > 0 ? "1" : "0");

        return defines;
    }

    Program::TypeConformanceList MaterialSystem::getTypeConformances() const
    {
        Program::TypeConformanceList typeConformances;
        for (const auto& it : mTypeConformances)
        {
            typeConformances.add(it.second);
        }
        return typeConformances;
    }

    Program::TypeConformanceList MaterialSystem::getTypeConformances(const MaterialType type) const
    {
        auto it = mTypeConformances.find(type);
        if (it == mTypeConformances.end())
        {
            throw RuntimeError(fmt::format("No type conformances for material type '{}'.", to_string(type)));
        }
        return it->second;
    }

    void MaterialSystem::createParameterBlock()
    {
        // Create parameter block.
        Program::DefineList defines = getDefines();
        defines.add("MATERIAL_SYSTEM_PARAMETER_BLOCK");
        auto pPass = ComputePass::create(kShaderFilename, "main", defines);
        auto pReflector = pPass->getProgram()->getReflector()->getParameterBlock("gMaterialsBlock");
        FALCOR_ASSERT(pReflector);

        mpMaterialsBlock = ParameterBlock::create(pReflector);
        FALCOR_ASSERT(mpMaterialsBlock);

        // Verify that the material data struct size on the GPU matches the host-side size.
        auto reflVar = mpMaterialsBlock->getReflection()->findMember(kMaterialDataName);
        FALCOR_ASSERT(reflVar);
        auto reflResType = reflVar->getType()->asResourceType();
        FALCOR_ASSERT(reflResType && reflResType->getType() == ReflectionResourceType::Type::StructuredBuffer);
        auto byteSize = reflResType->getStructType()->getByteSize();
        if (byteSize != sizeof(MaterialDataBlob))
        {
            throw RuntimeError("MaterialSystem material data buffer has unexpected struct size");
        }

        // Create materials data buffer.
        if (!mMaterials.empty() && (!mpMaterialDataBuffer || mpMaterialDataBuffer->getElementCount() < mMaterials.size()))
        {
            mpMaterialDataBuffer = Buffer::createStructured(mpMaterialsBlock[kMaterialDataName], (uint32_t)mMaterials.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mpMaterialDataBuffer->setName("MaterialSystem::mpMaterialDataBuffer");
        }

        // Bind resources to parameter block.
        mpMaterialsBlock[kMaterialDataName] = !mMaterials.empty() ? mpMaterialDataBuffer : nullptr;
        mpMaterialsBlock["materialCount"] = getMaterialCount();
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
