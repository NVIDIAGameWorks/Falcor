/***************************************************************************
# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/
#include "Framework.h"
#include "LeanMap.h"
#include "Graphics/Material/Material.h"
#include "Graphics/Scene/Scene.h"
#include "API/Device.h"

namespace Falcor
{
    Texture::SharedPtr LeanMap::createFromNormalMap(const Falcor::Texture* pNormalMap)
    {
        uint32_t texW = pNormalMap->getWidth();
        uint32_t texH = pNormalMap->getHeight();

        std::vector<vec4> leanData;

        leanData.resize(texW * texH);

        auto normalMapData = gpDevice->getRenderContext()->readTextureSubresource(pNormalMap, 0);

        const float oneBy255 = 1.0f / 255.0f;
        for(auto y = 0u; y < texH; y++)
        {
            for(auto x = 0u; x < texW; x++)
            {
                auto texIdx = (x + y * texW);
                vec3 tn;

                switch(pNormalMap->getFormat())
                {
                case ResourceFormat::RGBA8Unorm:
                {
                    tn.x = clamp(oneBy255 * (float)(normalMapData[texIdx * 4 + 0]), 0.0f, 1.0f);
                    tn.y = clamp(oneBy255 * (float)(normalMapData[texIdx * 4 + 1]), 0.0f, 1.0f);
                    tn.z = clamp(oneBy255 * (float)(normalMapData[texIdx * 4 + 2]), 0.0f, 1.0f);
                } break;
                case ResourceFormat::BGRA8Unorm:
                case ResourceFormat::BGRX8Unorm:
                {
                    tn.z = clamp(oneBy255 * (float)(normalMapData[texIdx * 4 + 0]), 0.0f, 1.0f);
                    tn.y = clamp(oneBy255 * (float)(normalMapData[texIdx * 4 + 1]), 0.0f, 1.0f);
                    tn.x = clamp(oneBy255 * (float)(normalMapData[texIdx * 4 + 2]), 0.0f, 1.0f);
                } break;
                case ResourceFormat::RGBA8UnormSrgb:
                {
                    tn.x = clamp(sRGBToLinear(oneBy255 * (float)(normalMapData[texIdx * 4 + 0])), 0.0f, 1.0f);
                    tn.y = clamp(sRGBToLinear(oneBy255 * (float)(normalMapData[texIdx * 4 + 1])), 0.0f, 1.0f);
                    tn.z = clamp(sRGBToLinear(oneBy255 * (float)(normalMapData[texIdx * 4 + 2])), 0.0f, 1.0f);
                } break;
                case ResourceFormat::BGRA8UnormSrgb:
                {
                    tn.z = clamp(sRGBToLinear(oneBy255 * (float)(normalMapData[texIdx * 4 + 0])), 0.0f, 1.0f);
                    tn.y = clamp(sRGBToLinear(oneBy255 * (float)(normalMapData[texIdx * 4 + 1])), 0.0f, 1.0f);
                    tn.x = clamp(sRGBToLinear(oneBy255 * (float)(normalMapData[texIdx * 4 + 2])), 0.0f, 1.0f);
                } break;
                default: 
                    logError("Can't generate LEAN map. Unsupported normal map format.");
                    return nullptr;
                };

                // Unpack
                static const float epsilon = 1e-3f;
                vec3 n = tn * 2.f - vec3(1.f);
                // And normalize the normal
                n.z = max(n.z, epsilon);
                n = normalize(n);

                // Write out the first moment (mean) in slope space
                vec2 b = vec2(n.x, n.y) / max(n.z, epsilon);
                vec2 m = b*b;
                leanData[texIdx] = vec4(b.x*0.5f + 0.5f, b.y*0.5f + 0.5f, m.x, m.y);
            }
        }

        Texture::SharedPtr pTex = Texture::create2D(texW, texH, ResourceFormat::RGBA32Float, 1, Texture::kMaxPossible, leanData.data());
        return pTex;
    }

    bool LeanMap::createLeanMap(const Material* pMaterial)
    {
        uint32_t materialID = pMaterial->getId();

        if(mpLeanMaps.find(materialID) != mpLeanMaps.end())
        {
            logError("Error when creating SceneLeanMaps. Scene material IDs should be unique for scene materials.");
            return false;
        }

        const Texture* pNormalMap = pMaterial->getNormalMap().get();
        if(pNormalMap)
        {
            mpLeanMaps[materialID] = createFromNormalMap(pNormalMap);
            mShaderArraySize = max(materialID + 1, mShaderArraySize);
        }
        return true;
    }

    LeanMap::UniquePtr LeanMap::create(const Scene* pScene)
    {
        UniquePtr pLeanMaps = UniquePtr(new LeanMap);

        // Initialize model materials
        for(uint32_t model = 0; model < pScene->getModelCount(); model++)
        {
            const Model* pModel = pScene->getModel(model).get();
            for(uint32_t meshID = 0; meshID < pModel->getMeshCount(); meshID++)
            {
                const Material* pMaterial = pModel->getMesh(meshID)->getMaterial().get();
                if(pLeanMaps->createLeanMap(pMaterial) == false)
                {
                    return nullptr;
                }
            }
        }

        if(pLeanMaps->mpLeanMaps.size() == 0)
        {
            logWarning("Trying to create SceneLeanMaps for a scene without materials.");
        }

        return pLeanMaps;
    }

    void LeanMap::setIntoProgramVars(ProgramVars* pVars, const Sampler::SharedPtr& pSampler) const
    {
        for (const auto& pMap : mpLeanMaps)
        {
            std::string name("gLeanMaps");
            if(mpLeanMaps.size() > 1)
            {
                uint32_t id = pMap.first;
                name += "[" + std::to_string(id) + "]";
            }
            Texture::SharedPtr pTex = pMap.second;
            pVars->setTexture(name, pTex);
        }

        pVars->setSampler("gLeanMapSampler", pSampler);
    }
}