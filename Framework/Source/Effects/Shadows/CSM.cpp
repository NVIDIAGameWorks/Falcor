/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
#include "CSM.h"
#include "Graphics/Scene/SceneRenderer.h"
#include "glm/gtx/transform.hpp"
#include "Utils/Math/FalcorMath.h"
#include "Graphics/FboHelper.h"

namespace Falcor
{
    const char* kDepthPassVSFile = "Effects/ShadowPass.vs.slang";
    const char* kDepthPassGsFile = "Effects/ShadowPass.gs.slang";
    const char* kDepthPassFsFile = "Effects/ShadowPass.ps.slang";

    const Gui::DropdownList kFilterList = {
        { (uint32_t)CsmFilterPoint, "Point" },
        { (uint32_t)CsmFilterHwPcf, "2x2 HW PCF" },
        { (uint32_t)CsmFilterFixedPcf, "Fixed-Size PCF" },
        { (uint32_t)CsmFilterVsm, "VSM" },
        { (uint32_t)CsmFilterEvsm2, "EVSM2" },
        { (uint32_t)CsmFilterEvsm4, "EVSM4" },
        { (uint32_t)CsmFilterStochasticPcf, "Stochastic Poisson PCF" }
    };

    const Gui::DropdownList kPartitionList = {
        { (uint32_t)CascadedShadowMaps::PartitionMode::Linear, "Linear" },
        { (uint32_t)CascadedShadowMaps::PartitionMode::Logarithmic, "Logarithmic" },
        { (uint32_t)CascadedShadowMaps::PartitionMode::PSSM, "PSSM" }
    };

    const Gui::DropdownList kMaxAniso = {
        { (uint32_t)1, "1" },
        { (uint32_t)2, "2" },
        { (uint32_t)4, "4" },
        { (uint32_t)8, "8" },
        { (uint32_t)16, "16" }
    };

    class CsmSceneRenderer : public SceneRenderer
    {
    public:
        using UniquePtr = std::unique_ptr<CsmSceneRenderer>;
        static UniquePtr create(const Scene::SharedConstPtr& pScene, const ProgramReflection::BindLocation& alphaMapCbLoc, const ProgramReflection::BindLocation& alphaMapLoc, const ProgramReflection::BindLocation& alphaMapSamplerLoc)
        { 
            return UniquePtr(new CsmSceneRenderer(pScene, alphaMapCbLoc, alphaMapLoc, alphaMapSamplerLoc)); 
        }

        void setDepthClamp(bool enable) { mDepthClamp = enable; }

        void renderScene(RenderContext* pContext, Camera* pCamera) override
        {
            pContext->getGraphicsState()->setRasterizerState(nullptr);
            mpLastSetRs = nullptr;
            SceneRenderer::renderScene(pContext, pCamera);
        }

    protected:
        CsmSceneRenderer(const Scene::SharedConstPtr& pScene, const ProgramReflection::BindLocation& alphaMapCbLoc, const ProgramReflection::BindLocation& alphaMapLoc, const ProgramReflection::BindLocation& alphaMapSamplerLoc)
            : SceneRenderer(std::const_pointer_cast<Scene>(pScene))
        { 
            mBindLocations.alphaCB = alphaMapCbLoc;
            mBindLocations.alphaMap = alphaMapLoc;
            mBindLocations.alphaMapSampler = alphaMapSamplerLoc;

            setObjectCullState(false); 
            Sampler::Desc desc;
            desc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
            mpAlphaSampler = Sampler::create(desc);

            RasterizerState::Desc rsDesc;
            rsDesc.setDepthClamp(true);
            mpDepthClampRS = RasterizerState::create(rsDesc);
            rsDesc.setCullMode(RasterizerState::CullMode::None);
            mpDepthClampNoCullRS = RasterizerState::create(rsDesc);
            rsDesc.setDepthClamp(false);
            mpNoCullRS = RasterizerState::create(rsDesc);
        }

        bool mMaterialChanged = false;
        Sampler::SharedPtr mpAlphaSampler;

        struct
        {
            ProgramReflection::BindLocation alphaMap;
            ProgramReflection::BindLocation alphaCB;
            ProgramReflection::BindLocation alphaMapSampler;
        } mBindLocations;

        bool mDepthClamp;
        RasterizerState::SharedPtr mpDepthClampNoCullRS;
        RasterizerState::SharedPtr mpNoCullRS;
        RasterizerState::SharedPtr mpDepthClampRS;

        RasterizerState::SharedPtr mpLastSetRs;

        RasterizerState::SharedPtr getRasterizerState(const Material* pMaterial)
        {
            if (pMaterial->getAlphaMode() == AlphaModeMask)
            {
                return mDepthClamp ? mpDepthClampNoCullRS : mpNoCullRS;
            }
            else
            {
                return mDepthClamp ? mpDepthClampRS : nullptr;
            }
        }

        bool setPerMaterialData(const CurrentWorkingData& currentData, const Material* pMaterial) override
        {
            mMaterialChanged = true;
            if (currentData.pMaterial->getAlphaMode() == AlphaModeMask)
            {
                float alphaThreshold = currentData.pMaterial->getAlphaThreshold();
                auto& pDefaultBlock = currentData.pContext->getGraphicsVars()->getDefaultBlock();
                pDefaultBlock->getConstantBuffer(mBindLocations.alphaCB, 0)->setBlob(&alphaThreshold, 0u, sizeof(float));
                if(currentData.pMaterial->getDiffuseTexture())
                {
                    pDefaultBlock->setSrv(mBindLocations.alphaMap, 0, currentData.pMaterial->getDiffuseTexture()->getSRV());
                }
                else
                {
                    pDefaultBlock->setSrv(mBindLocations.alphaMap, 0, nullptr);
                }
                pDefaultBlock->setSampler(mBindLocations.alphaMapSampler, 0, mpAlphaSampler);
                currentData.pContext->getGraphicsState()->getProgram()->addDefine("TEST_ALPHA");
            }
            else
            {
                currentData.pContext->getGraphicsState()->getProgram()->removeDefine("TEST_ALPHA");
            }
            const auto& pRsState = getRasterizerState(currentData.pMaterial);
            if(pRsState != mpLastSetRs)
            {
                currentData.pContext->getGraphicsState()->setRasterizerState(pRsState);
                mpLastSetRs = pRsState;
            }
            return true;
        };
    };

    void createShadowMatrix(const DirectionalLight* pLight, const glm::vec3& center, float radius, glm::mat4& shadowVP)
    {
        glm::mat4 view = glm::lookAt(center, center + pLight->getWorldDirection(), glm::vec3(0, 1, 0));
        glm::mat4 proj = glm::ortho(-radius, radius, -radius, radius, -radius, radius);

        shadowVP = proj * view;
    }

    void createShadowMatrix(const PointLight* pLight, const glm::vec3& center, float radius, float fboAspectRatio, glm::mat4& shadowVP)
    {
        const glm::vec3 lightPos = pLight->getWorldPosition();
        const glm::vec3 lookat = pLight->getWorldDirection() + lightPos;
        glm::vec3 up(0, 1, 0);
        if(abs(glm::dot(up, pLight->getWorldDirection())) >= 0.95f)
        {
            up = glm::vec3(1, 0, 0);
        }
     
        glm::mat4 view = glm::lookAt(lightPos, lookat, up);
        float distFromCenter = glm::length(lightPos - center);
        float nearZ = max(0.1f, distFromCenter - radius);
        float maxZ = min(radius * 2, distFromCenter + radius);
        float angle = pLight->getOpeningAngle() * 2;
        glm::mat4 proj = glm::perspective(angle, fboAspectRatio, nearZ, maxZ);

        shadowVP = proj * view;
    }

    void createShadowMatrix(const Light* pLight, const glm::vec3& center, float radius, float fboAspectRatio, glm::mat4& shadowVP)
    {
        switch(pLight->getType())
        {
        case LightDirectional:
            return createShadowMatrix((DirectionalLight*)pLight, center, radius, shadowVP);
        case LightPoint:
            return createShadowMatrix((PointLight*)pLight, center, radius, fboAspectRatio, shadowVP);
        default:
            should_not_get_here();
        }
    }

    CascadedShadowMaps::~CascadedShadowMaps() = default;

    CascadedShadowMaps::CascadedShadowMaps(uint32_t mapWidth, uint32_t mapHeight, Light::SharedConstPtr pLight, Scene::SharedConstPtr pScene, uint32_t cascadeCount, ResourceFormat shadowMapFormat) : mpLight(pLight), mpScene(pScene)
    {
        if(mpLight->getType() != LightDirectional)
        {
            if (cascadeCount != 1)
            {
                logWarning("CSM with point-light only supports a single cascade (the user requested " + std::to_string(cascadeCount) + ")");
            }
            cascadeCount = 1;
        }
        mCsmData.cascadeCount = cascadeCount;
        GraphicsProgram::SharedPtr pProg = GraphicsProgram::createFromFile(kDepthPassVSFile, "");
        pProg->addDefine("_APPLY_PROJECTION");
        mDepthPass.pState = GraphicsState::create();
        mDepthPass.pState->setProgram(pProg);
        mDepthPass.pGraphicsVars = GraphicsVars::create(pProg->getActiveVersion()->getReflector());
        createShadowPassResources(mapWidth, mapHeight);

        mpLightCamera = Camera::create();

        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point).setAddressingMode(Sampler::AddressMode::Border, Sampler::AddressMode::Border, Sampler::AddressMode::Border).setBorderColor(glm::vec4(1.0f));
        samplerDesc.setLodParams(0.f, 0.f, 0.f);
        samplerDesc.setComparisonMode(Sampler::ComparisonMode::LessEqual);
        mShadowPass.pPointCmpSampler = Sampler::create(samplerDesc);
        samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
        mShadowPass.pLinearCmpSampler = Sampler::create(samplerDesc);
        samplerDesc.setComparisonMode(Sampler::ComparisonMode::Disabled);
        samplerDesc.setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
        samplerDesc.setLodParams(-100.f, 100.f, 0.f);

        createVsmSampleState(1);
        mpGaussianBlur = GaussianBlur::create();
        mpGaussianBlur->setSigma(2.5f);
        mpGaussianBlur->setKernelWidth(5);
    }

    CascadedShadowMaps::UniquePtr CascadedShadowMaps::create(uint32_t mapWidth, uint32_t mapHeight, Light::SharedConstPtr pLight, Scene::SharedConstPtr pScene, uint32_t cascadeCount, ResourceFormat shadowMapFormat)
    {
        if(isDepthFormat(shadowMapFormat) == false)
        {
            logError(std::string("Can't create CascadedShadowMaps effect. Requested resource format ") + to_string(shadowMapFormat) + " is not a depth format", true);
        }

        CascadedShadowMaps* pCsm = new CascadedShadowMaps(mapWidth, mapHeight, pLight, pScene, cascadeCount, shadowMapFormat);
        return CascadedShadowMaps::UniquePtr(pCsm);
    }

    void CascadedShadowMaps::setSdsmReadbackLatency(uint32_t latency)
    {
        if(mSdsmData.readbackLatency != latency)
        {
            mSdsmData.readbackLatency = latency;
            mSdsmData.minMaxReduction = nullptr;
        }
    }

    void CascadedShadowMaps::createSdsmData(Texture::SharedPtr pTexture)
    {
        assert(pTexture);
        // Only create a new technique if it doesn't exist or the dimensions changed
        if (mSdsmData.minMaxReduction)
        {
            if (mSdsmData.width == pTexture->getWidth() && mSdsmData.height == pTexture->getHeight())
            {
                return;
            }
        }
        mSdsmData.width = pTexture->getWidth();
        mSdsmData.height = pTexture->getHeight();
        mSdsmData.minMaxReduction = ParallelReduction::create(ParallelReduction::Type::MinMax, mSdsmData.readbackLatency, mSdsmData.width, mSdsmData.height);
    }

    void CascadedShadowMaps::createShadowPassResources(uint32_t mapWidth, uint32_t mapHeight)
    {
        mShadowPass.mapSize = glm::vec2(float(mapWidth), float(mapHeight));
        const ResourceFormat depthFormat = ResourceFormat::D32Float;
        mCsmData.depthBias = 0.005f;
        Program::DefineList progDef;
        progDef.add("TEST_ALPHA");
        progDef.add("_CASCADE_COUNT", std::to_string(mCsmData.cascadeCount));
        progDef.add("_ALPHA_CHANNEL", "a");
        ResourceFormat colorFormat = ResourceFormat::Unknown;
        switch(mCsmData.filterMode)
        {
        case CsmFilterVsm:
            colorFormat = ResourceFormat::RG32Float;
            progDef.add("_VSM");
            break;
        case CsmFilterEvsm2:
            colorFormat = ResourceFormat::RG32Float;
            progDef.add("_EVSM2");
            break;
        case CsmFilterEvsm4:
            colorFormat = ResourceFormat::RGBA32Float;
            progDef.add("_EVSM4");
            break;
        }

        Fbo::Desc fboDesc;
        fboDesc.setDepthStencilTarget(depthFormat);
        uint32_t mipLevels = 1;

        if(colorFormat != ResourceFormat::Unknown)
        {
            fboDesc.setColorTarget(0, colorFormat);
            mipLevels = Texture::kMaxPossible;
        }
        mShadowPass.pFbo = FboHelper::create2D(mapWidth, mapHeight, fboDesc, mCsmData.cascadeCount, mipLevels);
        mDepthPass.pState->setFbo(FboHelper::create2D(mapWidth, mapHeight, fboDesc, mCsmData.cascadeCount));

        mShadowPass.fboAspectRatio = (float)mapWidth / (float)mapHeight;

        // Create the shadows program
        GraphicsProgram::SharedPtr pProg = GraphicsProgram::createFromFile(
            kDepthPassVSFile,
            kDepthPassFsFile,
            kDepthPassGsFile,
            "", "", progDef);
        mShadowPass.pState = GraphicsState::create();
        mShadowPass.pState->setProgram(pProg);
        mShadowPass.pState->setDepthStencilState(nullptr);
        mShadowPass.pState->setFbo(mShadowPass.pFbo);
        const auto& pReflector = pProg->getActiveVersion()->getReflector();
        mShadowPass.pGraphicsVars = GraphicsVars::create(pReflector);

        const auto& pDefaultBlock = pReflector->getDefaultParameterBlock();
        auto alphaSampler = pDefaultBlock->getResourceBinding("alphaSampler");
        auto alphaMapCB = pDefaultBlock->getResourceBinding("AlphaMapCB");
        auto alphaMap = pDefaultBlock->getResourceBinding("alphaMap");
        mPerLightCbLoc = pDefaultBlock->getResourceBinding("PerLightCB");

        mpCsmSceneRenderer = CsmSceneRenderer::create(mpScene, alphaMapCB, alphaMap, alphaSampler);
        mpSceneRenderer = SceneRenderer::create(std::const_pointer_cast<Scene>(mpScene));
        mpSceneRenderer->setObjectCullState(true);
    }

    void CascadedShadowMaps::setCascadeCount(uint32_t cascadeCount)
    {
        if(mpLight->getType() != LightDirectional)
        {
            cascadeCount = 1;
        }
        mCsmData.cascadeCount = cascadeCount;
        createShadowPassResources(mShadowPass.pFbo->getWidth(), mShadowPass.pFbo->getHeight());
    }

    void CascadedShadowMaps::renderUi(Gui* pGui, const char* uiGroup)
    {
        if (!uiGroup || pGui->beginGroup(uiGroup))
        {
            //Filter mode
            uint32_t filterIndex = static_cast<uint32_t>(mCsmData.filterMode);
            if (pGui->addDropdown("Filter Mode", kFilterList, filterIndex))
            {
                setFilterMode(filterIndex);
            }

            //partition mode
            uint32_t newPartitionMode = static_cast<uint32_t>(mControls.partitionMode);
            if (pGui->addDropdown("Partition Mode", kPartitionList, newPartitionMode))
            {
                mControls.partitionMode = static_cast<PartitionMode>(newPartitionMode);
            }

            if (mControls.partitionMode == PartitionMode::PSSM)
            {
                pGui->addFloatVar("PSSM Lambda", mControls.pssmLambda, 0, 1.0f);
            }

            if (mControls.useMinMaxSdsm == false)
            {
                pGui->addFloatVar("Min Distance", mControls.distanceRange.x, 0, 1);
                pGui->addFloatVar("Max Distance", mControls.distanceRange.y, 0, 1);
            }

            pGui->addFloatVar("Cascade Blend Threshold", mCsmData.cascadeBlendThreshold, 0, 1.0f);
            pGui->addCheckBox("Depth Clamp", mControls.depthClamp);

            pGui->addFloatVar("Depth Bias", mCsmData.depthBias, 0, FLT_MAX, 0.0001f);
            pGui->addCheckBox("Stabilize Cascades", mControls.stabilizeCascades);

            // SDSM data
            const char* sdsmGroup = "SDSM MinMax";
            if (pGui->beginGroup(sdsmGroup))
            {
                pGui->addCheckBox("Enable", mControls.useMinMaxSdsm);
                if(mControls.useMinMaxSdsm)
                {
                    if (pGui->addIntVar("Readback Latency", mSdsmData.readbackLatency))
                    {
                        setSdsmReadbackLatency(mSdsmData.readbackLatency);
                    }
                    std::string range = "SDSM Range=[" + std::to_string(mSdsmData.sdsmResult.x) + ", " + std::to_string(mSdsmData.sdsmResult.y) + ']';
                    pGui->addText(range.c_str());
                }
                pGui->endGroup();
            }
            
            if (mCsmData.filterMode == CsmFilterFixedPcf || mCsmData.filterMode == CsmFilterStochasticPcf)
            {
                i32 kernelWidth = mCsmData.pcfKernelWidth;
                if (pGui->addIntVar("Kernel Width", kernelWidth, 1, 15, 2))
                {
                    setPcfKernelWidth(kernelWidth);
                }
            }

            //VSM/ESM
            if (mCsmData.filterMode == CsmFilterVsm || mCsmData.filterMode == CsmFilterEvsm2 || mCsmData.filterMode == CsmFilterEvsm4)
            {
                const char* vsmGroup = "VSM/EVSM";
                if (pGui->beginGroup(vsmGroup))
                {


                    uint32_t newMaxAniso = mShadowPass.pVSMTrilinearSampler->getMaxAnisotropy();
                    pGui->addDropdown("Max Aniso", kMaxAniso, newMaxAniso);
                    {
                        createVsmSampleState(newMaxAniso);
                    }

                    pGui->addFloatVar("Light Bleed Reduction", mCsmData.lightBleedingReduction, 0, 1.0f, 0.01f);

                    if(mCsmData.filterMode == CsmFilterEvsm2 || mCsmData.filterMode == CsmFilterEvsm4)
                    {
                        const char* evsmExpGroup = "EVSM Exp";
                        if (pGui->beginGroup(evsmExpGroup))
                        {
                            pGui->addFloatVar("Positive", mCsmData.evsmExponents.x, 0.0f, 42.0f, 0.01f);
                            pGui->addFloatVar("Negative", mCsmData.evsmExponents.y, 0.0f, 42.0f, 0.01f);
                            pGui->endGroup();
                        }
                    }

                    mpGaussianBlur->renderUI(pGui, "Blur");
                    pGui->endGroup();
                }
            }

            if(uiGroup) pGui->endGroup();
        }
    }

    void camClipSpaceToWorldSpace(const Camera* pCamera, glm::vec3 viewFrustum[8], glm::vec3& center, float& radius)
    {
        glm::vec3 clipSpace[8] =
        {
            glm::vec3(-1.0f, 1.0f, 0),
            glm::vec3(1.0f, 1.0f, 0),
            glm::vec3(1.0f, -1.0f, 0),
            glm::vec3(-1.0f, -1.0f, 0),
            glm::vec3(-1.0f, 1.0f, 1.0f),
            glm::vec3(1.0f, 1.0f, 1.0f),
            glm::vec3(1.0f, -1.0f, 1.0f),
            glm::vec3(-1.0f, -1.0f, 1.0f),
        };

        glm::mat4 invViewProj = pCamera->getInvViewProjMatrix();
        center = glm::vec3(0, 0, 0);

        for(uint32_t i = 0; i < 8; i++)
        {
            glm::vec4 crd = invViewProj * glm::vec4(clipSpace[i], 1);
            viewFrustum[i] = glm::vec3(crd) / crd.w;
            center += viewFrustum[i];
        }

        center *= (1.0f / 8.0f);

        // Calculate bounding sphere radius
        radius = 0;
        for(uint32_t i = 0; i < 8; i++)
        {
            float d = glm::length(center - viewFrustum[i]);
            radius = max(d, radius);
        }
    }

    forceinline float calcPssmPartitionEnd(float nearPlane, float camDepthRange, const glm::vec2& distanceRange, float linearBlend, uint32_t cascade, uint32_t cascadeCount)
    {
        // Convert to camera space
        float minDepth = nearPlane + distanceRange.x * camDepthRange;
        float maxDepth = nearPlane + distanceRange.y * camDepthRange;

        float depthRange = maxDepth - minDepth;
        float depthScale = maxDepth / minDepth;

        float cascadeScale = float(cascade + 1) / float(cascadeCount);
        float logSplit = pow(depthScale, cascadeScale) * minDepth;
        float uniSplit = minDepth + depthRange * cascadeScale;

        float distance = linearBlend * logSplit + (1 - linearBlend) * uniSplit;

        // Convert back to clip-space
        distance = (distance - nearPlane) / camDepthRange;
        return distance;
    }

    void getCascadeCropParams(const glm::vec3 crd[8], const glm::mat4& lightVP, glm::vec4& scale, glm::vec4& offset)
    {
        // Transform the frustum into light clip-space and calculate min-max
        glm::vec4 maxCS(-1, -1, 0, 1);
        glm::vec4 minCS(1, 1, 1, 1);
        for(uint32_t i = 0; i < 8; i++)
        {
            glm::vec4 c = lightVP * glm::vec4(crd[i], 1.0f);
            c /= c.w;
            maxCS = max(maxCS, c);
            minCS = min(minCS, c);
        }

        glm::vec4 delta = maxCS - minCS;
        scale = glm::vec4(2, 2, 1, 1) / delta;

        offset.x = -0.5f * (maxCS.x + minCS.x) * scale.x;
        offset.y = -0.5f * (maxCS.y + minCS.y) * scale.y;
        offset.z = -minCS.z * scale.z;

        scale.w = 1;
        offset.w = 0;
    }

    void CascadedShadowMaps::partitionCascades(const Camera* pCamera, const glm::vec2& distanceRange)
    {
        struct
        {
            glm::vec3 crd[8];
            glm::vec3 center;
            float radius;
        } camFrustum;

        camClipSpaceToWorldSpace(pCamera, camFrustum.crd, camFrustum.center, camFrustum.radius);

        // Create the global shadow space
        createShadowMatrix(mpLight.get(), camFrustum.center, camFrustum.radius, mShadowPass.fboAspectRatio, mCsmData.globalMat);

        if(mCsmData.cascadeCount == 1)
        {
            mCsmData.cascadeScale[0] = glm::vec4(1);
            mCsmData.cascadeOffset[0] = glm::vec4(0);
            mCsmData.cascadeRange[0].x = 0;
            mCsmData.cascadeRange[0].y = 1;
            return;
        }

        float nearPlane = pCamera->getNearPlane();
        float farPlane = pCamera->getFarPlane();
        float depthRange = farPlane - nearPlane;

        float nextCascadeStart = distanceRange.x;

        for(int32_t c = 0; c < mCsmData.cascadeCount; c++)
        {
            float cascadeStart = nextCascadeStart;

            switch(mControls.partitionMode)
            {
            case PartitionMode::Linear:
                nextCascadeStart = cascadeStart + (distanceRange.y - distanceRange.x) / float(mCsmData.cascadeCount);
                break;
            case PartitionMode::Logarithmic:
                nextCascadeStart = calcPssmPartitionEnd(nearPlane, depthRange, distanceRange, 1.0f, c, mCsmData.cascadeCount);
                break;
            case PartitionMode::PSSM:
                nextCascadeStart = calcPssmPartitionEnd(nearPlane, depthRange, distanceRange, mControls.pssmLambda, c, mCsmData.cascadeCount);
                break;
            default:
                should_not_get_here();
            }

            // If we blend between cascades, we need to expand the range to make sure we will not try to read of the edge of the shadow-map
            float blendCorrection = (nextCascadeStart - cascadeStart) *  (mCsmData.cascadeBlendThreshold * 0.5f);
            float cascadeEnd = nextCascadeStart + blendCorrection;
            nextCascadeStart -= blendCorrection;

            // Calculate the cascade distance in camera-clip space(Where the clip-space range is [0, farPlane])
            mCsmData.cascadeRange[c].x = farPlane * cascadeStart;
            mCsmData.cascadeRange[c].y = (farPlane * cascadeEnd) - mCsmData.cascadeRange[c].x;

            // Calculate the cascade frustum
            glm::vec3 cascadeFrust[8];
            for(uint32_t i = 0; i < 4; i++)
            {
                glm::vec3 edge = camFrustum.crd[i + 4] - camFrustum.crd[i];
                glm::vec3 start = edge * cascadeStart;
                glm::vec3 end = edge * cascadeEnd;
                cascadeFrust[i] = camFrustum.crd[i] + start;
                cascadeFrust[i + 4] = camFrustum.crd[i] + end;
            }

            getCascadeCropParams(cascadeFrust, mCsmData.globalMat, mCsmData.cascadeScale[c], mCsmData.cascadeOffset[c]);
        }
    }

    static bool checkOffset(size_t cbOffset, size_t cppOffset, const char* field)
    {
        if (cbOffset != cppOffset)
        {
            logError("CsmData::" + std::string(field) + " CB offset mismatch. CB offset is " + std::to_string(cbOffset) + ", C++ data offset is " + std::to_string(cppOffset));
            return false;
        }
        return true;
    }

#if _LOG_ENABLED
#define check_offset(_a) {static bool b = true; if(b) {assert(checkOffset(pCB->getVariableOffset("gCsmData." #_a), offsetof(CsmData, _a), #_a));} b = false;}
#else
#define check_offset(_a)
#endif

    void CascadedShadowMaps::renderScene(RenderContext* pCtx)
    {
        ConstantBuffer* pCB = mShadowPass.pGraphicsVars->getDefaultBlock()->getConstantBuffer(mPerLightCbLoc, 0).get();
        check_offset(globalMat);
        check_offset(cascadeScale[0]);
        check_offset(cascadeOffset[0]);
        check_offset(cascadeRange[0]);
        check_offset(depthBias);
        check_offset(cascadeCount);
        check_offset(filterMode);
        check_offset(pcfKernelWidth);
        check_offset(lightDir);
        check_offset(lightBleedingReduction);
        check_offset(evsmExponents);
        check_offset(cascadeBlendThreshold);


        pCB->setBlob(&mCsmData, 0, sizeof(mCsmData));
        pCtx->pushGraphicsVars(mShadowPass.pGraphicsVars);
        pCtx->pushGraphicsState(mShadowPass.pState);
        mpCsmSceneRenderer->renderScene(pCtx, mpLightCamera.get());
        pCtx->popGraphicsState();
        pCtx->popGraphicsVars();
    }

    void CascadedShadowMaps::executeDepthPass(RenderContext* pCtx, const Camera* pCamera)
    {
        // Must have an FBO attached, otherwise don't know the size of the depth map
		const auto& pStateFbo = pCtx->getGraphicsState()->getFbo();
		uint32_t width, height;
		if(pStateFbo)
		{
			width = pStateFbo->getWidth();
			height = pStateFbo->getHeight();
		}
		else
		{
			width = (uint32_t)mShadowPass.mapSize.x;
			height = (uint32_t)mShadowPass.mapSize.y;
		}

        Fbo::SharedConstPtr pFbo = mDepthPass.pState->getFbo();
        if((pFbo == nullptr) || (pFbo->getWidth() != width) || (pFbo->getHeight() != height))
        {
            Fbo::Desc desc;
            desc.setDepthStencilTarget(mShadowPass.pFbo->getDepthStencilTexture()->getFormat());
            mDepthPass.pState->setFbo(FboHelper::create2D(width, height, desc));
        }

        pCtx->clearFbo(pFbo.get(), glm::vec4(), 1, 0, FboAttachmentType::Depth);
        pCtx->pushGraphicsState(mDepthPass.pState);
        pCtx->pushGraphicsVars(mDepthPass.pGraphicsVars);
        mpSceneRenderer->renderScene(pCtx, const_cast<Camera*>(pCamera));
        pCtx->popGraphicsVars();
        pCtx->popGraphicsState();
    }

    void CascadedShadowMaps::createVsmSampleState(uint32_t maxAnisotropy)
    {
        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
        samplerDesc.setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
        samplerDesc.setMaxAnisotropy(maxAnisotropy);
        mShadowPass.pVSMTrilinearSampler = Sampler::create(samplerDesc);
    }

    void CascadedShadowMaps::reduceDepthSdsmMinMax(RenderContext* pRenderCtx, const Camera* pCamera, Texture::SharedPtr pDepthBuffer)
    {
        if(pDepthBuffer == nullptr)
        {
            // Run a shadow pass
            executeDepthPass(pRenderCtx, pCamera);
            pDepthBuffer = mDepthPass.pState->getFbo()->getDepthStencilTexture();
        }

        createSdsmData(pDepthBuffer);
        vec2 distanceRange = glm::vec2(mSdsmData.minMaxReduction->reduce(pRenderCtx, pDepthBuffer));

        // Convert to linear
        glm::mat4 camProj = pCamera->getProjMatrix();
        distanceRange = camProj[2][2] - distanceRange*camProj[2][3];
        distanceRange = camProj[3][2] / distanceRange;
        distanceRange = (distanceRange - pCamera->getNearPlane()) / (pCamera->getFarPlane() - pCamera->getNearPlane());
        distanceRange = glm::clamp(distanceRange, glm::vec2(0), glm::vec2(1));
        mSdsmData.sdsmResult = distanceRange;

        if (mControls.stabilizeCascades)
        {
            // Ignore minor changes that can result in swimming
            distanceRange = round(distanceRange * 16.0f) / 16.0f;
            distanceRange.y = max(distanceRange.y, 0.005f);
        }
    }

    vec2 CascadedShadowMaps::calcDistanceRange(RenderContext* pRenderCtx, const Camera* pCamera, Texture::SharedPtr pDepthBuffer)
    {
        if(mControls.useMinMaxSdsm)
        {
            reduceDepthSdsmMinMax(pRenderCtx, pCamera, pDepthBuffer);
            return mSdsmData.sdsmResult;
        }
        else
        {
            return mControls.distanceRange;
        }
    }

    void CascadedShadowMaps::setup(RenderContext* pRenderCtx, const Camera* pCamera, Texture::SharedPtr pDepthBuffer)
    {
        const glm::vec4 clearColor(0);
        pRenderCtx->clearFbo(mShadowPass.pFbo.get(), clearColor, 1, 0, FboAttachmentType::All);

        // Calc the bounds
        glm::vec2 distanceRange = calcDistanceRange(pRenderCtx, pCamera, pDepthBuffer);

        GraphicsState::Viewport VP;
        VP.originX = 0;
        VP.originY = 0;
        VP.minDepth = 0;
        VP.maxDepth = 1;
        VP.height = mShadowPass.mapSize.x;
        VP.width = mShadowPass.mapSize.y;

        //Set shadow pass state
        mShadowPass.pState->setViewport(0, VP);
        mpCsmSceneRenderer->setDepthClamp(mControls.depthClamp);
        pRenderCtx->pushGraphicsState(mShadowPass.pState);
        partitionCascades(pCamera, distanceRange);
        renderScene(pRenderCtx);

        if(mCsmData.filterMode == CsmFilterVsm || mCsmData.filterMode == CsmFilterEvsm2 || mCsmData.filterMode == CsmFilterEvsm4)
        {
            mpGaussianBlur->execute(pRenderCtx, mShadowPass.pFbo->getColorTexture(0), mShadowPass.pFbo);
            mShadowPass.pFbo->getColorTexture(0)->generateMips(pRenderCtx);
        }

        pRenderCtx->popGraphicsState();
    }

    void CascadedShadowMaps::setDataIntoGraphicsVars(GraphicsVars::SharedPtr pVars, const std::string& varName)
    {
        switch (mCsmData.filterMode)
        {
        case CsmFilterPoint:
            pVars->setTexture(varName + ".shadowMap", mShadowPass.pFbo->getDepthStencilTexture());
            pVars->setSampler("gCsmCompareSampler", mShadowPass.pPointCmpSampler);
            break;
        case CsmFilterHwPcf:
        case CsmFilterFixedPcf:
        case CsmFilterStochasticPcf:
            pVars->setTexture(varName + ".shadowMap", mShadowPass.pFbo->getDepthStencilTexture());
            pVars->setSampler("gCsmCompareSampler", mShadowPass.pLinearCmpSampler);
            break;
        case CsmFilterVsm:
        case CsmFilterEvsm2:
        case CsmFilterEvsm4:
            pVars->setTexture(varName + ".shadowMap", mShadowPass.pFbo->getColorTexture(0));
            pVars->setSampler(varName + ".csmSampler", mShadowPass.pVSMTrilinearSampler);
            break;
        }    

        mCsmData.lightDir = glm::normalize(((DirectionalLight*)mpLight.get())->getWorldDirection());
        ConstantBuffer::SharedPtr pCB = pVars->getConstantBuffer("PerFrameCB");
        size_t offset = pCB->getVariableOffset(varName);
        pCB->setBlob(&mCsmData, offset, sizeof(mCsmData));
    }
    
    Texture::SharedPtr CascadedShadowMaps::getShadowMap() const
    {
        switch(mCsmData.filterMode)
        {
        case CsmFilterVsm:
        case CsmFilterEvsm2:
        case CsmFilterEvsm4:
            return mShadowPass.pFbo->getColorTexture(0);
        }
        return mShadowPass.pFbo->getDepthStencilTexture();
    }

    void CascadedShadowMaps::setFilterMode(uint32_t newFilterMode)
    {
        mCsmData.filterMode = newFilterMode;
        createShadowPassResources(mShadowPass.pFbo->getWidth(), mShadowPass.pFbo->getHeight());
    }

    void CascadedShadowMaps::setEvsmBlur(uint32_t kernelWidth, float sigma)
    {
        mpGaussianBlur->setKernelWidth(kernelWidth);
        mpGaussianBlur->setSigma(sigma);
    }
}
