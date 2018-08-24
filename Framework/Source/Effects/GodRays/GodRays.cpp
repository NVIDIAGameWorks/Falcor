/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "GodRays.h"
#include "Graphics/RenderGraph/RenderPassSerializer.h"
#include "Graphics/Light.h"
#include "API/RenderContext.h"
#include "Utils/Gui.h"

namespace Falcor
{
    static const char* kInputName = "color";
    static const char* kInputDepthName = "depth";
    static const char* kDstColorName = "dst";

    GodRays::UniquePtr GodRays::create(float mediumDensity, float mediumDecay, float mediumWeight, float exposer, int32_t numSamples)
    {
        return GodRays::UniquePtr(new GodRays(mediumDensity, mediumDecay, mediumWeight, exposer, numSamples));
    }

    GodRays::GodRays(float mediumDensity, float mediumDecay, float mediumWeight, float exposer, int32_t numSamples)
        : RenderPass("GodRays"), mMediumDensity(mediumDensity), mMediumDecay(mediumDecay), mMediumWeight(mediumWeight), mNumSamples(numSamples), mExposer(exposer)
    {
        BlendState::Desc desc;
        desc.setRtBlend(0, true);
        desc.setRtParams(0,
            BlendState::BlendOp::Add, BlendState::BlendOp::Add,
            BlendState::BlendFunc::One, BlendState::BlendFunc::One,
            BlendState::BlendFunc::SrcAlpha, BlendState::BlendFunc::OneMinusSrcAlpha);

        mpAdditiveBlend = BlendState::create(desc);

        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point);
        samplerDesc.setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
        mpSampler = Sampler::create(samplerDesc);

        mpLightPassFbo = Fbo::create();
    }

    void GodRays::reflect(RenderPassReflection& reflector) const
    {
        reflector.addInput(kInputName);
        reflector.addInput(kInputDepthName);
        reflector.addOutput(kDstColorName);
    }

    GodRays::UniquePtr GodRays::deserialize(const RenderPassSerializer& serializer)
    { 
        Scene::UserVariable thresholdVar = serializer.getValue("godRays.threshold");
        if (thresholdVar.type == Scene::UserVariable::Type::Unknown)
        {
            return create();
        }

        float threshold = static_cast<float>(thresholdVar.d64);
        float mediumDensity = static_cast<float>(serializer.getValue("godRays.mediumDensity").d64);
        float mediumDecay = static_cast<float>(serializer.getValue("godRays.mediumDecay").d64);
        float mediumWeight = static_cast<float>(serializer.getValue("godRays.mediumWeight").d64);
        float exposer = static_cast<float>(serializer.getValue("godRays.exposer").d64);
        int32_t numSamples = serializer.getValue("godRays.numSamples").i32;

        return create(mediumDensity, mediumDecay, mediumWeight, exposer, numSamples); 
    }

    void GodRays::serialize(RenderPassSerializer& renderPassSerializer)
    {
        renderPassSerializer.addVariable("godRays.mediumDensity", mMediumDensity);
        renderPassSerializer.addVariable("godRays.mediumDecay", mMediumDecay);
        renderPassSerializer.addVariable("godRays.mediumWeight", mMediumWeight);
        renderPassSerializer.addVariable("godRays.exposer", mExposer);
        renderPassSerializer.addVariable("godRays.numSamples", mNumSamples);
    }

    void GodRays::createShader()
    {
        Program::DefineList defines;
        defines.add("_NUM_SAMPLES", std::to_string(mNumSamples));

        if (mpBlitPass)
        {
            mpBlitPass->getProgram()->addDefines(defines);
        }
        else
        {
            mpBlitPass = FullScreenPass::create("Framework/Shaders/Blit.vs.slang", "Effects/GodRays.ps.slang", defines);

            mpVars = GraphicsVars::create(mpBlitPass->getProgram()->getReflector());
            mSrcTexLoc = mpBlitPass->getProgram()->getReflector()->getDefaultParameterBlock()->getResourceBinding("srcColor");
        }

        if (!mpLightPass)
        {
            mpLightPass = FullScreenPass::create("Effects/GodRaysLightPass.ps.slang");

            mpLightPassVars = GraphicsVars::create(mpLightPass->getProgram()->getReflector());
            mSrcDepthLoc = mpLightPass->getProgram()->getReflector()->getDefaultParameterBlock()->getResourceBinding("srcDepth");
        }

        mpVars["SrcRectCB"]["gOffset"] = vec2(0.0f);
        mpVars["SrcRectCB"]["gScale"] = vec2(1.0f);

        mLightVarOffset = mpLightPassVars["GodRaySettings"]->getVariableOffset("light");
    }

    void GodRays::updateLowResTexture(const Texture::SharedPtr& pTexture)
    {
        // Create FBO if not created already, or properties have changed since last use
        bool createFbo = mpLowResTexture == nullptr;

        float aspectRatio = (float)pTexture->getWidth() / (float)pTexture->getHeight();
        uint32_t lowResHeight = max(pTexture->getHeight() / 2, 256u);
        uint32_t lowResWidth = max(pTexture->getWidth() / 2, (uint32_t)(256.0f * aspectRatio));

        if (createFbo == false)
        {
            createFbo = (lowResWidth != mpLowResTexture->getWidth()) ||
                (lowResHeight != mpLowResTexture->getHeight()) ||
                (pTexture->getFormat() != mpLowResTexture->getFormat());
        }

        if (createFbo)
        {
            mpLowResTexture = Texture::create2D(
                lowResWidth,
                lowResHeight,
                pTexture->getFormat(),
                1,
                1,
                nullptr,
                Resource::BindFlags::ShaderResource | Resource::BindFlags::RenderTarget);
        }
    }

    void GodRays::execute(RenderContext* pRenderContext, const RenderData* pRenderData)
    {
        if (!mpTargetFbo) mpTargetFbo = Fbo::create();

        pRenderContext->blit(pRenderData->getTexture(kInputName)->getSRV(), pRenderData->getTexture(kDstColorName)->getRTV());
        mpTargetFbo->attachColorTarget(pRenderData->getTexture(kDstColorName), 0);

        execute(pRenderContext, pRenderData->getTexture(kInputName), pRenderData->getTexture(kInputDepthName), mpTargetFbo);
    }

    void GodRays::execute(RenderContext* pRenderContext, Fbo::SharedPtr pFbo)
    {
        execute(pRenderContext, pFbo->getColorTexture(0), pFbo->getDepthStencilTexture(), pFbo);
    }

    void GodRays::execute(RenderContext* pRenderContext, const Texture::SharedPtr& pSrcTex, const Texture::SharedPtr& pSrcDepthTex, Fbo::SharedPtr pFbo)
    {
        assert(pFbo->getWidth() == pSrcTex->getWidth() && pSrcTex->getHeight() == pFbo->getHeight());

        if (mDirty)
        {
            createShader();
            mDirty = false;
        }
        
        Camera::SharedPtr pActiveCamera =  mpScene->getActiveCamera();

        mpVars["GodRaySettings"]["gMedia.density"] = mMediumDensity;
        mpVars["GodRaySettings"]["gMedia.decay"] = mMediumDecay;
        mpVars["GodRaySettings"]["gMedia.weight"] = mMediumWeight;
        mpVars["GodRaySettings"]["cameraRatio"] = pActiveCamera->getAspectRatio();
        mpVars["GodRaySettings"]["exposer"] = mExposer;

        mpLightPassVars["GodRaySettings"]["exposer"] = mExposer;
        mpLightPassVars["GodRaySettings"]["cameraRatio"] = pActiveCamera->getAspectRatio();
        mpLightPassVars["GodRaySettings"]["viewportHeight"] = static_cast<float>(pFbo->getColorTexture(0)->getHeight());

        updateLowResTexture(pSrcTex);
        
        glm::vec3 screenSpaceLightPosition;
        Light::SharedPtr pLight = mpScene->getLight(mLightIndex);

        pLight->setIntoProgramVars(mpVars.get(), mpVars["GodRaySettings"].get(), mLightVarOffset);
        pLight->setIntoProgramVars(mpLightPassVars.get(), mpLightPassVars["GodRaySettings"].get(), mLightVarOffset);

        // project directional lights
        if (mpScene->getLight(mLightIndex)->getType() == 1)
        {
            glm::vec3 lightPoint = pActiveCamera->getPosition() - pLight->getData().dirW;
            glm::vec4 lightDirViewSpace = pActiveCamera->getViewProjMatrix() * glm::vec4(lightPoint, 1.0f);
            screenSpaceLightPosition = lightDirViewSpace / lightDirViewSpace.w;
            screenSpaceLightPosition.z = 1.0f;

            // avoid rays coming from the other side of the projected sphere
            if (glm::dot(-pLight->getData().dirW, glm::float3(pActiveCamera->getViewProjMatrix()[2])) < 0.0f)
            {
                // dont draw highlight
                mpLightPassVars["GodRaySettings"]["exposer"] = 0.0f;
            }
        }
        else
        {
            glm::vec4 ssLightPosition = pActiveCamera->getViewProjMatrix() * glm::vec4(pLight->getData().posW, 1.0f);
            screenSpaceLightPosition = ssLightPosition / ssLightPosition.w;
        }

        screenSpaceLightPosition.x = 0.5f * (screenSpaceLightPosition.x + 0.5f);
#ifdef FALCOR_VK
        screenSpaceLightPosition.y *= -1.0f;
#endif
        
        screenSpaceLightPosition.y = 0.5f * (-screenSpaceLightPosition.y + 0.5f);
        mpVars["GodRaySettings"]["screenSpaceLightPosition"] = screenSpaceLightPosition;
        mpLightPassVars["GodRaySettings"]["screenSpaceLightPosition"] = screenSpaceLightPosition;

        mpLightPassVars->getDefaultBlock()->setSrv(mSrcDepthLoc, 0, pSrcDepthTex->getSRV());
        mpLightPassVars->setSampler("gSampler", mpSampler);
        
        GraphicsState::SharedPtr pState = pRenderContext->getGraphicsState();
        
        // generate light texture
        pState->pushFbo(mpLightPassFbo);
        pRenderContext->pushGraphicsVars(mpLightPassVars);
        mpLightPassFbo->attachColorTarget(mpLowResTexture, 0);
        mpLightPass->execute(pRenderContext);
        pRenderContext->popGraphicsVars();
        pState->popFbo();

        if (mOutputIndex == 1)
        {
            pRenderContext->blit(mpLowResTexture->getSRV(), pFbo->getColorTexture(0)->getRTV());
        }
        else
        {
            mpVars->setSampler("gSampler", mpSampler);
            pState->pushFbo(pFbo);
            pRenderContext->pushGraphicsVars(mpVars);
            mpBlitPass->execute(pRenderContext, nullptr, (mOutputIndex == 2) ? nullptr : mpAdditiveBlend);
            mpVars->getDefaultBlock()->setSrv(mSrcTexLoc, 0, mpLowResTexture->getSRV());
            pRenderContext->popGraphicsVars();
            pState->popFbo();
        }
    }

    void GodRays::setNumSamples(int32_t numSamples)
    {
        mNumSamples = numSamples;
        mDirty = true;
    }

    void GodRays::renderUI(Gui* pGui, const char* uiGroup)
    {
        if (uiGroup == nullptr || pGui->beginGroup(uiGroup))
        {
            if (pGui->addFloatVar("Medium Density", mMediumDensity))
            {
                mpVars["GodRaySettings"]["gMedia.density"] = mMediumDensity;
            }
            if (pGui->addFloatVar("Medium Decay", mMediumDecay))
            {
                mpVars["GodRaySettings"]["gMedia.decay"] = mMediumDecay;
            }
            if (pGui->addFloatVar("Medium Weight", mMediumWeight))
            {
                mpVars["GodRaySettings"]["gMedia.weight"] = mMediumWeight;
            }
            if (pGui->addIntVar("Num Samples", mNumSamples, 0, 1000))
            {
                setNumSamples(mNumSamples);
            }
            if (mpScene->getLightCount())
            {
                Gui::DropdownList lightList;
                for (uint32_t i = 0; i < mpScene->getLightCount(); i++)
                {
                    Gui::DropdownValue value;
                    value.label = mpScene->getLight(i)->getName();
                    value.value = i;
                    lightList.push_back(value);
                }

                uint32_t lightIndex = mLightIndex;
                if (pGui->addDropdown("Source Light", lightList, lightIndex))
                {
                    mpVars["GodRaySettings"]["lightIndex"] = (mLightIndex = lightIndex);
                }
            }
            else
            {
                if (pGui->addIntVar("Light Index", mLightIndex, 0, 15))
                {
                    mpVars["GodRaySettings"]["lightIndex"] = mLightIndex;
                }
            }
            if (pGui->addFloatVar("Exposer", mExposer, 0.0f))
            {
                mpVars["GodRaySettings"]["exposer"] = mExposer;
            }

            Gui::DropdownList lightList;
            Gui::DropdownValue value;
            value.label = "Final GodRays";
            value.value = 0;
            lightList.push_back(value);
            value.label = "Light Texture";
            value.value = 1;
            lightList.push_back(value);
            value.label = "Blurred Light Texture";
            value.value = 2;
            lightList.push_back(value);

            pGui->addDropdown("output", lightList, mOutputIndex);

            if(uiGroup) pGui->endGroup();
        }
    }
}
