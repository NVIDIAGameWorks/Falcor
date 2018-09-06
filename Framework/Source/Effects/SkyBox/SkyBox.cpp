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
#include "SkyBox.h"
#include "glm/gtx/transform.hpp"
#include "Graphics/TextureHelper.h"
#include "Graphics/Camera/Camera.h"
#include "Graphics/Model/ModelRenderer.h"
#include "Graphics/Scene/Scene.h"

namespace Falcor
{
    SkyBox::SkyBox() : RenderPass("SkyBox") {}

    SkyBox::SharedPtr SkyBox::create(Texture::SharedPtr& pSkyTexture, Sampler::SharedPtr pSampler, bool renderStereo)
    {
        SharedPtr pSkyBox = SharedPtr(new SkyBox());
        if(pSkyBox->createResources(pSkyTexture, pSampler, renderStereo) == false)
        {
            return nullptr;
        }
        return pSkyBox;
    }

    SkyBox::SharedPtr SkyBox::create(bool loadAsSrgb, Sampler::SharedPtr pSampler, bool renderStereo)
    {
        SharedPtr pSkyBox = SharedPtr(new SkyBox());
        pSkyBox->mRenderStereo = renderStereo;
        pSkyBox->mLoadSrgb = loadAsSrgb;
        pSkyBox->mpSampler = pSampler;
        return pSkyBox;
    }

    void SkyBox::deserialize(const RenderPassSerializer& serializer)
    {
        mRenderStereo = serializer.getValue("skybox.renderStereo").b;
        mLoadSrgb = serializer.getValue("skybox.loadAsSrgb").b;
        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(static_cast<Falcor::Sampler::Filter>(serializer.getValue("skybox.sampleDesc.minFilter").i32),
            static_cast<Falcor::Sampler::Filter>(serializer.getValue("skybox.sampleDesc.magFilter").i32),
            static_cast<Falcor::Sampler::Filter>(serializer.getValue("skybox.sampleDesc.mipFilter").i32));

        mpSampler = Sampler::create(samplerDesc);

        Scene::UserVariable var = serializer.getValue("gSkyBoxFilename");

        if (var.type != Scene::UserVariable::Type::Unknown)
        {
            std::string skyBoxName = serializer.getValue("gSkyBoxFilename").str;
            std::string skyBoxFilePath = getDirectoryFromFile(serializer.getValue("gSceneFilename").str) + '/' + skyBoxName;
            mpTexture = createTextureFromFile(skyBoxFilePath, false, mLoadSrgb);
        }
    }

    RenderPassSerializer SkyBox::serialize()
    {
        RenderPassSerializer renderPassSerializer;
        renderPassSerializer.addVariable<std::uint32_t>("skybox.sampleDesc.minFilter", static_cast<uint32_t>(mpSampler->getMinFilter()));
        renderPassSerializer.addVariable<std::uint32_t>("skybox.sampleDesc.magFilter", static_cast<uint32_t>(mpSampler->getMagFilter()));
        renderPassSerializer.addVariable<std::uint32_t>("skybox.sampleDesc.mipFilter", static_cast<uint32_t>(mpSampler->getMipFilter()));
        
        renderPassSerializer.addVariable<bool>("skybox.loadAsSrgb", mLoadSrgb);
        renderPassSerializer.addVariable<bool>("skybox.renderStereo", mRenderStereo);
        return renderPassSerializer;
    }

    void SkyBox::setScene(const std::shared_ptr<Scene>& pScene)
    {
        mpScene = pScene;

        if (!mpTexture && pScene)
        {
            Scene::UserVariable var = pScene->getUserVariable("sky_box");
            std::string textureName = getDirectoryFromFile(pScene->mFileName) + '/' + var.str;
            Texture::SharedPtr pTexture = createTextureFromFile(textureName, false, mLoadSrgb);
            createResources(pTexture, mpSampler, mRenderStereo);
        }
    }

    bool SkyBox::createResources(Texture::SharedPtr& pTexture, Sampler::SharedPtr pSampler, bool renderStereo)
    {
        if(pTexture == nullptr)
        {
            logError("Trying to create a skybox with null texture");
            return false;
        }

        mpSampler = pSampler; 
        mpTexture = pTexture;
        mLoadSrgb = isSrgbFormat(pTexture->getFormat());
        mRenderStereo = renderStereo;

        mpCubeModel = Model::createFromFile("Effects/cube.obj");
        if(mpCubeModel == nullptr)
        {
            logError("Failed to load cube model for SkyBox");
            return false;
        }

        // Create the program
        Program::DefineList defines;
        if(renderStereo)
        {
            defines.add("_SINGLE_PASS_STEREO");
        }

        assert(mpTexture->getType() == Texture::Type::TextureCube || mpTexture->getType() == Texture::Type::Texture2D);
        if (mpTexture->getType() == Texture::Type::Texture2D)
        {
            defines.add("_SPHERICAL_MAP");
        }

        mpProgram = GraphicsProgram::createFromFile("Effects/SkyBox.slang", "vs", "ps", defines);
        mpVars = GraphicsVars::create(mpProgram->getReflector());

        const ParameterBlockReflection* pDefaultBlockReflection = mpProgram->getReflector()->getDefaultParameterBlock().get();
        mBindLocations.perFrameCB = pDefaultBlockReflection->getResourceBinding("PerFrameCB");
        mBindLocations.texture = pDefaultBlockReflection->getResourceBinding("gTexture");
        mBindLocations.sampler = pDefaultBlockReflection->getResourceBinding("gSampler");

        ParameterBlock* pDefaultBlock = mpVars->getDefaultBlock().get();
        ConstantBuffer* pCB = pDefaultBlock->getConstantBuffer(mBindLocations.perFrameCB, 0).get();
        mScaleOffset = pCB->getVariableOffset("gScale");
        mMatOffset = pCB->getVariableOffset("gWorld");

        pDefaultBlock->setSrv(mBindLocations.texture, 0, mpTexture->getSRV());
        pDefaultBlock->setSampler(mBindLocations.sampler, 0, pSampler);

        // Create state
        mpState = GraphicsState::create();
        BlendState::Desc blendDesc;
        for(uint32_t i = 1 ; i < Fbo::getMaxColorTargetCount() ; i++)
        {
            blendDesc.setRenderTargetWriteMask(i, false, false, false, false);
        }
        blendDesc.setIndependentBlend(true);
        mpState->setBlendState(BlendState::create(blendDesc));

        // Create the rasterizer state
        RasterizerState::Desc rastDesc;
        rastDesc.setCullMode(RasterizerState::CullMode::Front).setDepthClamp(true);
        mpState->setRasterizerState(RasterizerState::create(rastDesc));

        DepthStencilState::Desc dsDesc;
        dsDesc.setDepthWriteMask(false).setDepthFunc(DepthStencilState::Func::LessEqual).setDepthTest(true);
        mpState->setDepthStencilState(DepthStencilState::create(dsDesc));
        mpState->setProgram(mpProgram);
        return true;
    }

    Sampler::SharedPtr SkyBox::getSampler() const
    {
        return mpVars->getDefaultBlock()->getSampler(mBindLocations.sampler, 0);
    }

    Texture::SharedPtr SkyBox::getTexture() const
    {
        return mpTexture;
    }

    void SkyBox::setSampler(Sampler::SharedPtr pSampler)
    {
        mpVars->getDefaultBlock()->setSampler(mBindLocations.sampler, 0, pSampler);
    }

    SkyBox::SharedPtr SkyBox::createFromTexture(const std::string& textureName, bool loadAsSrgb, Sampler::SharedPtr pSampler, bool renderStereo)
    {
        Texture::SharedPtr pTexture = createTextureFromFile(textureName, false, loadAsSrgb);
        if(pTexture == nullptr)
        {
            return nullptr;
        }
        return create(pTexture, pSampler, renderStereo);
    }

    void SkyBox::render(RenderContext* pRenderCtx, Camera* pCamera, const Fbo::SharedPtr& pTarget)
    {
        glm::mat4 world = glm::translate(pCamera->getPosition());
        ConstantBuffer* pCB = mpVars->getDefaultBlock()->getConstantBuffer(mBindLocations.perFrameCB, 0).get();
        pCB->setVariable(mMatOffset, world);
        pCB->setVariable(mScaleOffset, mScale);

        mpState->setFbo(pTarget ? pTarget : pRenderCtx->getGraphicsState()->getFbo());
        pRenderCtx->pushGraphicsVars(mpVars);
        pRenderCtx->pushGraphicsState(mpState);

        ModelRenderer::render(pRenderCtx, mpCubeModel, pCamera, false);

        pRenderCtx->popGraphicsVars();
        pRenderCtx->popGraphicsState();
    }

    static const std::string kTarget = "target";
    static const std::string kDepth = "depth";

    void SkyBox::reflect(RenderPassReflection& reflector) const
    {
        reflector.addOutput(kTarget).setFormat(ResourceFormat::RGBA32Float);
        reflector.addInputOutput(kDepth);
    }

    void SkyBox::execute(RenderContext* pRenderContext, const RenderData* pData)
    {
        assert(mpTexture);

        DepthStencilState::Desc dsDesc;
        dsDesc.setDepthFunc(DepthStencilState::Func::Always);
        auto pDS = DepthStencilState::create(dsDesc);

        if (!mpFbo) mpFbo = Fbo::create();
        mpFbo->attachColorTarget(pData->getTexture(kTarget), 0);
        mpFbo->attachDepthStencilTarget(pData->getTexture(kDepth));

        pRenderContext->clearRtv(mpFbo->getRenderTargetView(0).get(), vec4(0));
        render(pRenderContext, mpScene->getActiveCamera().get(), mpFbo);
    }
}