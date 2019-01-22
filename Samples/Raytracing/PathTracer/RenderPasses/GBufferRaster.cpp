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
#include "GBuffer.h"
#include "GBufferRaster.h"

namespace
{
    const char kFileRasterPrimary[] = "RasterPrimary.slang";
}

RenderPassReflection GBufferRaster::reflect() const
{
    RenderPassReflection r;
    for (int i = 0; i < kGBufferChannelDesc.size(); ++i)
    {
        r.addOutput(kGBufferChannelDesc[i].name, kGBufferChannelDesc[i].desc).format(ResourceFormat::RGBA32Float).bindFlags(Resource::BindFlags::RenderTarget);
    }
    r.addOutput("depthStencil", "depth and stencil").format(ResourceFormat::D32Float).bindFlags(Resource::BindFlags::DepthStencil);
    return r;
}

bool GBufferRaster::parseDictionary(const Dictionary& dict)
{
    for (const auto& v : dict)
    {
        if (v.key() == kCull)
        {
            setCullMode((RasterizerState::CullMode)v.val());
        }
        else
        {
            logWarning("Unknown field `" + v.key() + "` in a GBufferRaster dictionary");
        }
    }
    return true;
}

GBufferRaster::SharedPtr GBufferRaster::create(const Dictionary& dict)
{
    SharedPtr pPass = SharedPtr(new GBufferRaster);
    return pPass->parseDictionary(dict) ? pPass : nullptr;
}

Dictionary GBufferRaster::getScriptingDictionary() const
{
    Dictionary dict;
    dict[kCull] = mCullMode;
    return dict;
}

GBufferRaster::GBufferRaster() : RenderPass("GBufferRaster")
{
    mpGraphicsState = GraphicsState::create();

    mRaster.pProgram = GraphicsProgram::createFromFile(kFileRasterPrimary, "", "ps");

    // Initialize graphics state
    mRaster.pState = GraphicsState::create();

    // Set default culling mode
    setCullMode(mCullMode); 

    mRaster.pVars = GraphicsVars::create(mRaster.pProgram->getReflector());
    mRaster.pState->setProgram(mRaster.pProgram);

    mpFbo = Fbo::create();
}

void GBufferRaster::onResize(uint32_t width, uint32_t height)
{
}

void GBufferRaster::setScene(const std::shared_ptr<Scene>& pScene)
{
    mpSceneRenderer = (pScene == nullptr) ? nullptr : SceneRenderer::create(pScene);
}

void GBufferRaster::renderUI(Gui* pGui, const char* uiGroup)
{
    uint32_t cullMode = (uint32_t)mCullMode;
    if (pGui->addDropdown("Cull Mode", kCullModeList, cullMode))
    {
        setCullMode((RasterizerState::CullMode)cullMode);
    }
}

void GBufferRaster::setCullMode(RasterizerState::CullMode mode)
{
    mCullMode = mode;
    RasterizerState::Desc rsDesc;
    rsDesc.setCullMode(mCullMode);
    mRaster.pState->setRasterizerState(RasterizerState::create(rsDesc));
}


void GBufferRaster::execute(RenderContext* pContext, const RenderData* pRenderData)
{
    if (mpSceneRenderer == nullptr)
    {
        logWarning("Invalid SceneRenderer in GBufferRaster::execute()");
        return;
    }

    mpFbo->attachDepthStencilTarget(pRenderData->getTexture("depthStencil"));

    for (int i = 0; i < kGBufferChannelDesc.size(); ++i)
    {
        mpFbo->attachColorTarget(pRenderData->getTexture(kGBufferChannelDesc[i].name), i);
    }

    pContext->clearFbo(mpFbo.get(), vec4(0), 1.f, 0, FboAttachmentType::All);
    mRaster.pState->setFbo(mpFbo);

    pContext->setGraphicsState(mRaster.pState);
    pContext->setGraphicsVars(mRaster.pVars);
    mpSceneRenderer->renderScene(pContext);
}
