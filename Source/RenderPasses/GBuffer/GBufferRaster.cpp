/***************************************************************************
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include "Falcor.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "GBufferRaster.h"
#include "RenderPasses/DepthPass.h"

namespace
{
    const char kFileRasterPrimary[] = "RenderPasses/GBuffer/RasterPrimary.slang";

    // Additional output channels.
    // TODO: Some are RG32 floats now. I'm sure that all of these could be fp16.
    const ChannelList kGBufferRasterChannels =
    {
        { "faceNormalW", "gFaceNormalW",     "Face normal in world space",       true /* optional */, ResourceFormat::RGBA32Float },
        { "mvec",        "gMotionVectors",   "motion vectors",                   true /* optional */, ResourceFormat::RG32Float },
        { "pnFwidth",    "gPosNormalFwidth", "position and normal filter width", true /* optional */, ResourceFormat::RG32Float },
        { "linearZ",     "gLinearZAndDeriv", "linear z (and derivative)",        true /* optional */, ResourceFormat::RG32Float },
    };
}

RenderPassReflection GBufferRaster::reflect(const CompileData& compileData)
{
    RenderPassReflection r;

    // Add the required depth/stencil output. This always exists.
    r.addOutput("depthStencil", "depth and stencil").format(ResourceFormat::D32Float).bindFlags(Resource::BindFlags::DepthStencil);

    // Add all the other outputs.
    // The default channels are written as render targets, the rest as UAVs as there is way to assign/pack render targets yet.
    auto addOutput = [&](const ChannelDesc& output, Resource::BindFlags bindFlags)
    {
        auto& f = r.addOutput(output.name, output.desc).format(output.format).bindFlags(bindFlags);
        if (output.optional) f.flags(RenderPassReflection::Field::Flags::Optional);
    };
    for (auto it : kGBufferChannels) addOutput(it, Resource::BindFlags::RenderTarget);
    for (auto it : kGBufferRasterChannels) addOutput(it, Resource::BindFlags::UnorderedAccess);

    return r;
}

GBufferRaster::SharedPtr GBufferRaster::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    SharedPtr pPass = SharedPtr(new GBufferRaster);
    return pPass->parseDictionary(dict) ? pPass : nullptr;
}

GBufferRaster::GBufferRaster() : GBuffer()
{
    // Create raster program
    mRaster.pProgram = GraphicsProgram::createFromFile(kFileRasterPrimary, "", "ps");
    if (!mRaster.pProgram) throw std::exception("Failed to create program");

    // Initialize graphics state
    mRaster.pState = GraphicsState::create();
    mRaster.pState->setProgram(mRaster.pProgram);

    // Set default cull mode
    setCullMode(mCullMode);

    // Set depth function
    DepthStencilState::Desc dsDesc;
    dsDesc.setDepthFunc(DepthStencilState::Func::Equal).setDepthWriteMask(false);
    DepthStencilState::SharedPtr pDsState = DepthStencilState::create(dsDesc);
    mRaster.pState->setDepthStencilState(pDsState);

    mpFbo = Fbo::create();
}

void GBufferRaster::compile(RenderContext* pContext, const CompileData& compileData)
{
    GBuffer::compile(pContext, compileData);
    mpDepthPrePassGraph = RenderGraph::create("Depth Pre-Pass");
    DepthPass::SharedPtr pDepthPass = DepthPass::create(pContext);
    pDepthPass->setDepthBufferFormat(ResourceFormat::D32Float);
    mpDepthPrePassGraph->addPass(pDepthPass, "DepthPrePass");
    mpDepthPrePassGraph->markOutput("DepthPrePass.depth");
    mpDepthPrePassGraph->setScene(mpScene);
}

void GBufferRaster::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    GBuffer::setScene(pRenderContext, pScene);

    mRaster.pProgram->addDefines(pScene->getSceneDefines());
    mRaster.pVars = GraphicsVars::create(mRaster.pProgram.get());
    if (!mRaster.pVars) throw std::exception("Failed to create program vars");

    if (mpDepthPrePassGraph) mpDepthPrePassGraph->setScene(pScene);
}

void GBufferRaster::setCullMode(RasterizerState::CullMode mode)
{
    GBuffer::setCullMode(mode);
    RasterizerState::Desc rsDesc;
    rsDesc.setCullMode(mCullMode);
    mRaster.pState->setRasterizerState(RasterizerState::create(rsDesc));
}

void GBufferRaster::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Update refresh flag if options that affect the output have changed.
    if (mOptionsChanged)
    {
        Dictionary& dict = renderData.getDictionary();
        auto prevFlags = (Falcor::RenderPassRefreshFlags)(dict.keyExists(kRenderPassRefreshFlags) ? dict[Falcor::kRenderPassRefreshFlags] : 0u);
        dict[Falcor::kRenderPassRefreshFlags] = (uint32_t)(prevFlags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged);
        mOptionsChanged = false;
    }

    if (mpScene == nullptr)
    {
        logWarning("GBufferRaster::execute() - No scene available");
        return;
    }

    mpDepthPrePassGraph->execute(pRenderContext);
    mpFbo->attachDepthStencilTarget(mpDepthPrePassGraph->getOutput("DepthPrePass.depth")->asTexture());
    pRenderContext->copyResource(renderData["depthStencil"].get(), mpDepthPrePassGraph->getOutput("DepthPrePass.depth").get());

    for (int i = 0; i < kGBufferChannels.size(); ++i)
    {
        Texture::SharedPtr pTex = renderData[kGBufferChannels[i].name]->asTexture();
        mpFbo->attachColorTarget(pTex, i);
    }

    pRenderContext->clearFbo(mpFbo.get(), vec4(0), 1.f, 0, FboAttachmentType::Color);
    mRaster.pState->setFbo(mpFbo);

    mRaster.pVars["PerFrameCB"]["gParams"].setBlob(mGBufferParams);

    // UAV output variables
    for (auto it : kGBufferRasterChannels)
    {
        Texture::SharedPtr pTex = renderData[it.name]->asTexture();
        if (pTex) pRenderContext->clearUAV(pTex->getUAV().get(), glm::vec4(0, 0, 0, 0));
        mRaster.pVars[it.texname] = pTex;
    }

    Scene::RenderFlags flags = mForceCullMode ? Scene::RenderFlags::UserRasterizerState : Scene::RenderFlags::None;
    mpScene->render(pRenderContext, mRaster.pState.get(), mRaster.pVars.get(), flags);

    mGBufferParams.frameCount++;
}
