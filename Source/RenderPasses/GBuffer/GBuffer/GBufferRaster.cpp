/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "Falcor.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "GBufferRaster.h"

const char* GBufferRaster::kDesc = "Rasterized G-buffer generation pass";

namespace
{
    const std::string kProgramFile = "RenderPasses/GBuffer/GBuffer/GBufferRaster.3d.slang";
    const std::string shaderModel = "6_1";

    // Additional output channels.
    // TODO: Some are RG32 floats now. I'm sure that all of these could be fp16.
    const ChannelList kGBufferExtraChannels =
    {
        { "vbuffer",          "gVBuffer",            "Visibility buffer",                true /* optional */, ResourceFormat::RG32Uint    },
        { "mvec",             "gMotionVectors",      "Motion vectors",                   true /* optional */, ResourceFormat::RG32Float   },
        { "faceNormalW",      "gFaceNormalW",        "Face normal in world space",       true /* optional */, ResourceFormat::RGBA32Float },
        { "pnFwidth",         "gPosNormalFwidth",    "position and normal filter width", true /* optional */, ResourceFormat::RG32Float   },
        { "linearZ",          "gLinearZAndDeriv",    "linear z (and derivative)",        true /* optional */, ResourceFormat::RG32Float   },
        { "surfSpreadAngle",  "gSurfaceSpreadAngle", "surface spread angle (texlod)",    true /* optional */, ResourceFormat::R16Float    },
    };

    const std::string kDepthName = "depth";
}

RenderPassReflection GBufferRaster::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    // Add the required depth output. This always exists.
    reflector.addOutput(kDepthName, "Depth buffer").format(ResourceFormat::D32Float).bindFlags(Resource::BindFlags::DepthStencil);

    // Add all the other outputs.
    // The default channels are written as render targets, the rest as UAVs as there is way to assign/pack render targets yet.
    addRenderPassOutputs(reflector, kGBufferChannels, Resource::BindFlags::RenderTarget);
    addRenderPassOutputs(reflector, kGBufferExtraChannels, Resource::BindFlags::UnorderedAccess);

    return reflector;
}

GBufferRaster::SharedPtr GBufferRaster::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new GBufferRaster(dict));
}

GBufferRaster::GBufferRaster(const Dictionary& dict)
    : GBuffer()
{
    parseDictionary(dict);

    // Create raster program
    Program::DefineList defines = { { "_DEFAULT_ALPHA_TEST", "" } };
    Program::Desc desc;
    desc.addShaderLibrary(kProgramFile).vsEntry("vsMain").psEntry("psMain");
    desc.setShaderModel(shaderModel);
    mRaster.pProgram = GraphicsProgram::create(desc, defines);

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
    mpDepthPrePass = DepthPass::create(pContext);
    mpDepthPrePass->setDepthBufferFormat(ResourceFormat::D32Float);
    mpDepthPrePassGraph->addPass(mpDepthPrePass, "DepthPrePass");
    mpDepthPrePassGraph->markOutput("DepthPrePass.depth");
    mpDepthPrePassGraph->setScene(mpScene);
}

void GBufferRaster::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    GBuffer::setScene(pRenderContext, pScene);

    mRaster.pVars = nullptr;

    if (pScene)
    {
        if (pScene->getVao()->getPrimitiveTopology() != Vao::Topology::TriangleList)
        {
            throw std::exception("GBufferRaster only works with triangle list geometry due to usage of SV_Barycentrics.");
        }

        mRaster.pProgram->addDefines(pScene->getSceneDefines());
    }

    if (mpDepthPrePassGraph) mpDepthPrePassGraph->setScene(pScene);
}

void GBufferRaster::setCullMode(RasterizerState::CullMode mode)
{
    GBuffer::setCullMode(mode);
    RasterizerState::Desc rsDesc;
    rsDesc.setCullMode(mCullMode);
    mRaster.pRsState = RasterizerState::create(rsDesc);
    assert(mRaster.pState);
    mRaster.pState->setRasterizerState(mRaster.pRsState);
}

void GBufferRaster::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    GBuffer::execute(pRenderContext, renderData);

    // Bind primary channels as render targets and clear them.
    for (size_t i = 0; i < kGBufferChannels.size(); ++i)
    {
        Texture::SharedPtr pTex = renderData[kGBufferChannels[i].name]->asTexture();
        mpFbo->attachColorTarget(pTex, uint32_t(i));
    }
    pRenderContext->clearFbo(mpFbo.get(), float4(0), 1.f, 0, FboAttachmentType::Color);

    // If there is no scene, clear the outputs and return.
    if (mpScene == nullptr)
    {
        auto clear = [&](const ChannelDesc& channel)
        {
            auto pTex = renderData[channel.name]->asTexture();
            if (pTex) pRenderContext->clearUAV(pTex->getUAV().get(), float4(0.f));
        };
        for (const auto& channel : kGBufferExtraChannels) clear(channel);
        auto pDepth = renderData[kDepthName]->asTexture();
        pRenderContext->clearDsv(pDepth->getDSV().get(), 1.f, 0);
        return;
    }

    // Set program defines.
    mRaster.pProgram->addDefine("DISABLE_ALPHA_TEST", mDisableAlphaTest ? "1" : "0");

    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    // TODO: This should be moved to a more general mechanism using Slang.
    mRaster.pProgram->addDefines(getValidResourceDefines(kGBufferExtraChannels, renderData));

    // Create program vars.
    if (!mRaster.pVars)
    {
        mRaster.pVars = GraphicsVars::create(mRaster.pProgram.get());
    }

    // Setup depth pass to use same culling mode.
    mpDepthPrePass->setRasterizerState(mForceCullMode ? mRaster.pRsState : nullptr);

    // Copy depth buffer.
    mpDepthPrePassGraph->execute(pRenderContext);
    mpFbo->attachDepthStencilTarget(mpDepthPrePassGraph->getOutput("DepthPrePass.depth")->asTexture());
    pRenderContext->copyResource(renderData[kDepthName].get(), mpDepthPrePassGraph->getOutput("DepthPrePass.depth").get());

    // Bind extra channels as UAV buffers.
    for (const auto& channel : kGBufferExtraChannels)
    {
        Texture::SharedPtr pTex = renderData[channel.name]->asTexture();
        if (pTex) pRenderContext->clearUAV(pTex->getUAV().get(), float4(0, 0, 0, 0));
        mRaster.pVars[channel.texname] = pTex;
    }

    mRaster.pVars["PerFrameCB"]["gParams"].setBlob(mGBufferParams);
    mRaster.pState->setFbo(mpFbo); // Sets the viewport

    Scene::RenderFlags flags = mForceCullMode ? Scene::RenderFlags::UserRasterizerState : Scene::RenderFlags::None;
    mpScene->render(pRenderContext, mRaster.pState.get(), mRaster.pVars.get(), flags);

    mGBufferParams.frameCount++;
}
