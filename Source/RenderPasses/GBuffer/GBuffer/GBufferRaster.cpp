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
#include "Falcor.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "GBufferRaster.h"

const RenderPass::Info GBufferRaster::kInfo { "GBufferRaster", "Rasterized G-buffer generation pass." };

namespace
{
    const std::string kProgramFile = "RenderPasses/GBuffer/GBuffer/GBufferRaster.3d.slang";
    const std::string shaderModel = "6_2";
    const RasterizerState::CullMode kDefaultCullMode = RasterizerState::CullMode::Back;

    // Additional output channels.
    // TODO: Some are RG32 floats now. I'm sure that all of these could be fp16.
    const std::string kVBufferName = "vbuffer";
    const ChannelList kGBufferExtraChannels =
    {
        { kVBufferName,     "gVBuffer",             "Visibility buffer",                        true /* optional */, ResourceFormat::Unknown /* set at runtime */ },
        { "diffuseOpacity", "gDiffOpacity",         "Diffuse reflection albedo and opacity",    true /* optional */, ResourceFormat::RGBA32Float },
        { "specRough",      "gSpecRough",           "Specular reflectance and roughness",       true /* optional */, ResourceFormat::RGBA32Float },
        { "emissive",       "gEmissive",            "Emissive color",                           true /* optional */, ResourceFormat::RGBA32Float },
        { "viewW",          "gViewW",               "View direction in world space",            true /* optional */, ResourceFormat::RGBA32Float }, // TODO: Switch to packed 2x16-bit snorm format.
        { "pnFwidth",       "gPosNormalFwidth",     "Position and normal filter width",         true /* optional */, ResourceFormat::RG32Float   },
        { "linearZ",        "gLinearZAndDeriv",     "Linear z (and derivative)",                true /* optional */, ResourceFormat::RG32Float   },
    };

    const std::string kDepthName = "depth";
}

RenderPassReflection GBufferRaster::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 sz = RenderPassHelpers::calculateIOSize(mOutputSizeSelection, mFixedOutputSize, compileData.defaultTexDims);

    // Add the required depth output. This always exists.
    reflector.addOutput(kDepthName, "Depth buffer").format(ResourceFormat::D32Float).bindFlags(Resource::BindFlags::DepthStencil).texture2D(sz.x, sz.y);

    // Add all the other outputs.
    // The default channels are written as render targets, the rest as UAVs as there is way to assign/pack render targets yet.
    addRenderPassOutputs(reflector, kGBufferChannels, Resource::BindFlags::RenderTarget, sz);
    addRenderPassOutputs(reflector, kGBufferExtraChannels, Resource::BindFlags::UnorderedAccess, sz);
    reflector.getField(kVBufferName)->format(mVBufferFormat);

    return reflector;
}

GBufferRaster::SharedPtr GBufferRaster::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new GBufferRaster(dict));
}

GBufferRaster::GBufferRaster(const Dictionary& dict)
    : GBuffer(kInfo)
{
    // Check for required features.
    if (!gpDevice->isFeatureSupported(Device::SupportedFeatures::Barycentrics))
    {
        throw RuntimeError("GBufferRaster: Pixel shader barycentrics are not supported by the current device");
    }
    if (!gpDevice->isFeatureSupported(Device::SupportedFeatures::RasterizerOrderedViews))
    {
        throw RuntimeError("GBufferRaster: Rasterizer ordered views (ROVs) are not supported by the current device");
    }

    parseDictionary(dict);

    // Initialize graphics state
    mRaster.pState = GraphicsState::create();

    // Set depth function
    DepthStencilState::Desc dsDesc;
    dsDesc.setDepthFunc(DepthStencilState::Func::Equal).setDepthWriteMask(false);
    DepthStencilState::SharedPtr pDsState = DepthStencilState::create(dsDesc);
    mRaster.pState->setDepthStencilState(pDsState);

    mpFbo = Fbo::create();
}

void GBufferRaster::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    GBuffer::compile(pRenderContext, compileData);

    mpDepthPrePassGraph = RenderGraph::create("Depth Pre-Pass");
    mpDepthPrePass = DepthPass::create(pRenderContext);
    mpDepthPrePass->setDepthBufferFormat(ResourceFormat::D32Float);
    mpDepthPrePassGraph->addPass(mpDepthPrePass, "DepthPrePass");
    mpDepthPrePassGraph->markOutput("DepthPrePass.depth");
    mpDepthPrePassGraph->setScene(mpScene);
}

void GBufferRaster::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    GBuffer::setScene(pRenderContext, pScene);

    mRaster.pProgram = nullptr;
    mRaster.pVars = nullptr;

    if (pScene)
    {
        if (pScene->getMeshVao() && pScene->getMeshVao()->getPrimitiveTopology() != Vao::Topology::TriangleList)
        {
            throw RuntimeError("GBufferRaster: Requires triangle list geometry due to usage of SV_Barycentrics.");
        }

        // Create raster program.
        Program::Desc desc;
        desc.addShaderModules(pScene->getShaderModules());
        desc.addShaderLibrary(kProgramFile).vsEntry("vsMain").psEntry("psMain");
        desc.addTypeConformances(pScene->getTypeConformances());
        desc.setShaderModel(shaderModel);

        mRaster.pProgram = GraphicsProgram::create(desc, pScene->getSceneDefines());
        mRaster.pState->setProgram(mRaster.pProgram);
    }

    if (mpDepthPrePassGraph) mpDepthPrePassGraph->setScene(pScene);
}

void GBufferRaster::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    GBuffer::execute(pRenderContext, renderData);

    // Update frame dimension based on render pass output.
    auto pDepth = renderData.getTexture(kDepthName);
    FALCOR_ASSERT(pDepth);
    updateFrameDim(uint2(pDepth->getWidth(), pDepth->getHeight()));

    // Bind primary channels as render targets and clear them.
    for (size_t i = 0; i < kGBufferChannels.size(); ++i)
    {
        Texture::SharedPtr pTex = getOutput(renderData, kGBufferChannels[i].name);
        mpFbo->attachColorTarget(pTex, uint32_t(i));
    }
    pRenderContext->clearFbo(mpFbo.get(), float4(0), 1.f, 0, FboAttachmentType::Color);

    // Clear extra output buffers.
    clearRenderPassChannels(pRenderContext, kGBufferExtraChannels, renderData);

    // If there is no scene, clear depth buffer and return.
    if (mpScene == nullptr)
    {
        pRenderContext->clearDsv(pDepth->getDSV().get(), 1.f, 0);
        return;
    }

    // Set program defines.
    mRaster.pProgram->addDefine("ADJUST_SHADING_NORMALS", mAdjustShadingNormals ? "1" : "0");
    mRaster.pProgram->addDefine("USE_ALPHA_TEST", mUseAlphaTest ? "1" : "0");

    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    // TODO: This should be moved to a more general mechanism using Slang.
    mRaster.pProgram->addDefines(getValidResourceDefines(kGBufferChannels, renderData));
    mRaster.pProgram->addDefines(getValidResourceDefines(kGBufferExtraChannels, renderData));

    // Create program vars.
    if (!mRaster.pVars)
    {
        mRaster.pVars = GraphicsVars::create(mRaster.pProgram.get());
    }

    // Setup depth pass to use same configuration as this pass.
    RasterizerState::CullMode cullMode = mForceCullMode ? mCullMode : kDefaultCullMode;
    mpDepthPrePass->setCullMode(cullMode);
    mpDepthPrePass->setOutputSize(mFrameDim);
    mpDepthPrePass->setAlphaTest(mUseAlphaTest);

    // Execute depth pass and copy depth buffer.
    mpDepthPrePassGraph->execute(pRenderContext);
    auto pPreDepth = mpDepthPrePassGraph->getOutput("DepthPrePass.depth")->asTexture();
    FALCOR_ASSERT(pPreDepth && pPreDepth->getWidth() == mFrameDim.x && pPreDepth->getHeight() == mFrameDim.y);
    mpFbo->attachDepthStencilTarget(pPreDepth);
    pRenderContext->copyResource(pDepth.get(), pPreDepth.get());

    // Bind extra channels as UAV buffers.
    for (const auto& channel : kGBufferExtraChannels)
    {
        Texture::SharedPtr pTex = getOutput(renderData, channel.name);
        mRaster.pVars[channel.texname] = pTex;
    }

    mRaster.pVars["PerFrameCB"]["gFrameDim"] = mFrameDim;
    mRaster.pState->setFbo(mpFbo); // Sets the viewport

    // Rasterize the scene.
    mpScene->rasterize(pRenderContext, mRaster.pState.get(), mRaster.pVars.get(), cullMode);

    mFrameCount++;
}
