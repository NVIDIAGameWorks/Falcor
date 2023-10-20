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
#include "Falcor.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "GBufferRaster.h"

namespace
{
const std::string kDepthPassProgramFile = "RenderPasses/GBuffer/GBuffer/DepthPass.3d.slang";
const std::string kGBufferPassProgramFile = "RenderPasses/GBuffer/GBuffer/GBufferRaster.3d.slang";
const RasterizerState::CullMode kDefaultCullMode = RasterizerState::CullMode::Back;

// Additional output channels.
// TODO: Some are RG32 floats now. I'm sure that all of these could be fp16.
const std::string kVBufferName = "vbuffer";
const ChannelList kGBufferExtraChannels = {
    // clang-format off
    { kVBufferName,     "gVBuffer",             "Visibility buffer",                        true /* optional */, ResourceFormat::Unknown /* set at runtime */ },
    { "guideNormalW",   "gGuideNormalW",        "Guide normal in world space",              true /* optional */, ResourceFormat::RGBA32Float },
    { "diffuseOpacity", "gDiffOpacity",         "Diffuse reflection albedo and opacity",    true /* optional */, ResourceFormat::RGBA32Float },
    { "specRough",      "gSpecRough",           "Specular reflectance and roughness",       true /* optional */, ResourceFormat::RGBA32Float },
    { "emissive",       "gEmissive",            "Emissive color",                           true /* optional */, ResourceFormat::RGBA32Float },
    { "viewW",          "gViewW",               "View direction in world space",            true /* optional */, ResourceFormat::RGBA32Float }, // TODO: Switch to packed 2x16-bit snorm format.
    { "pnFwidth",       "gPosNormalFwidth",     "Position and guide normal filter width",   true /* optional */, ResourceFormat::RG32Float   },
    { "linearZ",        "gLinearZAndDeriv",     "Linear z (and derivative)",                true /* optional */, ResourceFormat::RG32Float   },
    { "mask",           "gMask",                "Mask",                                     true /* optional */, ResourceFormat::R32Float    },
    // clang-format on
};

const std::string kDepthName = "depth";
} // namespace

GBufferRaster::GBufferRaster(ref<Device> pDevice, const Properties& props) : GBuffer(pDevice)
{
    // Check for required features.
    if (!mpDevice->isShaderModelSupported(ShaderModel::SM6_2))
        FALCOR_THROW("GBufferRaster requires Shader Model 6.2 support.");
    if (!mpDevice->isFeatureSupported(Device::SupportedFeatures::Barycentrics))
        FALCOR_THROW("GBufferRaster requires pixel shader barycentrics support.");
    if (!mpDevice->isFeatureSupported(Device::SupportedFeatures::RasterizerOrderedViews))
        FALCOR_THROW("GBufferRaster requires rasterizer ordered views (ROVs) support.");

    parseProperties(props);

    // Initialize graphics state
    mDepthPass.pState = GraphicsState::create(mpDevice);
    mGBufferPass.pState = GraphicsState::create(mpDevice);

    // Set depth function
    DepthStencilState::Desc dsDesc;
    dsDesc.setDepthFunc(ComparisonFunc::Equal).setDepthWriteMask(false);
    ref<DepthStencilState> pDsState = DepthStencilState::create(dsDesc);
    mGBufferPass.pState->setDepthStencilState(pDsState);

    mpFbo = Fbo::create(mpDevice);
}

RenderPassReflection GBufferRaster::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 sz = RenderPassHelpers::calculateIOSize(mOutputSizeSelection, mFixedOutputSize, compileData.defaultTexDims);

    // Add the required depth output. This always exists.
    reflector.addOutput(kDepthName, "Depth buffer")
        .format(ResourceFormat::D32Float)
        .bindFlags(ResourceBindFlags::DepthStencil)
        .texture2D(sz.x, sz.y);

    // Add all the other outputs.
    // The default channels are written as render targets, the rest as UAVs as there is way to assign/pack render targets yet.
    addRenderPassOutputs(reflector, kGBufferChannels, ResourceBindFlags::RenderTarget, sz);
    addRenderPassOutputs(reflector, kGBufferExtraChannels, ResourceBindFlags::UnorderedAccess, sz);
    reflector.getField(kVBufferName)->format(mVBufferFormat);

    return reflector;
}

void GBufferRaster::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    GBuffer::compile(pRenderContext, compileData);
}

void GBufferRaster::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    GBuffer::setScene(pRenderContext, pScene);

    recreatePrograms();

    if (pScene)
    {
        if (pScene->getMeshVao() && pScene->getMeshVao()->getPrimitiveTopology() != Vao::Topology::TriangleList)
        {
            FALCOR_THROW("GBufferRaster: Requires triangle list geometry due to usage of SV_Barycentrics.");
        }
    }
}

void GBufferRaster::recreatePrograms()
{
    mDepthPass.pProgram = nullptr;
    mDepthPass.pVars = nullptr;
    mGBufferPass.pProgram = nullptr;
    mGBufferPass.pVars = nullptr;
}

void GBufferRaster::onSceneUpdates(RenderContext* pRenderContext, Scene::UpdateFlags sceneUpdates) {}

void GBufferRaster::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    GBuffer::execute(pRenderContext, renderData);

    // Update frame dimension based on render pass output.
    auto pDepth = renderData.getTexture(kDepthName);
    FALCOR_ASSERT(pDepth);
    updateFrameDim(uint2(pDepth->getWidth(), pDepth->getHeight()));

    // Clear depth buffer.
    pRenderContext->clearDsv(pDepth->getDSV().get(), 1.f, 0);

    // Bind primary channels as render targets and clear them.
    for (size_t i = 0; i < kGBufferChannels.size(); ++i)
    {
        ref<Texture> pTex = getOutput(renderData, kGBufferChannels[i].name);
        mpFbo->attachColorTarget(pTex, uint32_t(i));
    }
    pRenderContext->clearFbo(mpFbo.get(), float4(0), 1.f, 0, FboAttachmentType::Color);

    // Clear extra output buffers.
    clearRenderPassChannels(pRenderContext, kGBufferExtraChannels, renderData);

    // If there is no scene, clear depth buffer and return.
    if (mpScene == nullptr)
    {
        return;
    }

    const RasterizerState::CullMode cullMode = mForceCullMode ? mCullMode : kDefaultCullMode;

    // Check for scene changes.
    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::RecompileNeeded))
    {
        recreatePrograms();
    }

    // Depth pass.
    {
        // Create depth pass program.
        if (!mDepthPass.pProgram)
        {
            ProgramDesc desc;
            desc.addShaderModules(mpScene->getShaderModules());
            desc.addShaderLibrary(kDepthPassProgramFile).vsEntry("vsMain").psEntry("psMain");
            desc.addTypeConformances(mpScene->getTypeConformances());

            mDepthPass.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
            mDepthPass.pState->setProgram(mDepthPass.pProgram);
        }

        // Set program defines.
        mDepthPass.pState->getProgram()->addDefine("USE_ALPHA_TEST", mUseAlphaTest ? "1" : "0");

        // Create program vars.
        if (!mDepthPass.pVars)
            mDepthPass.pVars = ProgramVars::create(mpDevice, mDepthPass.pProgram.get());

        mpFbo->attachDepthStencilTarget(pDepth);
        mDepthPass.pState->setFbo(mpFbo);

        mpScene->rasterize(pRenderContext, mDepthPass.pState.get(), mDepthPass.pVars.get(), cullMode);
    }

    // GBuffer pass.
    {
        // Create GBuffer pass program.
        if (!mGBufferPass.pProgram)
        {
            ProgramDesc desc;
            desc.addShaderModules(mpScene->getShaderModules());
            desc.addShaderLibrary(kGBufferPassProgramFile).vsEntry("vsMain").psEntry("psMain");
            desc.addTypeConformances(mpScene->getTypeConformances());

            mGBufferPass.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
            mGBufferPass.pState->setProgram(mGBufferPass.pProgram);
        }

        // Set program defines.
        mGBufferPass.pProgram->addDefine("ADJUST_SHADING_NORMALS", mAdjustShadingNormals ? "1" : "0");
        mGBufferPass.pProgram->addDefine("USE_ALPHA_TEST", mUseAlphaTest ? "1" : "0");

        // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
        // TODO: This should be moved to a more general mechanism using Slang.
        mGBufferPass.pProgram->addDefines(getValidResourceDefines(kGBufferChannels, renderData));
        mGBufferPass.pProgram->addDefines(getValidResourceDefines(kGBufferExtraChannels, renderData));

        // Create program vars.
        if (!mGBufferPass.pVars)
            mGBufferPass.pVars = ProgramVars::create(mpDevice, mGBufferPass.pProgram.get());

        auto var = mGBufferPass.pVars->getRootVar();

        // Bind extra channels as UAV buffers.
        for (const auto& channel : kGBufferExtraChannels)
        {
            ref<Texture> pTex = getOutput(renderData, channel.name);
            var[channel.texname] = pTex;
        }

        var["PerFrameCB"]["gFrameDim"] = mFrameDim;
        mGBufferPass.pState->setFbo(mpFbo); // Sets the viewport

        // Rasterize the scene.
        mpScene->rasterize(pRenderContext, mGBufferPass.pState.get(), mGBufferPass.pVars.get(), cullMode);
    }

    mFrameCount++;
}
