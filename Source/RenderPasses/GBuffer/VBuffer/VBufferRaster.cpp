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
#include "VBufferRaster.h"
#include "Scene/HitInfo.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"

namespace
{
const std::string kProgramFile = "RenderPasses/GBuffer/VBuffer/VBufferRaster.3d.slang";
const RasterizerState::CullMode kDefaultCullMode = RasterizerState::CullMode::Back;

const std::string kVBufferName = "vbuffer";
const std::string kVBufferDesc = "V-buffer in packed format (indices + barycentrics)";

const ChannelList kVBufferExtraChannels = {
    // clang-format off
    { "mvec",           "gMotionVector",    "Motion vector",                true /* optional */, ResourceFormat::RG32Float   },
    { "mask",           "gMask",            "Mask",                         true /* optional */, ResourceFormat::R32Float    },
    // clang-format on
};

const std::string kDepthName = "depth";
} // namespace

VBufferRaster::VBufferRaster(ref<Device> pDevice, const Properties& props) : GBufferBase(pDevice)
{
    // Check for required features.
    if (!mpDevice->isShaderModelSupported(ShaderModel::SM6_2))
        FALCOR_THROW("VBufferRaster requires Shader Model 6.2 support.");
    if (!mpDevice->isFeatureSupported(Device::SupportedFeatures::Barycentrics))
        FALCOR_THROW("VBufferRaster requires pixel shader barycentrics support.");
    if (!mpDevice->isFeatureSupported(Device::SupportedFeatures::RasterizerOrderedViews))
        FALCOR_THROW("VBufferRaster requires rasterizer ordered views (ROVs) support.");

    parseProperties(props);

    // Initialize graphics state
    mRaster.pState = GraphicsState::create(mpDevice);

    // Set depth function
    DepthStencilState::Desc dsDesc;
    dsDesc.setDepthFunc(ComparisonFunc::LessEqual).setDepthWriteMask(true);
    mRaster.pState->setDepthStencilState(DepthStencilState::create(dsDesc));

    mpFbo = Fbo::create(mpDevice);
}

RenderPassReflection VBufferRaster::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 sz = RenderPassHelpers::calculateIOSize(mOutputSizeSelection, mFixedOutputSize, compileData.defaultTexDims);

    // Add the required outputs. These always exist.
    reflector.addOutput(kDepthName, "Depth buffer")
        .format(ResourceFormat::D32Float)
        .bindFlags(ResourceBindFlags::DepthStencil)
        .texture2D(sz.x, sz.y);
    reflector.addOutput(kVBufferName, kVBufferDesc)
        .bindFlags(ResourceBindFlags::RenderTarget | ResourceBindFlags::UnorderedAccess)
        .format(mVBufferFormat)
        .texture2D(sz.x, sz.y);

    // Add all the other outputs.
    addRenderPassOutputs(reflector, kVBufferExtraChannels, ResourceBindFlags::UnorderedAccess, sz);

    return reflector;
}

void VBufferRaster::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    GBufferBase::setScene(pRenderContext, pScene);

    recreatePrograms();

    if (pScene)
    {
        if (pScene->getMeshVao() && pScene->getMeshVao()->getPrimitiveTopology() != Vao::Topology::TriangleList)
        {
            FALCOR_THROW("VBufferRaster: Requires triangle list geometry due to usage of SV_Barycentrics.");
        }
    }
}

void VBufferRaster::recreatePrograms()
{
    mRaster.pProgram = nullptr;
    mRaster.pVars = nullptr;
}

void VBufferRaster::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    GBufferBase::execute(pRenderContext, renderData);

    // Update frame dimension based on render pass output.
    auto pOutput = renderData.getTexture(kVBufferName);
    FALCOR_ASSERT(pOutput);
    updateFrameDim(uint2(pOutput->getWidth(), pOutput->getHeight()));

    // Clear depth and output buffer.
    auto pDepth = getOutput(renderData, kDepthName);
    pRenderContext->clearUAV(pOutput->getUAV().get(), uint4(0)); // Clear as UAV for integer clear value
    pRenderContext->clearDsv(pDepth->getDSV().get(), 1.f, 0);

    // Clear extra output buffers.
    clearRenderPassChannels(pRenderContext, kVBufferExtraChannels, renderData);

    // If there is no scene, we're done.
    if (mpScene == nullptr)
    {
        return;
    }

    // Check for scene changes.
    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::RecompileNeeded))
    {
        recreatePrograms();
    }

    // Create raster program.
    if (!mRaster.pProgram)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kProgramFile).vsEntry("vsMain").psEntry("psMain");
        desc.addTypeConformances(mpScene->getTypeConformances());

        mRaster.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
        mRaster.pState->setProgram(mRaster.pProgram);
    }

    // Set program defines.
    mRaster.pProgram->addDefine("USE_ALPHA_TEST", mUseAlphaTest ? "1" : "0");

    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    // TODO: This should be moved to a more general mechanism using Slang.
    mRaster.pProgram->addDefines(getValidResourceDefines(kVBufferExtraChannels, renderData));

    // Create program vars.
    if (!mRaster.pVars)
    {
        mRaster.pVars = ProgramVars::create(mpDevice, mRaster.pProgram.get());
    }

    mpFbo->attachColorTarget(pOutput, 0);
    mpFbo->attachDepthStencilTarget(pDepth);
    mRaster.pState->setFbo(mpFbo); // Sets the viewport

    auto var = mRaster.pVars->getRootVar();
    var["PerFrameCB"]["gFrameDim"] = mFrameDim;

    // Bind extra channels as UAV buffers.
    for (const auto& channel : kVBufferExtraChannels)
    {
        ref<Texture> pTex = getOutput(renderData, channel.name);
        var[channel.texname] = pTex;
    }

    // Rasterize the scene.
    RasterizerState::CullMode cullMode = mForceCullMode ? mCullMode : kDefaultCullMode;
    mpScene->rasterize(pRenderContext, mRaster.pState.get(), mRaster.pVars.get(), cullMode);
}
