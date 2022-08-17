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
#include "ForwardLightingPass.h"
#include "RenderGraph/RenderPassLibrary.h"

const RenderPass::Info ForwardLightingPass::kInfo
{
    "ForwardLightingPass",

    "Computes direct and indirect illumination and applies shadows for the current scene (if visibility map is provided).\n"
    "The pass can output the world-space normals and screen-space motion vectors, both are optional."
};

// Don't remove this. it's required for hot-reload to function properly
extern "C" FALCOR_API_EXPORT const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" FALCOR_API_EXPORT void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerPass(ForwardLightingPass::kInfo, ForwardLightingPass::create);
}

namespace
{
    const char kShaderFile[] = "RenderPasses/ForwardLightingPass/ForwardLightingPass.3d.slang";

    const std::string kDepth = "depth";
    const std::string kColor = "color";
    const std::string kMotionVecs = "motionVecs";
    const std::string kNormals = "normals";
    const std::string kVisBuffer = "visibilityBuffer";

    const std::string kSampleCount = "sampleCount";
    const std::string kSuperSampling = "enableSuperSampling";
}

ForwardLightingPass::SharedPtr ForwardLightingPass::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    auto pThis = SharedPtr(new ForwardLightingPass());
    pThis->setColorFormat(ResourceFormat::RGBA32Float).setMotionVecFormat(ResourceFormat::RG16Float).setNormalMapFormat(ResourceFormat::RGBA8Unorm).setSampleCount(1).usePreGeneratedDepthBuffer(true);

    for (const auto& [key, value] : dict)
    {
        if (key == kSampleCount) pThis->setSampleCount(value);
        else if (key == kSuperSampling) pThis->setSuperSampling(value);
        else logWarning("Unknown field '{}' in a ForwardLightingPass dictionary.", key);
    }

    return pThis;
}

Dictionary ForwardLightingPass::getScriptingDictionary()
{
    Dictionary d;
    d[kSampleCount] = mSampleCount;
    d[kSuperSampling] = mEnableSuperSampling;
    return d;
}

ForwardLightingPass::ForwardLightingPass()
    : RenderPass(kInfo)
{
    mpState = GraphicsState::create();
    mpFbo = Fbo::create();

    DepthStencilState::Desc dsDesc;
    dsDesc.setDepthWriteMask(false).setDepthFunc(DepthStencilState::Func::LessEqual);
    mpDsNoDepthWrite = DepthStencilState::create(dsDesc);
}

RenderPassReflection ForwardLightingPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    reflector.addInput(kVisBuffer, "Visibility buffer used for shadowing. Range is [0,1] where 0 means the pixel is fully-shadowed and 1 means the pixel is not shadowed at all").flags(RenderPassReflection::Field::Flags::Optional);
    reflector.addInputOutput(kColor, "Color texture").format(mColorFormat).texture2D(0, 0, mSampleCount);

    auto& depthField = mUsePreGenDepth ? reflector.addInputOutput(kDepth, "Pre-initialized depth-buffer") : reflector.addOutput(kDepth, "Depth buffer");
    depthField.bindFlags(Resource::BindFlags::DepthStencil).texture2D(0, 0, mSampleCount);

    if (mNormalMapFormat != ResourceFormat::Unknown)
    {
        reflector.addOutput(kNormals, "World-space shading normal, [0,1] range. Don't forget to transform it to [-1, 1] range").format(mNormalMapFormat).texture2D(0, 0, mSampleCount);
    }

    if (mMotionVecFormat != ResourceFormat::Unknown)
    {
        reflector.addOutput(kMotionVecs, "Screen-space motion vectors").format(mMotionVecFormat).texture2D(0, 0, mSampleCount);
    }

    return reflector;
}

void ForwardLightingPass::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    mpScene = pScene;
    mpVars = nullptr;

    if (mpScene)
    {
        Program::Desc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderFile).vsEntry("vsMain").psEntry("psMain");
        desc.addTypeConformances(mpScene->getTypeConformances());
        GraphicsProgram::SharedPtr pProgram = GraphicsProgram::create(desc, mpScene->getSceneDefines());

        mpVars = GraphicsVars::create(pProgram->getReflector());
        mpState->setProgram(pProgram);

        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
        setSampler(Sampler::create(samplerDesc));
    }
}

void ForwardLightingPass::initDepth(const RenderData& renderData)
{
    const auto& pTexture = renderData.getTexture(kDepth);

    if (pTexture)
    {
        mpState->setDepthStencilState(mpDsNoDepthWrite);
        mpFbo->attachDepthStencilTarget(pTexture);
    }
    else
    {
        mpState->setDepthStencilState(nullptr);
        if (mpFbo->getDepthStencilTexture() == nullptr)
        {
            auto pDepth = Texture::create2D(mpFbo->getWidth(), mpFbo->getHeight(), ResourceFormat::D32Float, 1, 1, nullptr, Resource::BindFlags::DepthStencil);
            mpFbo->attachDepthStencilTarget(pDepth);
        }
    }
}

void ForwardLightingPass::initFbo(RenderContext* pRenderContext, const RenderData& renderData)
{
    mpFbo->attachColorTarget(renderData.getTexture(kColor), 0);
    mpFbo->attachColorTarget(renderData.getTexture(kNormals), 1);
    mpFbo->attachColorTarget(renderData.getTexture(kMotionVecs), 2);

    for (uint32_t i = 1; i < 3; i++)
    {
        const auto& pRtv = mpFbo->getRenderTargetView(i).get();
        if (pRtv->getResource() != nullptr) pRenderContext->clearRtv(pRtv, float4(0));
    }

    // TODO Matt (not really matt, just need to fix that since if depth is not bound the pass crashes
    if (mUsePreGenDepth == false) pRenderContext->clearDsv(renderData.getTexture(kDepth)->getDSV().get(), 1, 0);
}

void ForwardLightingPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    initDepth(renderData);
    initFbo(pRenderContext, renderData);

    if (!mpScene) return;

    // Prepare program.
    if (mEnableSuperSampling)
    {
        mpState->getProgram()->addDefine("INTERPOLATION_MODE", "sample");
    }
    else
    {
        mpState->getProgram()->removeDefine("INTERPOLATION_MODE");
    }

    if (mMotionVecFormat != ResourceFormat::Unknown)
    {
        mpState->getProgram()->addDefine("_OUTPUT_MOTION_VECTORS");
    }
    else
    {
        mpState->getProgram()->removeDefine("_OUTPUT_MOTION_VECTORS");
    }

    // Update env map lighting
    const auto& pEnvMap = mpScene->getEnvMap();
    if (pEnvMap && (!mpEnvMapLighting || mpEnvMapLighting->getEnvMap() != pEnvMap))
    {
        mpEnvMapLighting = EnvMapLighting::create(pRenderContext, pEnvMap);
        mpEnvMapLighting->setShaderData(mpVars["gEnvMapLighting"]);
        mpState->getProgram()->addDefine("_USE_ENV_MAP");
    }
    else if (!pEnvMap)
    {
        mpEnvMapLighting = nullptr;
        mpState->getProgram()->removeDefine("_USE_ENV_MAP");
    }

    mpVars["PerFrameCB"]["gRenderTargetDim"] = float2(mpFbo->getWidth(), mpFbo->getHeight());
    mpVars["PerFrameCB"]["gFrameCount"] = mFrameCount;
    mpVars->setTexture(kVisBuffer, renderData.getTexture(kVisBuffer));

    mpState->setFbo(mpFbo);
    mpScene->rasterize(pRenderContext, mpState.get(), mpVars.get());

    mFrameCount++;
}

void ForwardLightingPass::renderUI(Gui::Widgets& widget)
{
    static const Gui::DropdownList kSampleCountList =
    {
        { 1, "1" },
        { 2, "2" },
        { 4, "4" },
        { 8, "8" },
    };

    if (widget.dropdown("Sample Count", kSampleCountList, mSampleCount))              setSampleCount(mSampleCount);
    if (mSampleCount > 1 && widget.checkbox("Super Sampling", mEnableSuperSampling))  setSuperSampling(mEnableSuperSampling);
}

ForwardLightingPass& ForwardLightingPass::setColorFormat(ResourceFormat format)
{
    mColorFormat = format;
    requestRecompile();
    return *this;
}

ForwardLightingPass& ForwardLightingPass::setNormalMapFormat(ResourceFormat format)
{
    mNormalMapFormat = format;
    requestRecompile();
    return *this;
}

ForwardLightingPass& ForwardLightingPass::setMotionVecFormat(ResourceFormat format)
{
    mMotionVecFormat = format;
    requestRecompile();
    return *this;
}

ForwardLightingPass& ForwardLightingPass::setSampleCount(uint32_t samples)
{
    mSampleCount = samples;
    requestRecompile();
    return *this;
}

ForwardLightingPass& ForwardLightingPass::setSuperSampling(bool enable)
{
    mEnableSuperSampling = enable;
    return *this;
}

ForwardLightingPass& ForwardLightingPass::usePreGeneratedDepthBuffer(bool enable)
{
    mUsePreGenDepth = enable;
    requestRecompile();
    mpState->setDepthStencilState(mUsePreGenDepth ? mpDsNoDepthWrite : nullptr);

    return *this;
}

ForwardLightingPass& ForwardLightingPass::setSampler(const Sampler::SharedPtr& pSampler)
{
    mpVars->setSampler("gSampler", pSampler);
    return *this;
}
