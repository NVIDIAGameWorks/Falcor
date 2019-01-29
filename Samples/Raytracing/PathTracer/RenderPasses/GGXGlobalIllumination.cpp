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
#include "GGXGlobalIllumination.h"

// Some global vars, used to simplify changing shader locations
namespace {
    // Shader files
    const char* kFileRayGen = "GGXGIRayGen.slang";
    const char* kFileRayTrace = "GGXGIIndirectRay.slang";
    const char* kFileShadowRay = "GGXGIShadowRay.slang";

    // Entry-point names
    const char* kEntryPointRayGen = "GGXGlobalIllumRayGen";
    const char* kEntryShadowMiss = "ShadowMiss";
    const char* kEntryShadowAnyHit = "ShadowAnyHit";

    const char* kEntryIndirectMiss = "IndirectMiss";
    const char* kEntryIndirectAnyHit = "IndirectAnyHit";
    const char* kEntryIndirectClosestHit = "IndirectClosestHit";
};

GGXGlobalIllumination::SharedPtr GGXGlobalIllumination::create(const Dictionary &params)
{
    GGXGlobalIllumination::SharedPtr pPass(new GGXGlobalIllumination());

    // Load parameters from Python
    if (params.keyExists("useEmissives"))    pPass->mUseEmissiveGeom = params["useEmissives"];
    if (params.keyExists("doDirectLight"))   pPass->mDoDirectGI = params["doDirectLight"];
    if (params.keyExists("doIndirectLight")) pPass->mDoIndirectGI = params["doIndirectLight"];
    if (params.keyExists("rayDepth"))        pPass->mUserSpecifiedRayDepth = params["rayDepth"];
    if (params.keyExists("randomSeed"))      pPass->mFrameCount = params["randomSeed"];
    if (params.keyExists("useBlackEnvMap"))  pPass->mEnvMapMode = params["useBlackEnvMap"] ? EnvMapMode::Black : EnvMapMode::Scene;

    return pPass;
}

Dictionary GGXGlobalIllumination::getScriptingDictionary() const
{
    Dictionary serialize;
    serialize["useEmissives"] = mUseEmissiveGeom;
    serialize["doDirectLight"] = mDoDirectGI;
    serialize["doIndirectLight"] = mDoIndirectGI;
    serialize["rayDepth"] = mUserSpecifiedRayDepth;
    serialize["randomSeed"] = mFrameCount;
    serialize["useBlackEnvMap"] = mEnvMapMode == EnvMapMode::Black;
    return serialize;
}

RenderPassReflection GGXGlobalIllumination::reflect(void) const
{
    RenderPassReflection r;
    r.addInput("posW", "");
    r.addInput("normW", "");
    r.addInput("diffuseOpacity", "");
    r.addInput("specRough", "");
    r.addInput("emissive", "");
    r.addInput("matlExtra", "");

    r.addOutput("output", "").format(ResourceFormat::RGBA32Float).bindFlags(Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess | Resource::BindFlags::RenderTarget);
    return r;
}

void GGXGlobalIllumination::initialize(RenderContext* pContext, const RenderData* pRenderData)
{
    mpBlackHDR = Texture::create2D(128, 128, ResourceFormat::RGBA32Float, 1u, 1u, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::RenderTarget);
    pContext->clearRtv(mpBlackHDR->getRTV().get(), vec4(0.0f, 0.0f, 0.0f, 1.0f));

    mpState = RtState::create();
    mpState->setMaxTraceRecursionDepth(mMaxPossibleRayDepth);

    RtProgram::Desc desc;
    desc.addShaderLibrary(kFileRayGen);
    desc.setRayGen(kEntryPointRayGen);

    // Add ray type #0 (shadow rays)
    desc.addShaderLibrary(kFileShadowRay);
    desc.addMiss(0, kEntryShadowMiss);
    desc.addHitGroup(0, "", kEntryShadowAnyHit);

    // Add ray type #1 (indirect GI rays)
    desc.addShaderLibrary(kFileRayTrace);
    desc.addMiss(1, kEntryIndirectMiss);
    desc.addHitGroup(1, kEntryIndirectClosestHit, kEntryIndirectAnyHit);

    // Now that we've passed all our shaders in, compile and (if available) setup the scene
    mpProgram = RtProgram::create(desc);
    mpState->setProgram(mpProgram);

    if (mpProgram != nullptr && mpScene != nullptr)
    {
        mpSceneRenderer = RtSceneRenderer::create(mpScene);
        mpVars = RtProgramVars::create(mpProgram, mpScene);
    }

    mIsInitialized = true;
}

void GGXGlobalIllumination::execute(RenderContext* pContext, const RenderData* pData)
{
    // On first execution, run some initialization
    if (!mIsInitialized)
    {
        initialize(pContext, pData);
    }

    // Get our output buffer and clear it
    Texture::SharedPtr pDstTex = pData->getTexture("output");
    pContext->clearUAV(pDstTex->getUAV().get(), vec4(0.0f, 0.0f, 0.0f, 1.0f));

    if (pDstTex == nullptr || mpScene == nullptr) return;

    // Set our variables into the global HLSL namespace
    auto globalVars = mpVars->getGlobalVars();

    ConstantBuffer::SharedPtr pCB = globalVars->getConstantBuffer("GlobalCB");
    pCB["gMinT"] = 1.0e-3f;
    pCB["gFrameCount"] = mFrameCount++;
    pCB["gDoIndirectGI"] = mDoIndirectGI;
    pCB["gDoDirectGI"] = mDoDirectGI;
    pCB["gMaxDepth"] = uint32_t(mUserSpecifiedRayDepth);
    pCB["gEmitMult"] = float(mUseEmissiveGeom ? mEmissiveGeomMult : 0.0f);

    globalVars->setTexture("gPos", pData->getTexture("posW"));
    globalVars->setTexture("gNorm", pData->getTexture("normW"));
    globalVars->setTexture("gDiffuseMatl", pData->getTexture("diffuseOpacity"));
    globalVars->setTexture("gSpecMatl", pData->getTexture("specRough"));
    globalVars->setTexture("gExtraMatl", pData->getTexture("matlExtra"));
    globalVars->setTexture("gEmissive", pData->getTexture("emissive"));
    globalVars->setTexture("gOutput", pDstTex);

    const Texture::SharedPtr& pEnvMap = mpScene->getEnvironmentMap();
    globalVars->setTexture("gEnvMap", (mEnvMapMode == EnvMapMode::Black || pEnvMap == nullptr) ? mpBlackHDR : pEnvMap);

    // Launch our ray tracing
    mpSceneRenderer->renderScene(pContext, mpVars, mpState, mRayLaunchDims);
}

void GGXGlobalIllumination::renderUI(Gui* pGui, const char* uiGroup)
{
    bool changed = pGui->addIntVar("Max RayDepth", mUserSpecifiedRayDepth, 0, mMaxPossibleRayDepth);
    changed |= pGui->addCheckBox("Compute direct illumination", mDoDirectGI);
    changed |= pGui->addCheckBox("Compute global illumination", mDoIndirectGI);

    pGui->addSeparator();

    const static Gui::RadioButtonGroup kButtons =
    {
        {(uint32_t)EnvMapMode::Scene, "Scene", false},
        {(uint32_t)EnvMapMode::Black, "Black", true}
    };

    pGui->addText("Environment Map");
    changed |= pGui->addRadioButtons(kButtons, (uint32_t&)mEnvMapMode);

    if (changed) mPassChangedCB();
}

void GGXGlobalIllumination::setScene(const std::shared_ptr<Scene>& pScene)
{
    // Stash a copy of the scene
    mpScene = std::dynamic_pointer_cast<RtScene>(pScene);
    if (!mpScene) return;

    mpSceneRenderer = RtSceneRenderer::create(mpScene);
    if(mpProgram != nullptr) mpVars = RtProgramVars::create(mpProgram, mpScene);
}

void GGXGlobalIllumination::onResize(uint32_t width, uint32_t height)
{
    mRayLaunchDims = uvec3(width, height, 1);
}
