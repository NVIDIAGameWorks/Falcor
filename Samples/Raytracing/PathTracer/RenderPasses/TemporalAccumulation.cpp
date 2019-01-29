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
#include "TemporalAccumulation.h"

namespace {
    // Where is our shader located? 
    const char *kAccumShader = "Accumulate.slang";

    // Key used to look up dirty state in the graph's shared dictionary
    const char *kDirtyFlag = "_dirty";
};

TemporalAccumulation::SharedPtr TemporalAccumulation::create(const Dictionary &params)
{
    TemporalAccumulation::SharedPtr ptr(new TemporalAccumulation());

    // Load parameters from Python
    if (params.keyExists("doAccumulation")) ptr->mDoAccumulation = params["doAccumulation"];

    return ptr;
}

Dictionary TemporalAccumulation::getScriptingDictionary() const
{
    Dictionary serialize;
    serialize["doAccumulation"] = mDoAccumulation;
    return serialize;
}

RenderPassReflection TemporalAccumulation::reflect(void) const
{
    RenderPassReflection r;
    r.addInput("input", "");

    Resource::BindFlags bindFlags = Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess | Resource::BindFlags::RenderTarget;
    r.addInternal("accum", "").format(ResourceFormat::RGBA32Float).bindFlags(bindFlags).flags(RenderPassReflection::Field::Flags::Persistent);
    r.addOutput("output", "").format(ResourceFormat::RGBA32Float).bindFlags(bindFlags);
    return r;
}

void TemporalAccumulation::initialize(const Dictionary& dict)
{
    mDict = dict;
    mpState = GraphicsState::create();
    mpPass = FullScreenPass::create(kAccumShader);
    mpVars = GraphicsVars::create(mpPass->getProgram()->getReflector());

    mIsInitialized = true;
}

void TemporalAccumulation::execute(RenderContext* pContext, const RenderData* pRenderData)
{
    // On first execution, run some initialization
    if (!mIsInitialized) initialize(pRenderData->getDictionary());

    // Get references to our input, output, and temporary accumulation texture
    Texture::SharedPtr pSrcTex = pRenderData->getTexture("input");
    Texture::SharedPtr pAccumTex = pRenderData->getTexture("accum");
    Texture::SharedPtr pDstTex = pRenderData->getTexture("output");

    Fbo::SharedPtr pDstFbo = Fbo::create();
    pDstFbo->attachColorTarget(pDstTex, 0);

    // If we don't have our input our output texture, give up....  Don't know what to do...
    if (!pSrcTex || !pDstTex) return;

    // We're not doing accumulation, copy the input to the output
    if (!mDoAccumulation)
    {
        pContext->blit(pSrcTex->getSRV(), pDstTex->getRTV());
        return;
    }

    // If the camera in our current scene has moved, or the GUI pass settings changed, we want to reset accumulation
    auto& pDict = pRenderData->getDictionary();
    bool giDirty = pDict.keyExists(kDirtyFlag) && bool(pDict[kDirtyFlag]);
    if (hasCameraMoved() || giDirty)
    {
        mAccumCount = 0;
        mpLastCameraMatrix = mpScene->getActiveCamera()->getViewMatrix();

        if (giDirty) pDict[kDirtyFlag] = false; // Reset the flag
    }

    // Set shader parameters for our accumulation pass
    mpVars["PerFrameCB"]["gAccumCount"] = mAccumCount++;  // Current count of accumulated samples
    mpVars->setTexture("gAccumBuf", pAccumTex); // Intermediate texture where running tally is accumulated
    mpVars->setTexture("gCurFrame", pSrcTex);   // Input samples for this frame

    mpState->setFbo(pDstFbo);
    pContext->pushGraphicsState(mpState);
    pContext->pushGraphicsVars(mpVars);
    mpPass->execute(pContext);
    pContext->popGraphicsVars();
    pContext->popGraphicsState();
}

void TemporalAccumulation::renderUI(Gui* pGui, const char* uiGroup)
{
    // Reset accumulation if settings changed, or manual reset requested
    bool reset = pGui->addCheckBox("Accumulate Samples", mDoAccumulation);
    reset |= pGui->addButton("Reset", true);

    if (reset) mAccumCount = 0;

    // Display amount of accumulated frames
    pGui->addText((std::string("Frames accumulated: ") + std::to_string(mAccumCount)).c_str());
}

bool TemporalAccumulation::hasCameraMoved()
{
    // Has our camera moved?
    return mpScene &&                   // No scene?  Then the answer is no
        mpScene->getActiveCamera() &&   // No camera in our scene?  Then the answer is no
        (mpLastCameraMatrix != mpScene->getActiveCamera()->getViewMatrix());   // Compare the current matrix with the last one
}

void TemporalAccumulation::setScene(const std::shared_ptr<Scene>& pScene)
{
    // Reset accumulation on loading a new scene
    mAccumCount = 0;

    // When our renderer moves around we want to reset accumulation, so stash the scene pointer
    mpScene = pScene;

    // Grab a copy of the current scene's camera matrix (if it exists)
    if (mpScene && mpScene->getActiveCamera())
        mpLastCameraMatrix = mpScene->getActiveCamera()->getViewMatrix();
}

void TemporalAccumulation::onResize(uint32_t width, uint32_t height)
{
    // Need to restart accumulation when we resize
    mAccumCount = 0;
}
