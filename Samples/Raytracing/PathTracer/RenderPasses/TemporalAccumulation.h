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
#pragma once
#include "Falcor.h"
#include "FalcorExperimental.h"

using namespace Falcor;

/** This render pass takes an input buffer, accumulates/averages that input (over multiple frames)
    into an internal buffer, and writes the (current) averaged accumulation buffer to the output.
    Accumulation is reset whenever the camera moves or when the other passes mark the dirty bit "_isDirty"
    in the shared Python dictionary
*/
class TemporalAccumulation : public RenderPass, inherit_shared_from_this<RenderPass, TemporalAccumulation>
{
public:
    using SharedPtr = std::shared_ptr<TemporalAccumulation>;

    /** Get a string describing what the pass is doing
    */
    virtual std::string getDesc() override { return "Accumulates its input buffer over time."; }

    /** Instantiate our pass.  The input Python dictionary is where you can extract pass parameters
    */
    static SharedPtr create(const Dictionary& params = {});

    /** Defines the inputs/outputs required for this render pass
    */
    virtual RenderPassReflection reflect(void) const override;

    /** Run our accumulation buffering
    */
    virtual void execute(RenderContext* pContext, const RenderData* pRenderData) override;

    /** Display a GUI allowing disabling accumulation and printing a running tally of frames accumulated
    */
    virtual void renderUI(Gui* pGui, const char* uiGroup) override;

    /** Grab the current scene so we can determine when the camera moves
    */
    virtual void setScene(const std::shared_ptr<Scene>& pScene) override;

    /** Check when the screen resizes, since we need to restart accumulation then, too
    */
    virtual void onResize(uint32_t width, uint32_t height) override;

    /** Serialize the render pass parameters out to a python dictionary
    */
    virtual Dictionary getScriptingDictionary() const override;

private:
    TemporalAccumulation() : RenderPass("TemporalAccumulation") {}

    void initialize(const Dictionary& dict);

    // A helper utility to determine if the current scene (if any) has had any camera motion
    bool hasCameraMoved();

    // State for our accumulation shader
    GraphicsVars::SharedPtr     mpVars;
    GraphicsProgram::SharedPtr  mpProgram;
    GraphicsState::SharedPtr    mpState;
    FullScreenPass::UniquePtr   mpPass;

    Fbo::SharedPtr              mpInternalFbo;

    // We stash a copy of our current scene.  Why?  To detect if changes have occurred.
    Scene::SharedPtr    mpScene;
    mat4                mpLastCameraMatrix;

    // Is our accumulation enabled?
    bool mDoAccumulation = true;

    // How many frames have we accumulated so far?
    uint32_t mAccumCount = 0;

    // Some common pass bookkeeping
    bool        mIsInitialized = false;
    bool        mDirtyLastFrame = false;
    Dictionary  mDict; ///< Our shared Python dictionary for pass communication (extracted during onInitialize())
};
