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
#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPassHelpers.h"

using namespace Falcor;

/** Temporal accumulation render pass.

    This pass takes a texture as input and writes the temporally accumulated
    result to an output texture. The pass keeps intermediate data internally.

    For accumulating many samples for ground truth rendering etc., fp32 precision
    is not always sufficient. The pass supports higher precision modes using
    either error compensation (Kahan summation) or double precision math.
*/
class AccumulatePass : public RenderPass
{
public:
    using SharedPtr = std::shared_ptr<AccumulatePass>;

    static const Info kInfo;

    virtual ~AccumulatePass() = default;

    static SharedPtr create(RenderContext* pRenderContext = nullptr, const Dictionary& dict = {});

    virtual Dictionary getScriptingDictionary() override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }
    virtual void onHotReload(HotReloadFlags reloaded) override;

    bool isEnabled() const { return mEnabled; }
    void setEnabled(bool enabled);

    // Scripting functions
    void reset();

    enum class Precision : uint32_t
    {
        Double,                 ///< Standard summation in double precision.
        Single,                 ///< Standard summation in single precision.
        SingleCompensated,      ///< Compensated summation (Kahan summation) in single precision.
    };

protected:
    AccumulatePass(const Dictionary& dict);
    void prepareAccumulation(RenderContext* pRenderContext, uint32_t width, uint32_t height);
    void accumulate(RenderContext* pRenderContext, const Texture::SharedPtr& pSrc, const Texture::SharedPtr& pDst);

    // Internal state
    Scene::SharedPtr            mpScene;                        ///< The current scene (or nullptr if no scene).
    std::map<Precision, ComputeProgram::SharedPtr> mpProgram;   ///< Accumulation programs, one per mode.
    ComputeVars::SharedPtr      mpVars;                         ///< Program variables.
    ComputeState::SharedPtr     mpState;
    FormatType                  mSrcType;                       ///< Format type of the source that gets accumulated.

    uint32_t                    mFrameCount = 0;                ///< Number of accumulated frames. This is reset upon changes.
    uint2                       mFrameDim = { 0, 0 };           ///< Current frame dimension in pixels.
    Texture::SharedPtr          mpLastFrameSum;                 ///< Last frame running sum. Used in Single and SingleKahan mode.
    Texture::SharedPtr          mpLastFrameCorr;                ///< Last frame running compensation term. Used in SingleKahan mode.
    Texture::SharedPtr          mpLastFrameSumLo;               ///< Last frame running sum (lo bits). Used in Double mode.
    Texture::SharedPtr          mpLastFrameSumHi;               ///< Last frame running sum (hi bits). Used in Double mode.

    // UI variables
    bool                        mEnabled = true;                ///< True if accumulation is enabled.
    bool                        mAutoReset = true;              ///< Reset accumulation automatically upon scene changes, refresh flags, and/or subframe count.
    Precision                   mPrecisionMode = Precision::Single;
    uint32_t                    mSubFrameCount = 0;             ///< Number of frames to accumulate before reset. Useful for generating references.
    uint32_t                    mMaxAccumulatedFrames = 0;      ///< Number of frames to accumulate before weights become constant. Useful for noise comparisons.

    ResourceFormat              mOutputFormat = ResourceFormat::Unknown;                    ///< Output format (uses default when set to ResourceFormat::Unknown).
    RenderPassHelpers::IOSize   mOutputSizeSelection = RenderPassHelpers::IOSize::Default;  ///< Selected output size.
    uint2                       mFixedOutputSize = { 512, 512 };                            ///< Output size in pixels when 'Fixed' size is selected.
};
