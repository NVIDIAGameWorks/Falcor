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
#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderPassHelpers.h"

using namespace Falcor;

class SimplePostFX : public RenderPass
{
public:
    using SharedPtr = std::shared_ptr<SimplePostFX>;

    static const Info kInfo;

    /** Create a new render pass object.
        \param[in] pRenderContext The render context.
        \param[in] dict Dictionary of serialized parameters.
        \return A new object, or an exception is thrown if creation failed.
    */
    static SharedPtr create(RenderContext* pRenderContext = nullptr, const Dictionary& dict = {});

    virtual Dictionary getScriptingDictionary() override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override {}
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) override {}

    bool    getEnabled() const { return mEnabled; }
    float   getWipe() const { return mWipe; }
    float   getBloomAmount() const { return mBloomAmount; }
    float   getStarAmount() const { return mStarAmount; }
    float   getStarAngle() const { return mStarAngle; }
    float   getVignetteAmount() const { return mVignetteAmount; }
    float   getChromaticAberrationAmount() const { return mChromaticAberrationAmount; }
    float   getBarrelDistortAmount() const { return mBarrelDistortAmount; }
    float3  getSaturationCurve() const { return mSaturationCurve; }
    float3  getColorOffset() const { return mColorOffset; }
    float3  getColorScale() const { return mColorScale; }
    float3  getColorPower() const { return mColorPower; }
    float   getColorOffsetScalar() const { return mColorOffsetScalar; }
    float   getColorScaleScalar() const { return mColorScaleScalar; }
    float   getColorPowerScalar() const { return mColorPowerScalar; }

    void setEnabled(bool e) { mEnabled = e; }
    void setWipe(float v) { mWipe = v; }
    void setBloomAmount(float v) { mBloomAmount = v; }
    void setStarAmount(float v) { mStarAmount = v; }
    void setStarAngle(float v) { mStarAngle = v; }
    void setVignetteAmount(float v) { mVignetteAmount = v; }
    void setChromaticAberrationAmount(float v) { mChromaticAberrationAmount = v; }
    void setBarrelDistortAmount(float v) { mBarrelDistortAmount = v; }
    void setSaturationCurve(float3 v) { mSaturationCurve = v; }
    void setColorOffset(float3 v) { mColorOffset = v; }
    void setColorScale(float3 v) { mColorScale = v; }
    void setColorPower(float3 v) { mColorPower = v; }
    void setColorOffsetScalar(float v) { mColorOffsetScalar = v; }
    void setColorScaleScalar(float v) { mColorScaleScalar = v; }
    void setColorPowerScalar(float v) { mColorPowerScalar = v; }

private:
    SimplePostFX(const Dictionary& dict);

    void preparePostFX(RenderContext* pRenderContext, uint32_t width, uint32_t height);

    const static int kNumLevels = 8;

    RenderPassHelpers::IOSize mOutputSizeSelection = RenderPassHelpers::IOSize::Default;    ///< Selected output size.
    uint2 mFixedOutputSize = { 512, 512 };                                                  ///< Output size in pixels when 'Fixed' size is selected.

    ComputePass::SharedPtr      mpDownsamplePass;
    ComputePass::SharedPtr      mpUpsamplePass;
    ComputePass::SharedPtr      mpPostFXPass;

    Texture::SharedPtr          mpPyramid[kNumLevels + 1];                  ///< Image pyramid, fine to coarse, full res down in steps of 4x (16x area).
    Sampler::SharedPtr          mpLinearSampler;

    float   mWipe = 0.f;                                                    ///< Wipe across to see the effect without fx. 0<=all effect, 1>= disabled.
    bool    mEnabled = true;                                                ///< Enable the entire pass.

    float   mBloomAmount = 0.f;                                             ///< Amount of bloom.
    float   mStarAmount = 0.f;                                              ///< how much of a 6 pointed star to add to the bloom kernel.
    float   mStarAngle = 0.1f;                                              ///< angle of star rays.
    float   mVignetteAmount = 0.f;                                          ///< Amount of circuit vignetting.
    float   mChromaticAberrationAmount = 0.f;                               ///< Amount of radial chromatic aberration.
    float   mBarrelDistortAmount = 0.f;                                     ///< Amount of Barrel distortion.
    float3  mSaturationCurve = float3(1.f, 1.f, 1.f);                       ///< Saturation amount for shadows, midtones and hilights.
    float3  mColorOffset = float3(0.5f, 0.5f, 0.5f);                        ///< Color offset, tints shadows.
    float3  mColorScale = float3(0.5f, 0.5f, 0.5f);                         ///< Color scale, tints hilights.
    float3  mColorPower = float3(0.5f, 0.5f, 0.5f);                         ///< Color power (gamma), tints midtones.
    // the above colors are also offered as scalars for ease of UI and also to set negative colors.
    float   mColorOffsetScalar = 0.f;                                       ///< Luma offset, crushes shadows if negative.
    float   mColorScaleScalar = 0.f;                                        ///< Luma scale, effectively another exposure control.
    float   mColorPowerScalar = 0.f;                                        ///< Luma power, ie a gamma curve.
};
