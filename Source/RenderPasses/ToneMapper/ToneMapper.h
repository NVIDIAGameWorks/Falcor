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
#include "ToneMapperParams.slang"
#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/BasePasses/FullScreenPass.h"

using namespace Falcor;

class ToneMapper : public RenderPass
{
public:
    using SharedPtr = std::shared_ptr<ToneMapper>;
    using Operator = ToneMapperOperator;

    static const Info kInfo;

    enum class ExposureMode
    {
        AperturePriority,       // Keep aperture constant when modifying EV
        ShutterPriority,        // Keep shutter constant when modifying EV
    };

    /** Create a new object
    */
    static SharedPtr create(RenderContext* pRenderContext, const Dictionary& dict);

    virtual Dictionary getScriptingDictionary() override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const std::shared_ptr<Scene>& pScene) override;

    // Scripting functions
    void setExposureCompensation(float exposureCompensation);
    void setAutoExposure(bool autoExposure);
    void setExposureValue(float exposureValue);
    void setFilmSpeed(float filmSpeed);
    void setWhiteBalance(bool whiteBalance);
    void setWhitePoint(float whitePoint);
    void setOperator(Operator op);
    void setClamp(bool clamp);
    void setWhiteMaxLuminance(float maxLuminance);
    void setWhiteScale(float whiteScale);
    void setFNumber(float fNumber);
    void setShutter(float shutter);
    void setExposureMode(ExposureMode mode);

    float getExposureCompensation() const { return mExposureCompensation; }
    bool getAutoExposure() const { return mAutoExposure; }
    float getExposureValue() const { return mExposureValue; }
    float getFilmSpeed() const { return mFilmSpeed; }
    bool getWhiteBalance() const { return mWhiteBalance; }
    float getWhitePoint() { return mWhitePoint; }
    Operator getOperator() const { return mOperator; }
    bool getClamp() const { return mClamp; }
    float getWhiteMaxLuminance() const { return mWhiteMaxLuminance; }
    float getWhiteScale() const { return mWhiteScale; }
    float getFNumber() const { return mFNumber; }
    float getShutter() const { return mShutter; }
    ExposureMode getExposureMode() const { return mExposureMode; }

private:
    ToneMapper(const Dictionary& dict);
    void parseDictionary(const Dictionary& dict);

    void createToneMapPass();
    void createLuminancePass();
    void createLuminanceFbo(const Texture::SharedPtr& pSrc);

    void updateWhiteBalanceTransform();
    void updateColorTransform();

    void updateExposureValue();

    FullScreenPass::SharedPtr mpToneMapPass;
    FullScreenPass::SharedPtr mpLuminancePass;
    Fbo::SharedPtr mpLuminanceFbo;
    Sampler::SharedPtr mpPointSampler;
    Sampler::SharedPtr mpLinearSampler;

    RenderPassHelpers::IOSize mOutputSizeSelection = RenderPassHelpers::IOSize::Default;    ///< Selected output size.
    ResourceFormat mOutputFormat = ResourceFormat::Unknown;                                 ///< Output format (uses default when set to ResourceFormat::Unknown).
    uint2 mFixedOutputSize = { 512, 512 };                                                  ///< Output size in pixels when 'Fixed' size is selected.

    bool mUseSceneMetadata = true;      ///< Use scene metadata for setting up tonemapper when loading a scene.

    float mExposureCompensation = 0.f;  ///< Exposure compensation (in F-stops).
    bool mAutoExposure = false;         ///< Enable auto exposure.
    float mExposureValue = 0.0f;        ///< Exposure value (EV), derived from fNumber, shutter, and film speed; only used when auto exposure is disabled.
    float mFilmSpeed = 100.f;           ///< Film speed (ISO), only used when auto exposure is disabled.
    float mFNumber = 1.f;               ///< Lens speed
    float mShutter = 1.f;               ///< Reciprocal of shutter time

    bool mWhiteBalance = false;         ///< Enable white balance.
    float mWhitePoint = 6500.0f;        ///< White point (K).

    Operator mOperator = Operator::Aces;///< Tone mapping operator.
    bool mClamp = true;                 ///< Clamp output to [0,1].

    float mWhiteMaxLuminance = 1.0f;    ///< Parameter used in ModifiedReinhard operator.
    float mWhiteScale = 11.2f;          ///< Parameter used in Uc2Hable operator.

    // Pre-computed fields based on above settings
    rmcv::mat3 mWhiteBalanceTransform;    ///< Color balance transform in RGB space.
    float3 mSourceWhite;                ///< Source illuminant in RGB (the white point to which the image is transformed to conform to).
    rmcv::mat3 mColorTransform;           ///< Final color transform with exposure value baked in.

    bool mRecreateToneMapPass = true;
    bool mUpdateToneMapPass = true;

    ExposureMode mExposureMode = ExposureMode::AperturePriority;
};
