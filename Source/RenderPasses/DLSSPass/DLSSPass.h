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
#include "NGXWrapper.h"

using namespace Falcor;

class DLSSPass : public RenderPass
{
public:
    using SharedPtr = std::shared_ptr<DLSSPass>;

    static const Info kInfo;

    enum class Profile : uint32_t
    {
        MaxPerf,
        Balanced,
        MaxQuality,
    };

    enum class MotionVectorScale : uint32_t
    {
        Absolute,   ///< Motion vectors are provided in absolute screen space length (pixels).
        Relative,   ///< Motion vectors are provided in relative screen space length (pixels divided by screen width/height).
    };

    static SharedPtr create(RenderContext* pRenderContext, const Dictionary& dict);

    virtual Dictionary getScriptingDictionary() override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void setScene(RenderContext* pRenderContext, const std::shared_ptr<Scene>& pScene) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;

private:
    DLSSPass(const Dictionary& dict);

    void initializeDLSS(RenderContext* pRenderContext);
    void executeInternal(RenderContext* pRenderContext, const RenderData& renderData);

    // Options
    bool                        mEnabled = true;
    Profile                     mProfile = Profile::Balanced;
    MotionVectorScale           mMotionVectorScale = MotionVectorScale::Absolute;
    bool                        mIsHDR = true;
    float                       mSharpness = 0.f;
    float                       mExposure = 0.f;
    bool                        mExposureUpdated = true;

    bool                        mRecreate = true;
    uint2                       mInputSize = {};            ///< Input size in pixels.
    uint2                       mDLSSOutputSize = {};       ///< DLSS output size in pixels.
    uint2                       mPassOutputSize = {};       ///< Pass output size in pixels. If different from DLSS output size, the image gets bilinearly resampled.
    RenderPassHelpers::IOSize   mOutputSizeSelection = RenderPassHelpers::IOSize::Default; ///< Selected output size.

    Scene::SharedPtr            mpScene;
    Texture::SharedPtr          mpOutput;                   ///< Internal output buffer. This is used if format/size conversion upon output is needed.
    Texture::SharedPtr          mpExposure;                 ///< Texture of size 1x1 holding exposure value.

    std::unique_ptr<NGXWrapper> mpNGXWrapper;
};
