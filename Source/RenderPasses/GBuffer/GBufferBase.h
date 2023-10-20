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
#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderPassHelpers.h"

using namespace Falcor;

/**
 * Base class for the different types of G-buffer passes (including V-buffer).
 */
class GBufferBase : public RenderPass
{
public:
    enum class SamplePattern : uint32_t
    {
        Center,
        DirectX,
        Halton,
        Stratified,
    };

    FALCOR_ENUM_INFO(
        SamplePattern,
        {
            {SamplePattern::Center, "Center"},
            {SamplePattern::DirectX, "DirectX"},
            {SamplePattern::Halton, "Halton"},
            {SamplePattern::Stratified, "Stratified"},
        }
    );

    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual Properties getProperties() const override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;

protected:
    GBufferBase(ref<Device> pDevice) : RenderPass(pDevice) {}
    virtual void parseProperties(const Properties& props);
    virtual void setCullMode(RasterizerState::CullMode mode) { mCullMode = mode; }
    void updateFrameDim(const uint2 frameDim);
    void updateSamplePattern();
    ref<Texture> getOutput(const RenderData& renderData, const std::string& name) const;

    // Internal state

    ref<Scene> mpScene;
    /// Sample generator for camera jitter.
    ref<CPUSampleGenerator> mpSampleGenerator;

    /// Frames rendered since last change of scene. This is used as random seed.
    uint32_t mFrameCount = 0;
    /// Current frame dimension in pixels. Note this may be different from the window size.
    uint2 mFrameDim = {};
    float2 mInvFrameDim = {};
    ResourceFormat mVBufferFormat = HitInfo::kDefaultFormat;

    // UI variables

    /// Selected output size.
    RenderPassHelpers::IOSize mOutputSizeSelection = RenderPassHelpers::IOSize::Default;
    /// Output size in pixels when 'Fixed' size is selected.
    uint2 mFixedOutputSize = {512, 512};
    /// Which camera jitter sample pattern to use.
    SamplePattern mSamplePattern = SamplePattern::Center;
    /// Sample count for camera jitter.
    uint32_t mSampleCount = 16;
    /// Enable alpha test.
    bool mUseAlphaTest = true;
    /// Adjust shading normals.
    bool mAdjustShadingNormals = true;
    /// Force cull mode for all geometry, otherwise set it based on the scene.
    bool mForceCullMode = false;
    /// Cull mode to use for when mForceCullMode is true.
    RasterizerState::CullMode mCullMode = RasterizerState::CullMode::Back;

    /// Indicates whether any options that affect the output have changed since last frame.
    bool mOptionsChanged = false;
};

FALCOR_ENUM_REGISTER(GBufferBase::SamplePattern);
