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
#include "Core/Platform/MonitorInfo.h"
#include "Utils/Algorithm/ParallelReduction.h"
#include "ToneMappers.slang"

using namespace Falcor;

class FLIPPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(
        FLIPPass,
        "FLIPPass",
        {"FLIP Metric Pass.\n\n"
         "If the input has high dynamic range, check the \"Compute HDR-FLIP\" box below.\n\n"
         "The errorMapDisplay shows the FLIP error map. "
         "When HDR-FLIP is computed, the user may also show the HDR-FLIP exposure map.\n\n"
         "When \"List all output\" is checked, the user may also store the errorMap. "
         "This is a high-precision, linear buffer which is transformed to sRGB before display. "
         "NOTE: This sRGB transform will make the displayed output look different compared "
         "to the errorMapDisplay. The transform is only added before display, however, "
         "and will NOT affect the output when it is saved to disk."}
    );

    static ref<FLIPPass> create(ref<Device> pDevice, const Properties& props) { return make_ref<FLIPPass>(pDevice, props); }

    FLIPPass(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;

protected:
    void updatePrograms();
    void computeExposureParameters(const float Ymedian, const float Ymax);
    void parseProperties(const Properties& props);

private:
    /// Enables FLIP calculation.
    bool mEnabled = true;

    /// Enable to map FLIP result to magma colormap.
    bool mUseMagma = true;
    /// Enable to clamp FLIP input to the expected range ([0,1] for LDR-FLIP and [0, inf) for HDR-FLIP).
    bool mClampInput = false;
    /// Horizontal monitor resolution.
    uint mMonitorWidthPixels;
    /// Width of the monitor in meters.
    float mMonitorWidthMeters;
    /// Distance of monitor from the viewer in meters.
    float mMonitorDistanceMeters;

    /// Enable to compute HDR-FLIP.
    bool mIsHDR = false;
    /// Enable to choose custom HDR-FLIP exposure parameters (start exposure, stop exposure, and number of exposures).
    bool mUseCustomExposureParameters = false;
    /// Mode for controlling adaptive sampling.
    FLIPToneMapperType mToneMapper = FLIPToneMapperType::ACES;
    /// Start exposure used for HDR-FLIP.
    float mStartExposure = 0.0f;
    /// Stop exposure used for HDR-FLIP.
    float mStopExposure = 0.0f;
    /// Exposure delta used for HDR-FLIP (startExposure + (numExposures - 1) * exposureDelta = stopExposure).
    float mExposureDelta = 0.0f;
    /// Number of exposures used for HDR-FLIP.
    uint32_t mNumExposures = 2;

    /// Internal buffer for temporary display output buffer.
    ref<Texture> mpFLIPErrorMapDisplay;
    /// Internal buffer for the HDR-FLIP exposure map.
    ref<Texture> mpExposureMapDisplay;
    /// Internal buffer for temporary luminance.
    ref<Buffer> mpLuminance;
    /// Compute pass to calculate FLIP.
    ref<ComputePass> mpFLIPPass;
    /// Compute pass for computing the luminance of an image.
    ref<ComputePass> mpComputeLuminancePass;
    /// Helper for parallel reduction on the GPU.
    std::unique_ptr<ParallelReduction> mpParallelReduction;

    /// Enable to use parallel reduction to compute FLIP mean/min/max across whole frame.
    bool mComputePooledFLIPValues = false;
    /// Average FLIP value across whole frame.
    float mAverageFLIP;
    /// Minimum FLIP value across whole frame.
    float mMinFLIP;
    /// Maximum FLIP value across whole frame.
    float mMaxFLIP;
    /// When enabled, user-proided monitor data will be overriden by real monitor data from the OS.
    bool mUseRealMonitorInfo = false;
    /// Recompilation flag.
    bool mRecompile = true;
};
