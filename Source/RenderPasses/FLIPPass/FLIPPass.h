/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include "Core/Platform/MonitorInfo.h"
#include "Utils/Algorithm/ComputeParallelReduction.h"
#include "ToneMappers.slang"

using namespace Falcor;

class FLIPPass : public RenderPass
{
public:
    using SharedPtr = std::shared_ptr<FLIPPass>;

    static const Info kInfo;

    /** Create a new render pass object.
        \param[in] pRenderContext The render context.
        \param[in] dict Dictionary of serialized parameters.
        \return A new object, or an exception is thrown if creation failed.
    */
    static SharedPtr create(RenderContext* pRenderContext = nullptr, const Dictionary& dict = {});

    virtual Dictionary getScriptingDictionary() override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;

    static void registerBindings(pybind11::module& m);

protected:
    void updatePrograms();
    void computeExposureParameters(const float Ymedian, const float Ymax);
    void parseDictionary(const Dictionary& dict);

private:
    FLIPPass(const Dictionary& dict);

    bool                                mEnabled = true;                        ///< Enables FLIP calculation.

    bool                                mUseMagma = true;                       ///< Enable to map FLIP result to magma colormap.
    bool                                mClampInput = false;                    ///< Enable to clamp FLIP input to the expected range ([0,1] for LDR-FLIP and [0, inf) for HDR-FLIP).
    uint                                mMonitorWidthPixels;                    ///< Horizontal monitor resolution.
    float                               mMonitorWidthMeters;                    ///< Width of the monitor in meters.
    float                               mMonitorDistanceMeters;                 ///< Distance of monitor from the viewer in meters.

    bool                                mIsHDR = false;                         ///< Enable to compute HDR-FLIP.
    bool                                mUseCustomExposureParameters = false;   ///< Enable to choose custom HDR-FLIP exposure parameters (start exposure, stop exposure, and number of exposures).
    FLIPToneMapperType                  mToneMapper = FLIPToneMapperType::ACES; ///< Mode for controlling adaptive sampling.
    float                               mStartExposure = 0.0f;                  ///< Start exposure used for HDR-FLIP.
    float                               mStopExposure = 0.0f;                   ///< Stop exposure used for HDR-FLIP.
    float                               mExposureDelta = 0.0f;                  ///< Exposure delta used for HDR-FLIP (startExposure + (numExposures - 1) * exposureDelta = stopExposure).
    uint32_t                            mNumExposures = 2;                      ///< Number of exposures used for HDR-FLIP.

    Texture::SharedPtr                  mpFLIPErrorMapDisplay;                  ///< Internal buffer for temporary display output buffer.
    Texture::SharedPtr                  mpExposureMapDisplay;                   ///< Internal buffer for the HDR-FLIP exposure map.
    Buffer::SharedPtr                   mpLuminance;                            ///< Internal buffer for temporary luminance.
    ComputePass::SharedPtr              mpFLIPPass;                             ///< Compute pass to calculate FLIP.
    ComputePass::SharedPtr              mpComputeLuminancePass;                 ///< Compute pass for computing the luminance of an image.
    ComputeParallelReduction::SharedPtr mpParallelReduction;                    ///< Helper for parallel reduction on the GPU.

    bool                                mComputePooledFLIPValues = false;       ///< Enable to use parallel reduction to compute FLIP mean/min/max across whole frame.
    float                               mAverageFLIP;                           ///< Average FLIP value across whole frame.
    float                               mMinFLIP;                               ///< Minimum FLIP value across whole frame.
    float                               mMaxFLIP;                               ///< Maximum FLIP value across whole frame.
    bool                                mUseRealMonitorInfo = false;            ///< When enabled, user-proided monitor data will be overriden by real monitor data from the OS.
    bool                                mRecompile = true;                      ///< Recompilation flag.
};
