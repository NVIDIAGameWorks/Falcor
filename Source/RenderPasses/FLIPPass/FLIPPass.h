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
#include "FalcorExperimental.h"
#include "Core/Platform/MonitorInfo.h"
#include "Utils/Algorithm/ComputeParallelReduction.h"

using namespace Falcor;

class FLIPPass : public RenderPass
{
public:
    using SharedPtr = std::shared_ptr<FLIPPass>;

    /** Create a new render pass object.
        \param[in] pRenderContext The render context.
        \param[in] dict Dictionary of serialized parameters.
        \return A new object, or an exception is thrown if creation failed.
    */
    static SharedPtr create(RenderContext* pRenderContext = nullptr, const Dictionary& dict = {});

    virtual std::string getDesc() override;
    virtual Dictionary getScriptingDictionary() override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;

private:
    FLIPPass(const Dictionary& dict);
    
    bool                                mEnabled = true;                    ///< Enables FLIP calculation

    bool                                mUseMagma = true;                   ///< Enable to map FLIP result to magma colormap
    uint                                mMonitorWidthPixels;                ///< Horizontal monitor resolution
    float                               mMonitorWidthMeters;                ///< Width of the monitor in meters
    float                               mMonitorDistanceMeters;             ///< Distance of monitor from the viewer in meters

    Texture::SharedPtr                  mpFLIPOutput;                       ///< Internal buffer for high-recision FLIP result
    ComputePass::SharedPtr              mpFLIPPass;                         ///< Compute pass to calculate FLIP

    bool                                mCalculatePerFrameFLIP = false;     ///< Enable to use parallel reduction to calculate FLIP metric across whole frame
    float                               mAverageFLIP;                       ///< Average FLIP value across whole frame
    float                               mMinFLIP;                           ///< Minimum FLIP value across whole frame
    float                               mMaxFLIP;                           ///< Maximum FLIP value across whole frame
    bool                                mUseRealMonitorInfo = false;        ///< When enabled, user-proided monitor data will be overriden by real monitor data from the OS
    
    ComputeParallelReduction::SharedPtr mpParallelReduction;                ///< Helper for parallel reduction on the GPU.
};
