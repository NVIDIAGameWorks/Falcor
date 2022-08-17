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
#include <memory>

#define NGX_ENABLE_DEPRECATED_GET_PARAMETERS
#include <nvsdk_ngx.h>

// Forward declarations from NGX library.
struct NVSDK_NGX_Parameter;
struct NVSDK_NGX_Handle;

namespace Falcor
{
    // This is a wrapper around the NGX functionality for DLSS.  It is seperated to provide focus to the calls specific to NGX for code sample purposes.
    class NGXWrapper
    {
    public:
        struct OptimalSettings
        {
            float sharpness;
            uint2 optimalRenderSize;
            uint2 minRenderSize;
            uint2 maxRenderSize;
        };

        /** Constructor.
            Throws an exception if unable to initialize NGX.
        */
        NGXWrapper(Device* pDevice, const wchar_t* logFolder=L".");
        ~NGXWrapper();

        /** Query optimal DLSS settings for a given resolution and performance/quality profile.
        */
        OptimalSettings queryOptimalSettings(uint2 displaySize, NVSDK_NGX_PerfQuality_Value perfQuality) const;

        /** Initialize DLSS.
            Throws an exception if unable to initialize.
        */
        void initializeDLSS(
            RenderContext* pRenderContext,
            uint2 maxRenderSize,
            uint2 displayOutSize,
            Texture* pTarget,
            bool isContentHDR,
            bool depthInverted,
            NVSDK_NGX_PerfQuality_Value perfQuality = NVSDK_NGX_PerfQuality_Value_MaxPerf);

        /** Release DLSS.
        */
        void releaseDLSS();

        /** Checks if DLSS is initialized.
        */
        bool isDLSSInitialized() const { return mpFeature != nullptr; }

        /** Evaluate DLSS.
        */
        bool evaluateDLSS(
            RenderContext* pRenderContext,
            Texture* pUnresolvedColor,
            Texture* pResolvedColor,
            Texture* pMotionVectors,
            Texture* pDepth,
            Texture* pExposure,
            bool resetAccumulation = false,
            float sharpness = 0.0f,
            float2 jitterOffset = { 0.f, 0.f },
            float2 motionVectorScale = { 1.f, 1.f }) const;

    private:
        void initializeNGX(const wchar_t* logFolder);
        void shutdownNGX();

        Device* mpDevice = nullptr;
        bool mInitialized = false;

        NVSDK_NGX_Parameter* mpParameters = nullptr;
        NVSDK_NGX_Handle* mpFeature = nullptr;
    };
}
