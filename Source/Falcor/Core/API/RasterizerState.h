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
#include "Handles.h"
#include "Core/Macros.h"
#include <memory>

namespace Falcor
{
    /** Rasterizer state
    */
    class FALCOR_API RasterizerState
    {
    public:
        using SharedPtr = std::shared_ptr<RasterizerState>;
        using SharedConstPtr = std::shared_ptr<const RasterizerState>;

        /** Cull mode
        */
        enum class CullMode : uint32_t
        {
            None,   ///< No culling
            Front,  ///< Cull front-facing primitives
            Back,   ///< Cull back-facing primitives
        };

        /** Polygon fill mode
        */
        enum class FillMode
        {
            Wireframe,   ///< Wireframe
            Solid        ///< Solid
        };

        /** Rasterizer state descriptor
        */
        class FALCOR_API Desc
        {
        public:
            friend class RasterizerState;

            /** Set the cull mode
            */
            Desc& setCullMode(CullMode mode) { mCullMode = mode; return *this; }

            /** Set the fill mode
            */
            Desc& setFillMode(FillMode mode) { mFillMode = mode; return *this; }

            /** Determines how to interpret triangle direction.
                \param isFrontCCW If true, a triangle is front-facing if is vertices are counter-clockwise. If false, the opposite.
            */
            Desc& setFrontCounterCW(bool isFrontCCW) { mIsFrontCcw = isFrontCCW; return *this; }

            /** Set the depth-bias. The depth bias is calculated as
                \code
                bias = (float)depthBias * r + slopeScaledBias * maxDepthSlope
                \endcode
                where r is the minimum representable value in the depth buffer and maxDepthSlope is the maximum of the horizontal and vertical slopes of the depth value in the pixel.\n
                See <a href="https://msdn.microsoft.com/en-us/library/windows/desktop/cc308048%28v=vs.85%29.aspx">the DX documentation</a> for depth bias explanation.
            */
            Desc& setDepthBias(int32_t depthBias, float slopeScaledBias) { mSlopeScaledDepthBias = slopeScaledBias; mDepthBias = depthBias; return *this; }

            /** Determines weather to clip or cull the vertices based on depth
                \param clampDepth If true, clamp depth value to the viewport extent. If false, clip primitives to near/far Z-planes
            */
            Desc& setDepthClamp(bool clampDepth) { mClampDepth = clampDepth; return *this; }

            /** Enable/disable anti-aliased lines. Actual anti-aliasing algorithm is implementation dependent, but usually uses quadrilateral lines.
            */
            Desc& setLineAntiAliasing(bool enableLineAA) { mEnableLinesAA = enableLineAA; return *this; };

            /** Enable/disable scissor test
            */
            Desc& setScissorTest(bool enabled) { mScissorEnabled = enabled ; return *this; }

            /** Enable/disable conservative rasterization
            */
            Desc& setConservativeRasterization(bool enabled) { mConservativeRaster = enabled; return *this; }

            /** Set the forced sample count. Useful when using UAV
            */
            Desc& setForcedSampleCount(uint32_t samples) { mForcedSampleCount = samples; return *this; }

        protected:
            CullMode mCullMode = CullMode::Back;
            FillMode mFillMode = FillMode::Solid;
            bool     mIsFrontCcw = true;
            float    mSlopeScaledDepthBias = 0;
            int32_t  mDepthBias = 0;
            bool     mClampDepth = false;
            bool     mScissorEnabled = false;
            bool     mEnableLinesAA = true;
            uint32_t mForcedSampleCount = 0;
            bool     mConservativeRaster = false;
        };

        ~RasterizerState();

        /** Create a new rasterizer state.
            \param[in] desc Rasterizer state descriptor.
            \return A new object, or throws an exception if creation failed.
        */
        static SharedPtr create(const Desc& desc);

        /** Get the cull mode
        */
        CullMode getCullMode() const { return mDesc.mCullMode; }
        /** Get the fill mode
        */
        FillMode getFillMode() const { return mDesc.mFillMode; }
        /** Check what is the winding order for triangles to be considered front-facing
        */
        bool isFrontCounterCW() const{ return mDesc.mIsFrontCcw; }
        /** Get the slope-scaled depth bias
        */
        float getSlopeScaledDepthBias() const { return mDesc.mSlopeScaledDepthBias; }
        /** Get the depth bias
        */
        int32_t getDepthBias() const { return mDesc.mDepthBias; }
        /** Check if depth clamp is enabled
        */
        bool isDepthClampEnabled() const { return mDesc.mClampDepth; }
        /** Check if scissor test is enabled
        */
        bool isScissorTestEnabled() const { return mDesc.mScissorEnabled; }
        /** Check if anti-aliased lines are enabled
        */
        bool isLineAntiAliasingEnabled() const { return mDesc.mEnableLinesAA; }

        /** Check if conservative rasterization is enabled
        */
        bool isConservativeRasterizationEnabled() const { return mDesc.mConservativeRaster; }

        /** Get the forced sample count
        */
        uint32_t getForcedSampleCount() const { return mDesc.mForcedSampleCount; }

        /** Get the API handle
        */
        const RasterizerStateHandle& getApiHandle() const;
    private:
        RasterizerStateHandle mApiHandle;
        RasterizerState(const Desc& Desc) : mDesc(Desc) {}
        Desc mDesc;
    };
}
