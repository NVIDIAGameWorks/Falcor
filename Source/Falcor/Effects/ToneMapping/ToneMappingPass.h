/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/
#pragma once
#include "RenderGraph/RenderPass.h"
#include "Core/Program/ProgramVars.h"
#include "RenderGraph/BasePasses/FullScreenPass.h"
#include "Data/Effects/ToneMappingData.h"

namespace Falcor
{
    /** Tone-mapping effect
    */
    class dlldecl ToneMappingPass : public RenderPass
    {
    public:
        using SharedPtr = std::shared_ptr<ToneMappingPass>;
        static const char* kDesc;

        /** The tone-mapping operator to use
        */
        enum class Operator
        {
            Clamp,              ///< Clamp to [0, 1]. Just like LDR
            Linear,             ///< Linear mapping
            Reinhard,           ///< Reinhard operator
            ReinhardModified,   ///< Reinhard operator with maximum white intensity
            HejiHableAlu,       ///< John Hable's ALU approximation of Jim Heji's filmic operator
            HableUc2,           ///< John Hable's filmic tone-mapping used in Uncharted 2
            Aces,               ///< Aces Filmic Tone-Mapping
            Photo,              ///< Photographic tonemapper with manual controls
        };

        /** Create a new object
        */
        static SharedPtr create(RenderContext* pRenderContext, const Dictionary& dict);

        std::string getDesc() override { return kDesc; }
        virtual Dictionary getScriptingDictionary() override;
        virtual RenderPassReflection reflect(const CompileData& compileData) override;
        virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
        virtual void renderUI(Gui::Widgets& widget) override;

        // Scripting functions
        void setOperator(Operator op);
        void setExposureKey(float exposureKey) { mToneMappingData.exposureKey = max(0.001f, exposureKey); }
        void setWhiteMaxLuminance(float maxLuminance) { mToneMappingData.whiteMaxLuminance = maxLuminance; }
        void setLuminanceLod(float lod) { mToneMappingData.luminanceLod = clamp(lod, 0.0f, 16.0f); }
        void setWhiteScale(float whiteScale) { mToneMappingData.whiteScale = max(0.001f, whiteScale); }
        void setExposureValue(float exposureValue);
        void setFilmSpeed(float filmSpeed);
        void setWhitePoint(float whitePoint);
        Operator getOperator() const { return mOperator; }
        float getExposureKey() const { return mToneMappingData.exposureKey; }
        float getWhiteMaxLuminance() const { return mToneMappingData.whiteMaxLuminance; }
        float getLuminanceLod() const { return mToneMappingData.luminanceLod; }
        float getWhiteScale() const { return mToneMappingData.whiteScale; }
        float getExposureValue() { return mExposureValue; }
        float getFilmSpeed() { return mFilmSpeed; }
        float getWhitePoint() { return mWhitePoint; }

    private:
        ToneMappingPass(Operator op);
        void createToneMapPass(Operator op);
        void createLuminancePass();
        void createLuminanceFbo(const Texture::SharedPtr& pSrc);

        void calculateColorTransform();
        void updateConstants();

        Operator mOperator;
        FullScreenPass::SharedPtr mpToneMapPass;
        FullScreenPass::SharedPtr mpLuminancePass;
        Fbo::SharedPtr mpLuminanceFbo;
        Sampler::SharedPtr mpPointSampler;
        Sampler::SharedPtr mpLinearSampler;

        float mExposureValue = 0.0f; // Exposure value (EV).
        float mFilmSpeed = 100.0f;   // Film speed (ISO).
        float mWhitePoint = 6500.0f; // White point (K).

        // Pre-computed fields based on above settings
        float mLinearScale;          // Precomputed linear exposure scaling.
        float3 mSourceWhite;         // Source illuminant in RGB (the white point to which the image is transformed to conform to).

        float3x4 mColorTransform;    // Color balance transform in RGB space (we only use the 3x3 part).

        ToneMappingData mToneMappingData;
    };

#define tonemap_op(a) case ToneMappingPass::Operator::a: return #a
    inline std::string to_string(ToneMappingPass::Operator op)
    {
        switch (op)
        {
            tonemap_op(Clamp);
            tonemap_op(Linear);
            tonemap_op(Reinhard);
            tonemap_op(ReinhardModified);
            tonemap_op(HejiHableAlu);
            tonemap_op(HableUc2);
            tonemap_op(Aces);
            tonemap_op(Photo);
        default:
            should_not_get_here();
            return "";
        }
    }
#undef tonemap_op
}
