/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "Utils/Sampling/SampleGenerator.h"
#include "Utils/Debug/PixelDebug.h"
#include "Experimental/Scene/Lights/EnvMapSampler.h"
#include "Experimental/Scene/Lights/EmissiveUniformSampler.h"
#include "Experimental/Scene/Lights/LightBVHSampler.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "PathTracerParams.slang"
#include "PixelStats.h"

namespace Falcor
{
    /** Base class for path tracers.
    */
    class dlldecl PathTracer : public RenderPass
    {
    public:
        using SharedPtr = std::shared_ptr<PathTracer>;

        virtual Dictionary getScriptingDictionary() override;
        virtual RenderPassReflection reflect(const CompileData& compileData) override;
        virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
        virtual void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) override;
        virtual void renderUI(Gui::Widgets& widget) override;
        virtual bool onMouseEvent(const MouseEvent& mouseEvent) override;
        virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

    protected:
        PathTracer(const Dictionary& dict, const ChannelList& outputs);

        virtual void recreateVars() {}

        void validateParameters();
        bool initLights(RenderContext* pRenderContext);
        bool updateLights(RenderContext* pRenderContext);
        uint32_t maxRaysPerPixel() const;
        bool beginFrame(RenderContext* pRenderContext, const RenderData& renderData);
        void endFrame(RenderContext* pRenderContext, const RenderData& renderData);
        bool renderSamplingUI(Gui::Widgets& widget);
        void renderLoggingUI(Gui::Widgets& widget);

        virtual void setStaticParams(Program* pProgram) const;

        // Internal state
        Scene::SharedPtr                    mpScene;                        ///< Current scene.

        SampleGenerator::SharedPtr          mpSampleGenerator;              ///< GPU sample generator.
        EmissiveLightSampler::SharedPtr     mpEmissiveSampler;              ///< Emissive light sampler or nullptr if disabled.
        EnvMapSampler::SharedPtr            mpEnvMapSampler;                ///< Environment map sampler or nullptr if disabled.

        PixelStats::SharedPtr               mpPixelStats;                    ///< Utility class for collecting pixel stats.
        PixelDebug::SharedPtr               mpPixelDebug;                    ///< Utility class for pixel debugging (print in shaders).

        ChannelList                         mInputChannels;                 ///< Render pass inputs.
        const ChannelList                   mOutputChannels;                ///< Render pass outputs.

        // Configuration
        PathTracerParams                    mSharedParams;                  ///< Host/device shared rendering parameters.
        uint32_t                            mSelectedSampleGenerator = SAMPLE_GENERATOR_DEFAULT;            ///< Which pseudorandom sample generator to use.
        EmissiveLightSamplerType            mSelectedEmissiveSampler = EmissiveLightSamplerType::LightBVH;  ///< Which emissive light sampler to use.

        EmissiveUniformSampler::Options     mUniformSamplerOptions;         ///< Current options for the uniform sampler.
        LightBVHSampler::Options            mLightBVHSamplerOptions;        ///< Current options for the light BVH sampler.

        // Runtime data
        bool                                mOptionsChanged = false;        ///< True if the config has changed since last frame.
        bool                                mUseAnalyticLights = false;     ///< True if analytic lights should be used for the current frame. See compile-time constant in StaticParams.slang.
        bool                                mUseEnvLight = false;           ///< True if env map light should be used for the current frame. See compile-time constant in StaticParams.slang.
        bool                                mUseEmissiveLights = false;     ///< True if emissive lights should be taken into account. See compile-time constant in StaticParams.slang.
        bool                                mUseEmissiveSampler = false;    ///< True if emissive light sampler should be used for the current frame. See compile-time constant in StaticParams.slang.
        uint32_t                            mMaxRaysPerPixel = 0;           ///< Maximum number of rays per pixel that will be traced. This is computed based on the current configuration.
        bool                                mIsRayFootprintSupported = true;       ///< Globally enable/disable ray footprint. Requires v-buffer. Set to false if any requirement is not met.

        // Scripting
    #define serialize(var) \
        if constexpr (!loadFromDict) dict[#var] = var; \
        else if (dict.keyExists(#var)) { if constexpr (std::is_same<decltype(var), std::string>::value) var = (const std::string &)dict[#var]; else var = dict[#var]; vars.emplace(#var); }

        template<bool loadFromDict, typename DictType>
        void serializePass(DictType& dict)
        {
            std::unordered_set<std::string> vars;

            // Add variables here that should be serialized to/from the dictionary.
            serialize(mSharedParams);
            serialize(mSelectedSampleGenerator);
            serialize(mSelectedEmissiveSampler);
            serialize(mUniformSamplerOptions);
            serialize(mLightBVHSamplerOptions);

            if constexpr (loadFromDict)
            {
                for (const auto& [key, value] : dict)
                {
                    if (vars.find(key) == vars.end()) logWarning("Unknown field '" + key + "' in a PathTracer dictionary");
                }
            }
        }
    #undef serialize
    };

}
