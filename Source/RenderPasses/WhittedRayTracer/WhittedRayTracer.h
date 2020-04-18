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
 **************************************************************************/
#pragma once
#include "Falcor.h"
#include "FalcorExperimental.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "Experimental/Scene/Lights/EnvProbe.h"
#include "Experimental/Scene/Material/TexLODTypes.slang"  // Using the enum with Mip0, RayCones, etc

using namespace Falcor;

/** Whitted ray tracer.

    This pass implements the simplest possible Whitted ray tracer.
*/
class WhittedRayTracer : public RenderPass, public inherit_shared_from_this<RenderPass, WhittedRayTracer>
{
public:
    using SharedPtr = std::shared_ptr<WhittedRayTracer>;
    using inherit_shared_from_this::shared_from_this;


    static SharedPtr create(RenderContext* pRenderContext = nullptr, const Dictionary& dict = {});

    virtual std::string getDesc() override { return "Whitted ray tracer"; }
    virtual Dictionary getScriptingDictionary() override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

    void setTexLODMode(const TexLODMode mode)   { mTexLODMode = mode; }
    TexLODMode getTexLODMode() const            { return mTexLODMode; }

private:
    WhittedRayTracer(const Dictionary& dict);

    void prepareVars();
    void setStaticParams(RtProgram* pProgram) const;

    // Internal state
    Scene::SharedPtr            mpScene;                                    ///< Current scene.
    SampleGenerator::SharedPtr  mpSampleGenerator;                          ///< GPU sample generator.
    EnvProbe::SharedPtr         mpEnvProbe;                                 ///< Environment map sampling (if used).
    std::string                 mEnvProbeFilename;                          ///< Name of loaded environment map (stripped of full path).

    ChannelList                 mInputChannels;
    Gui::DropdownList           mTexLODModes;

    uint                        mMaxBounces = 3;                            ///< Max number of indirect bounces (0 = none).
    int                         mUseAnalyticLights = true;                  ///< Use built-in analytic lights.
    int                         mUseEmissiveLights = true;                  ///< Use emissive geometry as light sources.
    int                         mUseEnvLight = false;                       ///< Use environment map as light source (if loaded).
    int                         mUseEnvBackground = true;                   ///< Use environment map as background (if loaded).
    TexLODMode                  mTexLODMode = TexLODMode::Mip0;             ///< Which texture LOD mode to use
    bool                        mUsingRasterizedGBuffer = true;             ///< Set by the Python file (whether rasterized GBUffer or ray traced GBuffer is used)
    // Runtime data
    uint                        mFrameCount = 0;                            ///< Frame count since scene was loaded.
    bool                        mOptionsChanged = false;

    // Ray tracing program.
    struct
    {
        RtProgram::SharedPtr pProgram;
        RtProgramVars::SharedPtr pVars;
    } mTracer;

    // Scripting
#define serialize(var) \
    if constexpr (!loadFromDict) dict[#var] = var; \
    else if (dict.keyExists(#var)) { if constexpr (std::is_same<decltype(var), std::string>::value) var = (const std::string &)dict[#var]; else var = dict[#var]; vars.emplace(#var); }

    template<bool loadFromDict, typename DictType>
    void serializePass(DictType& dict)
    {
        std::unordered_set<std::string> vars;

        // Add variables here that should be serialized to/from the dictionary.
        serialize(mMaxBounces);
        serialize(mUseAnalyticLights);
        serialize(mUseEmissiveLights);
        serialize(mUseEnvLight);
        serialize(mUseEnvBackground);
        serialize(mTexLODMode);
        serialize(mUsingRasterizedGBuffer);

        if constexpr (loadFromDict)
        {
            for (const auto& v : dict)
            {
                auto k = v.key();
                if (vars.find(v.key()) == vars.end())
                    logWarning("Unknown field `" + v.key() + "` in a WhittedRayTracer dictionary");
            }
        }
    }
#undef serialize

};
