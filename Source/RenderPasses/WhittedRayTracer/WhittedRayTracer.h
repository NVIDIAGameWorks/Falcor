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
#include "RenderGraph/RenderPassHelpers.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "Rendering/Materials/TexLODTypes.slang"  // Using the enum with Mip0, RayCones, etc
#include "WhittedRayTracerTypes.slang"

using namespace Falcor;

/** Whitted ray tracer.

    This pass implements the simplest possible Whitted ray tracer.
*/
class WhittedRayTracer : public RenderPass
{
public:
    using SharedPtr = std::shared_ptr<WhittedRayTracer>;

    static const Info kInfo;

    static SharedPtr create(RenderContext* pRenderContext = nullptr, const Dictionary& dict = {});

    virtual Dictionary getScriptingDictionary() override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

    void setTexLODMode(const TexLODMode mode) { mTexLODMode = mode; }
    TexLODMode getTexLODMode() const { return mTexLODMode; }

    void setRayConeMode(const RayConeMode mode) { mRayConeMode = mode; }
    RayConeMode getRayConeMode() const { return mRayConeMode; }

    void setRayConeFilterMode(const RayFootprintFilterMode mode) { mRayConeFilterMode = mode; }
    RayFootprintFilterMode getRayConeFilterMode() const { return mRayConeFilterMode; }

    void setRayDiffFilterMode(const RayFootprintFilterMode mode) { mRayDiffFilterMode = mode; }
    RayFootprintFilterMode getRayDiffFilterMode() const { return mRayDiffFilterMode; }

    static void registerBindings(pybind11::module& m);

private:
    WhittedRayTracer(const Dictionary& dict);

    void prepareVars();
    void setStaticParams(RtProgram* pProgram) const;

    // Internal state
    Scene::SharedPtr            mpScene;                                    ///< Current scene.
    SampleGenerator::SharedPtr  mpSampleGenerator;                          ///< GPU sample generator.

    uint                        mMaxBounces = 3;                                        ///< Max number of indirect bounces (0 = none).
    TexLODMode                  mTexLODMode = TexLODMode::Mip0;                         ///< Which texture LOD mode to use.
    RayConeMode                 mRayConeMode = RayConeMode::Combo;                      ///< Which variant of ray cones to use.
    RayFootprintFilterMode      mRayConeFilterMode = RayFootprintFilterMode::Isotropic; ///< Which filter mode to use for ray cones.
    RayFootprintFilterMode      mRayDiffFilterMode = RayFootprintFilterMode::Isotropic; ///< Which filter mode to use for ray differentials.
    bool                        mVisualizeSurfaceSpread = false;                        ///< Visualize surface spread angle at the first hit for the ray cones methods.
    bool                        mUseRoughnessToVariance = false;                        ///< Use roughness to variance to grow ray cones based on BDSF roughness.
    bool                        mUseFresnelAsBRDF = false;                              ///< Use Fresnel term as BRDF (instead of hacky throughput adjustment)
    // Runtime data
    uint                        mFrameCount = 0;                                        ///< Frame count since scene was loaded.
    bool                        mOptionsChanged = false;

    // Ray tracing program.
    struct
    {
        RtProgram::SharedPtr pProgram;
        RtBindingTable::SharedPtr pBindingTable;
        RtProgramVars::SharedPtr pVars;
    } mTracer;
};
