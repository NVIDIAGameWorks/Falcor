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
#include "EmissiveLightSampler.h"
#include "LightBVH.h"
#include "LightBVHBuilder.h"
#include "LightBVHSamplerSharedDefinitions.slang"
#include "Core/Macros.h"
#include "Utils/Properties.h"
#include "Utils/Math/AABB.h"
#include "Scene/Lights/LightCollection.h"
#include <memory>

namespace Falcor
{
    /** The CPU pointer to the lighting system's acceleration structure is
        passed to RenderPass::execute() via a field with this name in the
        dictionary.
    */
    static const char kLightingAccelerationStructure[] = "_lightingAccelerationStructure";

    /** Emissive light sampler using a light BVH.

        This class wraps a LightCollection object, which holds the set of lights to sample.
        Internally, the class build a BVH over the light sources.
    */
    class FALCOR_API LightBVHSampler : public EmissiveLightSampler
    {
    public:
        /** LightBVHSampler configuration.
            Note if you change options, please update FALCOR_SCRIPT_BINDING in LightBVHSampler.cpp
        */
        struct Options
        {
            // Build options
            LightBVHBuilder::Options buildOptions;

            // Traversal options
            bool        useBoundingCone = true;             ///< Use bounding cone to BVH nodes to bound NdotL when computing probabilities.
            bool        useLightingCone = true;             ///< Use lighting cone in BVH nodes to cull backfacing lights when computing probabilities.
            bool        disableNodeFlux = false;            ///< Do not take per-node flux into account in sampling.
            bool        useUniformTriangleSampling = true;  ///< Use uniform sampling to select a triangle within the sampled leaf node.

            SolidAngleBoundMethod solidAngleBoundMethod = SolidAngleBoundMethod::Sphere; ///< Method to use to bound the solid angle subtended by a cluster.

            // Note: Empty constructor needed for clang due to the use of the nested struct constructor in the parent constructor.
            Options() {}

            template<typename Archive>
            void serialize(Archive& ar)
            {
                ar("buildOptions", buildOptions);
                ar("useBoundingCone", useBoundingCone);
                ar("useLightingCone", useLightingCone);
                ar("disableNodeFlux", disableNodeFlux);
                ar("useUniformTriangleSampling", useUniformTriangleSampling);
                ar("solidAngleBoundMethod", solidAngleBoundMethod);
            }
        };

        /** Creates a LightBVHSampler for a given scene.
            \param[in] pRenderContext The render context.
            \param[in] pScene The scene.
            \param[in] options The options to override the default behavior.
        */
        LightBVHSampler(RenderContext* pRenderContext, ref<Scene> pScene, const Options& options = Options());
        virtual ~LightBVHSampler() = default;

        /** Updates the sampler to the current frame.
            \param[in] pRenderContext The render context.
            \return True if the sampler was updated.
        */
        virtual bool update(RenderContext* pRenderContext) override;

        /** Return a list of shader defines to use this light sampler.
        *   \return Returns a list of shader defines.
        */
        virtual DefineList getDefines() const override;

        /** Bind the light sampler data to a given shader variable.
            \param[in] var Shader variable.
        */
        virtual void bindShaderData(const ShaderVar& var) const override;

        /** Render the GUI.
            \return True if setting the refresh flag is needed, false otherwise.
        */
        virtual bool renderUI(Gui::Widgets& widget) override;

        /** Returns the current configuration.
        */
        const Options& getOptions() const { return mOptions; }

        void setOptions(const Options& options);

    protected:
        /// Configuration options.
        Options mOptions;

        // Internal state
        std::unique_ptr<LightBVHBuilder> mpBVHBuilder;
        std::unique_ptr<LightBVH> mpBVH;

        /// Trigger rebuild on the next call to update(). We should always build on the first call, so the initial value is true.
        bool mNeedsRebuild = true;
    };
}
