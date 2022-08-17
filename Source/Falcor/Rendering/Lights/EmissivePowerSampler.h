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
#include "EmissiveLightSampler.h"
#include "Core/Macros.h"
#include "Scene/Lights/LightCollection.h"
#include <memory>
#include <random>
#include <vector>

namespace Falcor
{
    class RenderContext;
    struct ShaderVar;

    /** Sample geometry proportionally to its emissive power.
    */
    class FALCOR_API EmissivePowerSampler : public EmissiveLightSampler
    {
    public:
        using SharedPtr = std::shared_ptr<EmissivePowerSampler>;
        using SharedConstPtr = std::shared_ptr<const EmissivePowerSampler>;

        struct AliasTable
        {
            float weightSum;                ///< Total weight of all elements used to create the alias table
            uint32_t N;                     ///< Number of entries in the alias table (and # elements in the buffers)
            Buffer::SharedPtr fullTable;    ///< A compressed/packed merged table.  Max 2^24 (16 million) entries per table.
        };

        virtual ~EmissivePowerSampler() = default;

        /** Creates a EmissivePowerSampler for a given scene.
            \param[in] pRenderContext The render context.
            \param[in] pScene The scene.
            \param[in] options The options to override the default behavior.
        */
        static SharedPtr create(RenderContext* pRenderContext, Scene::SharedPtr pScene);

        /** Updates the sampler to the current frame.
            \param[in] pRenderContext The render context.
            \return True if the sampler was updated.
        */
        virtual bool update(RenderContext* pRenderContext) override;

        /** Bind the light sampler data to a given shader variable.
            \param[in] var Shader variable.
        */
        virtual void setShaderData(const ShaderVar& var) const override;

    protected:
        EmissivePowerSampler(RenderContext* pRenderContext, Scene::SharedPtr pScene);

        /** Generate an alias table
            \param[in] weights  The weights we'd like to sample each entry proportional to
            \returns The alias table
        */
        AliasTable generateAliasTable(std::vector<float> weights);

        // Internal state
        bool                            mNeedsRebuild = true;   ///< Trigger rebuild on the next call to update(). We should always build on the first call, so the initial value is true.

        LightCollection::SharedConstPtr mpLightCollection;

        std::mt19937                    mAliasTableRng;
        AliasTable                      mTriangleTable;
    };
}
