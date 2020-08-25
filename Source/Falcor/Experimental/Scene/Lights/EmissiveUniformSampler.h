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
#include "EmissiveLightSampler.h"
#include "LightCollection.h"

namespace Falcor
{
    /** Emissive light sampler using uniform sampling of the lights.

        This class wraps a LightCollection object, which holds the set of lights to sample.
    */
    class dlldecl EmissiveUniformSampler : public EmissiveLightSampler
    {
    public:
        using SharedPtr = std::shared_ptr<EmissiveUniformSampler>;
        using SharedConstPtr = std::shared_ptr<const EmissiveUniformSampler>;

        /** EmissiveUniformSampler configuration.
            Note if you change options, please update SCRIPT_BINDING in EmissiveUniformSampler.cpp
        */
        struct Options
        {
            // TODO
            //bool        usePreintegration = true;           ///< Use pre-integrated flux per triangle to guide BVH build/sampling. Only relevant if mUseBVHTree == true.
        };

        virtual ~EmissiveUniformSampler() = default;

        /** Creates a EmissiveUniformSampler for a given scene.
            \param[in] pRenderContext The render context.
            \param[in] pScene The scene.
            \param[in] options The options to override the default behavior.
        */
        static SharedPtr create(RenderContext* pRenderContext, Scene::SharedPtr pScene, const Options& options = Options());

        /** Returns the current configuration.
        */
        const Options& getOptions() const { return mOptions; }

    protected:
        EmissiveUniformSampler(RenderContext* pRenderContext, Scene::SharedPtr pScene, const Options& options);

        // Configuration
        Options mOptions;
    };
}
