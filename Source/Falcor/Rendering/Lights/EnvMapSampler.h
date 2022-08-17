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
#include "Core/Macros.h"
#include "Core/API/Texture.h"
#include "Core/API/Sampler.h"
#include "Scene/Lights/EnvMap.h"
#include "RenderGraph/BasePasses/ComputePass.h"
#include <memory>

namespace Falcor
{
    class RenderContext;

    /** Environment map sampler.
        Utily class for sampling and evaluating radiance stored in an omnidirectional environment map.
    */
    class FALCOR_API EnvMapSampler
    {
    public:
        using SharedPtr = std::shared_ptr<EnvMapSampler>;

        virtual ~EnvMapSampler() = default;

        /** Create a new object.
            \param[in] pRenderContext A render-context that will be used for processing.
            \param[in] pEnvMap The environment map.
        */
        static SharedPtr create(RenderContext* pRenderContext, EnvMap::SharedPtr pEnvMap);

        /** Bind the environment map sampler to a given shader variable.
            \param[in] var Shader variable.
        */
        void setShaderData(const ShaderVar& var) const;

        const EnvMap::SharedPtr& getEnvMap() const { return mpEnvMap; }

        const Texture::SharedPtr& getImportanceMap() const { return mpImportanceMap; }

    protected:
        EnvMapSampler(RenderContext* pRenderContext, EnvMap::SharedPtr pEnvMap);

        bool createImportanceMap(RenderContext* pRenderContext, uint32_t dimension, uint32_t samples);

        EnvMap::SharedPtr       mpEnvMap;           ///< Environment map.

        ComputePass::SharedPtr  mpSetupPass;        ///< Compute pass for creating the importance map.

        Texture::SharedPtr      mpImportanceMap;    ///< Hierarchical importance map (luminance).
        Sampler::SharedPtr      mpImportanceSampler;
    };
}
