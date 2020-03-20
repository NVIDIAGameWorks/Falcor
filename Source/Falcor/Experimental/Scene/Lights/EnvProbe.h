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

namespace Falcor
{
    /** Environment map based radiance probe.
        Utily class for sampling and evaluating radiance stored in an omnidirectional environment map.
    */
    class dlldecl EnvProbe : public std::enable_shared_from_this<EnvProbe>
    {
    public:
        using SharedPtr = std::shared_ptr<EnvProbe>;
        using SharedConstPtr = std::shared_ptr<const EnvProbe>;

        virtual ~EnvProbe() = default;

        /** Create a new object
            \param[in] pRenderContext A render-context that will be used for processing
            \param[in] filename The env-map texture filename
        */
        static SharedPtr create(RenderContext* pRenderContext, const std::string& filename);

        /** Bind the environment map probe to a given shader variable.
            \param[in] var Shader variable.
            \return True if successful, false otherwise.
        */
        bool setShaderData(const ShaderVar& var) const;

        const Texture::SharedPtr& getEnvMap() const { return mpEnvMap; }
        const Texture::SharedPtr& getImportanceMap() const { return mpImportanceMap; }
        const Sampler::SharedPtr& getEnvSampler() const { return mpEnvSampler; }

    protected:
        EnvProbe() = default;

        bool init(RenderContext* pRenderContext, const std::string& filename);
        bool createImportanceMap(RenderContext* pRenderContext, uint32_t dimension, uint32_t samples);

        ComputePass::SharedPtr  mpSetupPass;        ///< Compute pass for creating the importance map.

        Texture::SharedPtr      mpEnvMap;           ///< Loaded environment map (RGB).
        Texture::SharedPtr      mpImportanceMap;    ///< Hierarchical importance map (luminance).

        Sampler::SharedPtr      mpEnvSampler;
        Sampler::SharedPtr      mpImportanceSampler;
    };
}
