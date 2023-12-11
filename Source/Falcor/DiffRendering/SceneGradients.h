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
#include "Core/API/ParameterBlock.h"
#include "RenderGraph/RenderPass.h"
#include "SharedTypes.slang"

namespace Falcor
{
class FALCOR_API SceneGradients : public Object
{
    FALCOR_OBJECT(SceneGradients);

public:
    struct GradConfig
    {
        GradientType type;
        uint32_t dim;
        uint32_t hashSize;

        GradConfig() {}
        GradConfig(GradientType _type, uint32_t _dim, uint32_t _hashSize) : type(_type), dim(_dim), hashSize(_hashSize) {}
    };

    SceneGradients(
        ref<Device> pDevice,
        const std::vector<GradConfig>& gradConfigs,
        GradientAggregateMode mode = GradientAggregateMode::HashGrid
    );

    static ref<SceneGradients> create(ref<Device> pDevice, const std::vector<GradConfig>& gradConfigs)
    {
        return make_ref<SceneGradients>(pDevice, gradConfigs, GradientAggregateMode::HashGrid);
    }

    ~SceneGradients() = default;

    void bindShaderData(const ShaderVar& var) const { var = mpSceneGradientsBlock; }

    void clearGrads(RenderContext* pRenderContext, GradientType gradType);
    void aggregateGrads(RenderContext* pRenderContext, GradientType gradType);

    void clearAllGrads(RenderContext* pRenderContext);
    void aggregateAllGrads(RenderContext* pRenderContext);

    uint32_t getGradDim(GradientType gradType) const { return mGradInfos[size_t(gradType)].dim; }
    uint32_t getHashSize(GradientType gradType) const { return mGradInfos[size_t(gradType)].hashSize; }

    const ref<Buffer>& getTmpGradsBuffer(GradientType gradType) const { return mpTmpGrads[size_t(gradType)]; }
    const ref<Buffer>& getGradsBuffer(GradientType gradType) const { return mpGrads[size_t(gradType)]; }

    std::vector<GradientType> getActiveGradTypes() const;

private:
    struct GradInfo
    {
        bool active;
        uint32_t dim;
        uint32_t hashSize;
    };

    void createParameterBlock();

    ref<Device> mpDevice;
    std::array<GradInfo, size_t(GradientType::Count)> mGradInfos;
    GradientAggregateMode mAggregateMode;

    ref<ParameterBlock> mpSceneGradientsBlock;

    ref<Buffer> mpGrads[size_t(GradientType::Count)];
    ref<Buffer> mpTmpGrads[size_t(GradientType::Count)];

    ref<ComputePass> mpAggregatePass;
};
} // namespace Falcor
