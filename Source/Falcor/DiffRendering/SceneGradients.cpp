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
#include "SceneGradients.h"

namespace Falcor
{
namespace
{
const char kShaderFilename[] = "DiffRendering/SceneGradients.slang";
const char kSceneGradientsBlockName[] = "gSceneGradients";
const char kGradsBufferName[] = "grads";
const char kTmpGradsBufferName[] = "tmpGrads";

const char kAggregateShaderFilename[] = "DiffRendering/AggregateGradients.cs.slang";
} // namespace

SceneGradients::SceneGradients(ref<Device> pDevice, const std::vector<GradConfig>& gradConfigs, GradientAggregateMode mode)
    : mpDevice(pDevice), mAggregateMode(mode)
{
    // Initialization.
    mGradInfos.fill({false, 0, 0});

    for (size_t i = 0; i < gradConfigs.size(); i++)
    {
        auto type = size_t(gradConfigs[i].type);
        mGradInfos[type] = {true, gradConfigs[i].dim, gradConfigs[i].hashSize};
    }

    createParameterBlock();

    // Create a pass for aggregating gradients.
    ProgramDesc desc;
    if (mAggregateMode == GradientAggregateMode::Direct)
        desc.addShaderLibrary(kAggregateShaderFilename).csEntry("mainDirect");
    else
        desc.addShaderLibrary(kAggregateShaderFilename).csEntry("mainHashGrid");

    mpAggregatePass = ComputePass::create(mpDevice, desc);
}

void SceneGradients::createParameterBlock()
{
    DefineList defines;
    defines.add("SCENE_GRADIENTS_BLOCK");
    auto pPass = ComputePass::create(mpDevice, kShaderFilename, "main", defines);
    auto pReflector = pPass->getProgram()->getReflector()->getParameterBlock(kSceneGradientsBlockName);
    FALCOR_ASSERT(pReflector);

    // Create GPU buffers.
    auto bindFlags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    for (size_t i = 0; i < size_t(GradientType::Count); i++)
    {
        if (mGradInfos[i].active)
        {
            mpGrads[i] = mpDevice->createBuffer(
                mGradInfos[i].dim * sizeof(float), bindFlags | ResourceBindFlags::Shared, MemoryType::DeviceLocal, nullptr
            );
            mpTmpGrads[i] = mpDevice->createBuffer(
                mGradInfos[i].dim * mGradInfos[i].hashSize * sizeof(float), bindFlags, MemoryType::DeviceLocal, nullptr
            );
        }
        else
        {
            mpGrads[i] = mpTmpGrads[i] = nullptr;
        }
    }

    // Bind resources to parameter block.
    mpSceneGradientsBlock = ParameterBlock::create(mpDevice, pReflector);
    auto var = mpSceneGradientsBlock->getRootVar();
    for (size_t i = 0; i < size_t(GradientType::Count); i++)
    {
        var["gradDim"][i] = mGradInfos[i].dim;
        var["hashSize"][i] = mGradInfos[i].hashSize;
        var[kTmpGradsBufferName][i] = mpTmpGrads[i];
    }
}

void SceneGradients::clearGrads(RenderContext* pRenderContext, GradientType _gradType)
{
    uint32_t gradType = uint32_t(_gradType);
    if (!mGradInfos[gradType].active)
        return;
    pRenderContext->clearUAV(mpTmpGrads[gradType]->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(mpGrads[gradType]->getUAV().get(), uint4(0));
}

void SceneGradients::aggregateGrads(RenderContext* pRenderContext, GradientType _gradType)
{
    uint32_t gradType = uint32_t(_gradType);
    if (!mGradInfos[gradType].active)
        return;

    uint32_t hashSize = (mAggregateMode == GradientAggregateMode::Direct ? 1 : mGradInfos[gradType].hashSize);

    // Bind resources.
    auto var = mpAggregatePass->getRootVar()["gAggregator"];
    var["gradDim"] = mGradInfos[gradType].dim;
    var["hashSize"] = hashSize;
    var[kTmpGradsBufferName] = mpTmpGrads[gradType];
    var[kGradsBufferName] = mpGrads[gradType];

    // Dispatch.
    mpAggregatePass->execute(pRenderContext, uint3(mGradInfos[gradType].dim, hashSize, 1));
}

void SceneGradients::clearAllGrads(RenderContext* pRenderContext)
{
    for (size_t i = 0; i < size_t(GradientType::Count); i++)
        clearGrads(pRenderContext, GradientType(i));
}

void SceneGradients::aggregateAllGrads(RenderContext* pRenderContext)
{
    for (size_t i = 0; i < size_t(GradientType::Count); i++)
        aggregateGrads(pRenderContext, GradientType(i));
}

std::vector<GradientType> SceneGradients::getActiveGradTypes() const
{
    std::vector<GradientType> activeGradTypes;
    for (size_t i = 0; i < size_t(GradientType::Count); i++)
        if (mGradInfos[i].active)
            activeGradTypes.push_back(GradientType(i));
    return activeGradTypes;
}

inline static ref<SceneGradients> createPython(ref<Device> device, const pybind11::list& gradConfigList)
{
    std::vector<SceneGradients::GradConfig> gradConfigs;
    for (const auto& gradConfig : gradConfigList)
    {
        auto config = gradConfig.cast<SceneGradients::GradConfig>();
        gradConfigs.push_back(config);
    }
    return SceneGradients::create(device, gradConfigs);
}

inline void aggregate(SceneGradients& self, RenderContext* pRenderContext, GradientType gradType)
{
    self.aggregateGrads(pRenderContext, gradType);
#if FALCOR_HAS_CUDA
    pRenderContext->waitForFalcor();
#endif
}

inline void aggregateAll(SceneGradients& self, RenderContext* pRenderContext)
{
    self.aggregateAllGrads(pRenderContext);
#if FALCOR_HAS_CUDA
    pRenderContext->waitForFalcor();
#endif
}

inline pybind11::list getGradTypes(SceneGradients& self)
{
    pybind11::list gradTypes;
    for (const auto& gradType : self.getActiveGradTypes())
        gradTypes.append(gradType);
    return gradTypes;
}

FALCOR_SCRIPT_BINDING(SceneGradients)
{
    using namespace pybind11::literals;

    pybind11::falcor_enum<GradientType>(m, "GradientType");

    pybind11::class_<SceneGradients::GradConfig> gc(m, "GradConfig");
    gc.def(pybind11::init<>());
    gc.def(pybind11::init<GradientType, uint32_t, uint32_t>(), "grad_type"_a, "dim"_a, "hash_size"_a);

    pybind11::class_<SceneGradients, ref<SceneGradients>> sg(m, "SceneGradients");
    sg.def_static("create", createPython, "device"_a, "grad_config_list"_a);
    sg.def("clear_grads", &SceneGradients::clearGrads, "render_context"_a, "grad_type"_a);
    sg.def("aggregate_grads", aggregate, "render_context"_a, "grad_type"_a);
    sg.def("clear_all_grads", &SceneGradients::clearAllGrads, "render_context"_a);
    sg.def("aggregate_all_grads", aggregateAll, "render_context"_a);
    sg.def("get_grad_types", getGradTypes);
    sg.def("get_grads_buffer", &SceneGradients::getGradsBuffer, "grad_type"_a);
}
} // namespace Falcor
