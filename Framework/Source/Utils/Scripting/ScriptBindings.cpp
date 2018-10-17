/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
***************************************************************************/
#include "Framework.h"
#include "ScriptBindings.h"
#include "Scripting.h"
#include "API/Sampler.h"
#include "Effects/ToneMapping/ToneMapping.h"
#include "Graphics/Scene/Scene.h"
#include "Graphics/RenderGraph/RenderGraph.h"

#ifdef FALCOR_D3D12
#include "Raytracing/RtScene.h"
#endif

using namespace pybind11::literals;

namespace Falcor
{
    const char* ScriptBindings::kLoadScene = "loadScene";
    const char* ScriptBindings::kLoadRtScene = "loadRtScene";

#define val(a) value(to_string(a).c_str(), a)

    static void globalEnums(pybind11::module& m)
    {
        // Resource formats
        auto formats = pybind11::enum_<ResourceFormat>(m, "Format");
        for (uint32_t i = 0; i < (uint32_t)ResourceFormat::Count; i++)
        {
            formats.val(ResourceFormat(i));
        }

        // Comparison mode
        auto comparison = pybind11::enum_<ComparisonFunc>(m, "Comparison");
        comparison.val(ComparisonFunc::Disabled).val(ComparisonFunc::LessEqual).val(ComparisonFunc::GreaterEqual).val(ComparisonFunc::Less).val(ComparisonFunc::Greater);
        comparison.val(ComparisonFunc::Equal).val(ComparisonFunc::NotEqual).val(ComparisonFunc::Always).val(ComparisonFunc::Never);
    }

    static void samplerState(pybind11::module& m)
    {
        auto filter = pybind11::enum_<Sampler::Filter>(m, "Filter");
        filter.val(Sampler::Filter::Linear).val(Sampler::Filter::Point);

        auto addressing = pybind11::enum_<Sampler::AddressMode>(m, "AddressMode");
        addressing.val(Sampler::AddressMode::Wrap).val(Sampler::AddressMode::Mirror).val(Sampler::AddressMode::Clamp).val(Sampler::AddressMode::Border).val(Sampler::AddressMode::MirrorOnce);
    }

    static void toneMapping(pybind11::module& m)
    {
        auto op = pybind11::enum_<ToneMapping::Operator>(m, "ToneMapOp");
        op.val(ToneMapping::Operator::Clamp).val(ToneMapping::Operator::Linear).val(ToneMapping::Operator::Reinhard).val(ToneMapping::Operator::ReinhardModified).val(ToneMapping::Operator::HejiHableAlu);
        op.val(ToneMapping::Operator::HableUc2).val(ToneMapping::Operator::Aces);
    }

    static void scene(pybind11::module& m)
    {
        // Model load flags
        auto model = pybind11::enum_<Model::LoadFlags>(m, "ModelLoadFlags");
        model.val(Model::LoadFlags::None).val(Model::LoadFlags::DontGenerateTangentSpace).val(Model::LoadFlags::FindDegeneratePrimitives).val(Model::LoadFlags::AssumeLinearSpaceTextures);
        model.val(Model::LoadFlags::DontMergeMeshes).val(Model::LoadFlags::BuffersAsShaderResource).val(Model::LoadFlags::RemoveInstancing).val(Model::LoadFlags::UseSpecGlossMaterials);

        // Scene load flags
        auto scene = pybind11::enum_<Scene::LoadFlags>(m, "SceneLoadFlags");
        scene.val(Scene::LoadFlags::None).val(Scene::LoadFlags::GenerateAreaLights);

        // Scene
        m.def(ScriptBindings::kLoadScene, &Scene::loadFromFile, "filename"_a, "modelLoadFlags"_a = Model::LoadFlags::None, "sceneLoadFlags"_a = Scene::LoadFlags::None);
        auto sceneClass = pybind11::class_<Scene, Scene::SharedPtr>(m, "Scene");

        // RtScene
#ifdef FALCOR_D3D12
        // RtSceneFlags
        auto rtScene = pybind11::enum_<RtBuildFlags>(m, "RtBuildFlags");
        rtScene.val(RtBuildFlags::None).val(RtBuildFlags::AllowUpdate).val(RtBuildFlags::AllowCompaction).val(RtBuildFlags::FastTrace).val(RtBuildFlags::FastBuild);
        rtScene.val(RtBuildFlags::MinimizeMemory).val(RtBuildFlags::PerformUpdate);

        auto rtSceneClass = pybind11::class_<RtScene, RtScene::SharedPtr>(m, "RtScene", sceneClass);
        m.def(ScriptBindings::kLoadRtScene, &RtScene::loadFromFile, "filename"_a, "rtBuildFlags"_a = RtBuildFlags::None, "modelLoadFlags"_a = Model::LoadFlags::None, "sceneLoadFlags"_a = Scene::LoadFlags::None);
#endif
    }

    static void coreClasses(pybind11::module& m)
    {
#define reg_class(c_) pybind11::class_<c_, c_::SharedPtr>(m, #c_);

        // API
        reg_class(BlendState);
        reg_class(Buffer);
//        reg_class(ConstantBuffer); Doesn't work since it uses user-defined SharedPtr
        reg_class(DepthStencilState);
        reg_class(Fbo);
        reg_class(GpuTimer);
        reg_class(RasterizerState);
        reg_class(Resource);
        reg_class(ShaderResourceView);
        reg_class(DepthStencilView);
        reg_class(RenderTargetView);
        reg_class(ConstantBufferView);
        reg_class(UnorderedAccessView);
        reg_class(Sampler);
//        reg_class(StructuredBuffer); Doesn't work since it uses user-defined SharedPtr
        reg_class(Texture);
//        reg_class(TypedBuffer); Doesn't work since it uses user-defined SharedPtr
        reg_class(Vao);
        reg_class(VertexLayout);

        // Graphics
        reg_class(Camera);
        reg_class(Material);
        reg_class(Model);
        reg_class(Mesh);
        reg_class(ObjectPath);
        reg_class(Program);
//         reg_class(GraphicsVars); Doesn't work since it uses user-defined SharedPtr
//         reg_class(ComputeVars); Doesn't work since it uses user-defined SharedPtr
        reg_class(GraphicsState);
        reg_class(ComputeState);
        reg_class(Light);
        reg_class(LightProbe);

#undef reg_class
    }

    void ScriptBindings::registerScriptingObjects(pybind11::module& m)
    {
        globalEnums(m);
        coreClasses(m);
        samplerState(m);
        toneMapping(m);
        scene(m);
    }
}