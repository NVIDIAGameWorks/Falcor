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
#include "SceneDebugger.h"

namespace
{
const char kShaderFile[] = "RenderPasses/SceneDebugger/SceneDebugger.cs.slang";

const std::string kOutput = "output";

std::string getModeDesc(SceneDebuggerMode mode)
{
    switch (mode)
    {
    // Geometry
    case SceneDebuggerMode::HitType:
        return "Hit type in pseudocolor";
    case SceneDebuggerMode::InstanceID:
        return "Instance ID in pseudocolor";
    case SceneDebuggerMode::MaterialID:
        return "Material ID in pseudocolor";
    case SceneDebuggerMode::PrimitiveID:
        return "Primitive ID in pseudocolor";
    case SceneDebuggerMode::GeometryID:
        return "Geometry ID in pseudocolor";
    case SceneDebuggerMode::BlasID:
        return "Raytracing bottom-level acceleration structure (BLAS) ID in pseudocolor";
    case SceneDebuggerMode::InstancedGeometry:
        return "Green = instanced geometry\n"
               "Red = non-instanced geometry";
    // Shading data
    case SceneDebuggerMode::FaceNormal:
        return "Face normal in RGB color";
    case SceneDebuggerMode::ShadingNormal:
        return "Shading normal in RGB color";
    case SceneDebuggerMode::ShadingTangent:
        return "Shading tangent in RGB color";
    case SceneDebuggerMode::ShadingBitangent:
        return "Shading bitangent in RGB color";
    case SceneDebuggerMode::FrontFacingFlag:
        return "Green = front-facing\n"
               "Red = back-facing";
    case SceneDebuggerMode::BackfacingShadingNormal:
        return "Pixels where the shading normal is back-facing with respect to view vector are highlighted";
    case SceneDebuggerMode::TexCoords:
        return "Texture coordinates in RG color wrapped to [0,1]";
    // Material properties
    case SceneDebuggerMode::GuideNormal:
        return "Guide normal in RGB color";
    case SceneDebuggerMode::Roughness:
        return "Material roughness estimate";
    case SceneDebuggerMode::FlatShaded:
        return "Flat shaded";
    default:
        FALCOR_UNREACHABLE();
        return "";
    }
}

// Scripting
const char kMode[] = "mode";
const char kShowVolumes[] = "showVolumes";

void registerBindings(pybind11::module& m)
{
    pybind11::class_<SceneDebugger, RenderPass, ref<SceneDebugger>> pass(m, "SceneDebugger");
    pass.def_property(
        kMode,
        [](const SceneDebugger& self) { return enumToString(self.getMode()); },
        [](SceneDebugger& self, const std::string& value) { self.setMode(stringToEnum<SceneDebuggerMode>(value)); }
    );
}
} // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, SceneDebugger>();
    Falcor::ScriptBindings::registerBinding(registerBindings);
}

SceneDebugger::SceneDebugger(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    if (!mpDevice->isShaderModelSupported(ShaderModel::SM6_5))
        FALCOR_THROW("SceneDebugger requires Shader Model 6.5 support.");
    if (!mpDevice->isFeatureSupported(Device::SupportedFeatures::RaytracingTier1_1))
        FALCOR_THROW("SceneDebugger requires Raytracing Tier 1.1 support.");

    // Parse dictionary.
    for (const auto& [key, value] : props)
    {
        if (key == kMode)
            mParams.mode = (uint32_t)value.operator SceneDebuggerMode();
        else if (key == kShowVolumes)
            mParams.showVolumes = value;
        else
            logWarning("Unknown property '{}' in a SceneDebugger properties.", key);
    }

    mpFence = mpDevice->createFence();
}

Properties SceneDebugger::getProperties() const
{
    Properties props;
    props[kMode] = SceneDebuggerMode(mParams.mode);
    props[kShowVolumes] = mParams.showVolumes;
    return props;
}

RenderPassReflection SceneDebugger::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addOutput(kOutput, "Scene debugger output").bindFlags(ResourceBindFlags::UnorderedAccess).format(ResourceFormat::RGBA32Float);

    return reflector;
}

void SceneDebugger::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    mParams.frameDim = compileData.defaultTexDims;
}

void SceneDebugger::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpMeshToBlasID = nullptr;
    mpDebugPass = nullptr;

    if (mpScene)
    {
        // Prepare our programs for the scene.
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderFile).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());
        mpDebugPass = ComputePass::create(mpDevice, desc, mpScene->getSceneDefines());

        // Create lookup table for mesh to BLAS ID.
        auto blasIDs = mpScene->getMeshBlasIDs();
        if (!blasIDs.empty())
        {
            mpMeshToBlasID = mpDevice->createStructuredBuffer(
                sizeof(uint32_t),
                (uint32_t)blasIDs.size(),
                ResourceBindFlags::ShaderResource,
                MemoryType::DeviceLocal,
                blasIDs.data(),
                false
            );
        }

        // Create instance metadata.
        initInstanceInfo();

        // Bind variables.
        auto var = mpDebugPass->getRootVar()["CB"]["gSceneDebugger"];
        if (!mpPixelData)
        {
            mpPixelData = mpDevice->createStructuredBuffer(
                var["pixelData"],
                1,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal,
                nullptr,
                false
            );
            mpPixelDataStaging =
                mpDevice->createStructuredBuffer(var["pixelData"], 1, ResourceBindFlags::None, MemoryType::ReadBack, nullptr, false);
        }
        var["pixelData"] = mpPixelData;
        var["meshToBlasID"] = mpMeshToBlasID;
        var["instanceInfo"] = mpInstanceInfo;
    }
}

void SceneDebugger::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    mPixelDataAvailable = false;
    const auto& pOutput = renderData.getTexture(kOutput);

    if (mpScene == nullptr)
    {
        pRenderContext->clearUAV(pOutput->getUAV().get(), float4(0.f));
        return;
    }
    // DEMO21:
    // mpScene->getCamera()->setJitter(0.f, 0.f);

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::RecompileNeeded) ||
        is_set(mpScene->getUpdates(), Scene::UpdateFlags::GeometryChanged))
    {
        FALCOR_THROW("This render pass does not support scene changes that require shader recompilation.");
    }

    mpScene->setRaytracingShaderData(pRenderContext, mpDebugPass->getRootVar());

    ShaderVar var = mpDebugPass->getRootVar()["CB"]["gSceneDebugger"];
    var["params"].setBlob(mParams);
    var["output"] = pOutput;

    mpDebugPass->execute(pRenderContext, uint3(mParams.frameDim, 1));

    pRenderContext->copyResource(mpPixelDataStaging.get(), mpPixelData.get());
    pRenderContext->submit(false);
    pRenderContext->signal(mpFence.get());

    mPixelDataAvailable = true;
    mParams.frameCount++;
}

void SceneDebugger::renderUI(Gui::Widgets& widget)
{
    widget.dropdown("Mode", reinterpret_cast<SceneDebuggerMode&>(mParams.mode));
    widget.tooltip("Selects visualization mode");

    widget.checkbox("Clamp to [0,1]", mParams.clamp);
    widget.tooltip("Clamp pixel values to [0,1] before output.");

    if ((SceneDebuggerMode)mParams.mode == SceneDebuggerMode::FaceNormal ||
        (SceneDebuggerMode)mParams.mode == SceneDebuggerMode::GuideNormal ||
        (SceneDebuggerMode)mParams.mode == SceneDebuggerMode::ShadingNormal ||
        (SceneDebuggerMode)mParams.mode == SceneDebuggerMode::ShadingTangent ||
        (SceneDebuggerMode)mParams.mode == SceneDebuggerMode::ShadingBitangent ||
        (SceneDebuggerMode)mParams.mode == SceneDebuggerMode::TexCoords)
    {
        widget.checkbox("Flip sign", mParams.flipSign);
        widget.checkbox("Remap to [0,1]", mParams.remapRange);
        widget.tooltip("Remap range from [-1,1] to [0,1] before output.");
    }

    widget.checkbox("Show volumes", mParams.showVolumes);
    if (mParams.showVolumes)
    {
        widget.var("Density scale", mParams.densityScale, 0.f, 1000.f, 0.1f);
    }

    widget.textWrapped("Description:\n" + getModeDesc((SceneDebuggerMode)mParams.mode));

    // Show data for the currently selected pixel.
    widget.dummy("#spacer0", {1, 20});
    widget.var("Selected pixel", mParams.selectedPixel);

    renderPixelDataUI(widget);

    widget.dummy("#spacer1", {1, 20});
    widget.text("Scene: " + (mpScene ? mpScene->getPath().string() : "No scene loaded"));
}

void SceneDebugger::renderPixelDataUI(Gui::Widgets& widget)
{
    if (!mPixelDataAvailable)
        return;

    FALCOR_ASSERT(mpPixelDataStaging);
    mpFence->wait();
    const PixelData data = mpPixelDataStaging->getElement<PixelData>(0);

    switch ((HitType)data.hitType)
    {
    case HitType::Triangle:
    {
        {
            std::string text;
            text += fmt::format("Mesh ID: {}\n", data.geometryID);
            text += fmt::format("Mesh name: {}\n", mpScene->hasMesh(data.geometryID) ? mpScene->getMeshName(data.geometryID) : "unknown");
            text += fmt::format("Instance ID: {}\n", data.instanceID);
            text += fmt::format("Material ID: {}\n", data.materialID);
            text += fmt::format("BLAS ID: {}\n", data.blasID);
            widget.text(text);
            widget.dummy("#spacer2", {1, 10});
        }

        // Show mesh details.
        if (auto g = widget.group("Mesh info"); g.open())
        {
            FALCOR_ASSERT(data.geometryID < mpScene->getMeshCount());
            const auto& mesh = mpScene->getMesh(MeshID{data.geometryID});
            std::string text;
            text += fmt::format("flags: 0x{:08x}\n", mesh.flags);
            text += fmt::format("materialID: {}\n", mesh.materialID);
            text += fmt::format("vertexCount: {}\n", mesh.vertexCount);
            text += fmt::format("indexCount: {}\n", mesh.indexCount);
            text += fmt::format("triangleCount: {}\n", mesh.getTriangleCount());
            text += fmt::format("vbOffset: {}\n", mesh.vbOffset);
            text += fmt::format("ibOffset: {}\n", mesh.ibOffset);
            text += fmt::format("skinningVbOffset: {}\n", mesh.skinningVbOffset);
            text += fmt::format("prevVbOffset: {}\n", mesh.prevVbOffset);
            text += fmt::format("use16BitIndices: {}\n", mesh.use16BitIndices());
            text += fmt::format("isFrontFaceCW: {}\n", mesh.isFrontFaceCW());
            g.text(text);
        }

        // Show mesh instance info.
        if (auto g = widget.group("Mesh instance info"); g.open())
        {
            FALCOR_ASSERT(data.instanceID < mpScene->getGeometryInstanceCount());
            const auto& instance = mpScene->getGeometryInstance(data.instanceID);
            std::string text;
            text += fmt::format("flags: 0x{:08x}\n", instance.flags);
            text += fmt::format("nodeID: {}\n", instance.globalMatrixID);
            text += fmt::format("meshID: {}\n", instance.geometryID);
            text += fmt::format("materialID: {}\n", instance.materialID);
            text += fmt::format("vbOffset: {}\n", instance.vbOffset);
            text += fmt::format("ibOffset: {}\n", instance.ibOffset);
            text += fmt::format("isDynamic: {}\n", instance.isDynamic());
            g.text(text);

            // Print the list of scene graph nodes affecting this mesh instance.
            std::vector<NodeID> nodes;
            {
                NodeID nodeID{instance.globalMatrixID};
                while (nodeID != NodeID::Invalid())
                {
                    nodes.push_back(nodeID);
                    nodeID = mpScene->getParentNodeID(nodeID);
                }
            }
            FALCOR_ASSERT(!nodes.empty());

            g.text("Scene graph (root first):");
            const auto& localMatrices = mpScene->getAnimationController()->getLocalMatrices();
            for (auto it = nodes.rbegin(); it != nodes.rend(); it++)
            {
                auto nodeID = *it;
                float4x4 mat = localMatrices[nodeID.get()];
                if (auto nodeGroup = widget.group("ID " + to_string(nodeID)); nodeGroup.open())
                {
                    g.matrix("", mat);
                }
            }
        }
        break;
    }
    case HitType::Curve:
    {
        {
            std::string text;
            text += fmt::format("Curve ID: {}\n", data.geometryID);
            text += fmt::format("Instance ID: {}\n", data.instanceID);
            text += fmt::format("Material ID: {}\n", data.materialID);
            text += fmt::format("BLAS ID: {}\n", data.blasID);
            widget.text(text);
            widget.dummy("#spacer2", {1, 10});
        }

        // Show mesh details.
        if (auto g = widget.group("Curve info"); g.open())
        {
            const auto& curve = mpScene->getCurve(CurveID{data.geometryID});
            std::string text;
            text += fmt::format("degree: {}\n", curve.degree);
            text += fmt::format("vertexCount: {}\n", curve.vertexCount);
            text += fmt::format("indexCount: {}\n", curve.indexCount);
            text += fmt::format("vbOffset: {}\n", curve.vbOffset);
            text += fmt::format("ibOffset: {}\n", curve.ibOffset);
            g.text(text);
        }
        break;
    }
    case HitType::SDFGrid:
    {
        {
            std::string text;
            text += fmt::format("SDF Grid ID: {}\n", data.geometryID);
            text += fmt::format("Instance ID: {}\n", data.instanceID);
            text += fmt::format("Material ID: {}\n", data.materialID);
            text += fmt::format("BLAS ID: {}\n", data.blasID);
            widget.text(text);
            widget.dummy("#spacer2", {1, 10});
        }

        // Show SDF grid details.
        if (auto g = widget.group("SDF grid info"); g.open())
        {
            const ref<SDFGrid>& pSDFGrid = mpScene->getSDFGrid(SdfGridID{data.geometryID});
            std::string text;
            text += fmt::format("gridWidth: {}\n", pSDFGrid->getGridWidth());
            g.text(text);
        }
        break;
    }
    case HitType::None:
        widget.text("Background pixel");
        break;
    default:
        widget.text("Unsupported hit type");
        break;
    }

    // Show shading data.
    if ((HitType)data.hitType != HitType::None)
    {
        if (auto g = widget.group("Shading data"); g.open())
        {
            std::string text;
            text += fmt::format("posW: {}\n", data.posW);
            text += fmt::format("V: {}\n", data.V);
            text += fmt::format("N: {}\n", data.N);
            text += fmt::format("T: {}\n", data.T);
            text += fmt::format("B: {}\n", data.B);
            text += fmt::format("uv: {}\n", data.uv);
            text += fmt::format("faceN: {}\n", data.faceN);
            text += fmt::format("tangentW: {}\n", data.tangentW);
            text += fmt::format("frontFacing: {}\n", data.frontFacing);
            text += fmt::format("curveRadius: {}\n", data.curveRadius);
            g.text(text);
        }
    }

    // Show material info.
    if (data.materialID != PixelData::kInvalidID)
    {
        if (auto g = widget.group("Material info"); g.open())
        {
            const auto& material = *mpScene->getMaterial(MaterialID{data.materialID});
            const auto& header = material.getHeader();
            std::string text;
            text += fmt::format("name: {}\n", material.getName());
            text += fmt::format("materialType: {}\n", to_string(header.getMaterialType()));
            text += fmt::format("alphaMode: {}\n", (uint32_t)header.getAlphaMode());
            text += fmt::format("alphaThreshold: {}\n", (float)header.getAlphaThreshold());
            text += fmt::format("nestedPriority: {}\n", header.getNestedPriority());
            text += fmt::format("activeLobes: 0x{:08x}\n", (uint32_t)header.getActiveLobes());
            text += fmt::format("defaultTextureSamplerID: {}\n", header.getDefaultTextureSamplerID());
            text += fmt::format("doubleSided: {}\n", header.isDoubleSided());
            text += fmt::format("thinSurface: {}\n", header.isThinSurface());
            text += fmt::format("emissive: {}\n", header.isEmissive());
            text += fmt::format("basicMaterial: {}\n", header.isBasicMaterial());
            text += fmt::format("lightProfileEnabled: {}\n", header.isLightProfileEnabled());
            text += fmt::format("deltaSpecular: {}\n", header.isDeltaSpecular());
            g.text(text);
        }
    }
}

bool SceneDebugger::onMouseEvent(const MouseEvent& mouseEvent)
{
    if (mouseEvent.type == MouseEvent::Type::ButtonDown && mouseEvent.button == Input::MouseButton::Left)
    {
        float2 cursorPos = mouseEvent.pos * (float2)mParams.frameDim;
        mParams.selectedPixel = (uint2)clamp(cursorPos, float2(0.f), float2(mParams.frameDim.x - 1, mParams.frameDim.y - 1));
    }

    return false;
}

void SceneDebugger::initInstanceInfo()
{
    const uint32_t instanceCount = mpScene ? mpScene->getGeometryInstanceCount() : 0;

    // If there are no instances. Just clear the buffer and return.
    if (instanceCount == 0)
    {
        mpInstanceInfo = nullptr;
        return;
    }

    // Count number of times each geometry is used.
    std::vector<std::vector<uint32_t>> instanceCounts((size_t)GeometryType::Count);
    for (auto& counts : instanceCounts)
        counts.resize(mpScene->getGeometryCount());

    for (uint32_t instanceID = 0; instanceID < instanceCount; instanceID++)
    {
        const auto& instance = mpScene->getGeometryInstance(instanceID);
        instanceCounts[(size_t)instance.getType()][instance.geometryID]++;
    }

    // Setup instance metadata.
    std::vector<InstanceInfo> instanceInfo(instanceCount);
    for (uint32_t instanceID = 0; instanceID < instanceCount; instanceID++)
    {
        const auto& instance = mpScene->getGeometryInstance(instanceID);
        auto& info = instanceInfo[instanceID];
        if (instanceCounts[(size_t)instance.getType()][instance.geometryID] > 1)
        {
            info.flags |= (uint32_t)InstanceInfoFlags::IsInstanced;
        }
    }

    // Create GPU buffer.
    mpInstanceInfo = mpDevice->createStructuredBuffer(
        sizeof(InstanceInfo),
        (uint32_t)instanceInfo.size(),
        ResourceBindFlags::ShaderResource,
        MemoryType::DeviceLocal,
        instanceInfo.data(),
        false
    );
}
