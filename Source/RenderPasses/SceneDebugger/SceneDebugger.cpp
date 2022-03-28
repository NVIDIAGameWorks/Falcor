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
#include "SceneDebugger.h"

const RenderPass::Info SceneDebugger::kInfo { "SceneDebugger", "Scene debugger for identifying asset issues." };

namespace
{
    const char kShaderFile[] = "RenderPasses/SceneDebugger/SceneDebugger.cs.slang";
    const char kShaderModel[] = "6_5";

    const std::string kOutput = "output";

    // UI elements
    const Gui::DropdownList kModeList =
    {
        { (uint32_t)SceneDebuggerMode::FaceNormal, "Face normal" },
        { (uint32_t)SceneDebuggerMode::ShadingNormal, "Shading normal" },
        { (uint32_t)SceneDebuggerMode::ShadingTangent, "Shading tangent" },
        { (uint32_t)SceneDebuggerMode::ShadingBitangent, "Shading bitangent" },
        { (uint32_t)SceneDebuggerMode::FrontFacingFlag, "Front-facing flag" },
        { (uint32_t)SceneDebuggerMode::BackfacingShadingNormal, "Back-facing shading normal" },
        { (uint32_t)SceneDebuggerMode::TexCoords, "Texture coordinates" },
        { (uint32_t)SceneDebuggerMode::HitType, "Hit type" },
        { (uint32_t)SceneDebuggerMode::InstanceID, "Instance ID" },
        { (uint32_t)SceneDebuggerMode::MaterialID, "Material ID" },
        { (uint32_t)SceneDebuggerMode::GeometryID, "Geometry ID" },
        { (uint32_t)SceneDebuggerMode::BlasID, "BLAS ID" },
        { (uint32_t)SceneDebuggerMode::InstancedGeometry, "Instanced geometry" },
    };

    std::string getModeDesc(SceneDebuggerMode mode)
    {
        switch (mode)
        {
        case SceneDebuggerMode::FaceNormal: return
            "Face normal in RGB color";
        case SceneDebuggerMode::ShadingNormal: return
            "Shading normal in RGB color";
        case SceneDebuggerMode::ShadingTangent: return
            "Shading tangent in RGB color";
        case SceneDebuggerMode::ShadingBitangent: return
            "Shading bitangent in RGB color";
        case SceneDebuggerMode::FrontFacingFlag: return
            "Green = front-facing\n"
            "Red = back-facing";
        case SceneDebuggerMode::BackfacingShadingNormal: return
            "Pixels where the shading normal is back-facing with respect to view vector are highlighted";
        case SceneDebuggerMode::TexCoords: return
            "Texture coordinates in RG color wrapped to [0,1]";
        case SceneDebuggerMode::HitType: return
            "Hit type in pseudocolor";
        case SceneDebuggerMode::InstanceID: return
            "Instance ID in pseudocolor";
        case SceneDebuggerMode::MaterialID: return
            "Material ID in pseudocolor";
        case SceneDebuggerMode::GeometryID: return
            "Geometry ID in pseudocolor";
        case SceneDebuggerMode::BlasID: return
            "Raytracing bottom-level acceleration structure (BLAS) ID in pseudocolor";
        case SceneDebuggerMode::InstancedGeometry: return
            "Green = instanced geometry\n"
            "Red = non-instanced geometry";
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
        pybind11::enum_<SceneDebuggerMode> mode(m, "SceneDebuggerMode");
        mode.value("FaceNormal", SceneDebuggerMode::FaceNormal);
        mode.value("ShadingNormal", SceneDebuggerMode::ShadingNormal);
        mode.value("ShadingTangent", SceneDebuggerMode::ShadingTangent);
        mode.value("ShadingBitangent", SceneDebuggerMode::ShadingBitangent);
        mode.value("FrontFacingFlag", SceneDebuggerMode::FrontFacingFlag);
        mode.value("BackfacingShadingNormal", SceneDebuggerMode::BackfacingShadingNormal);
        mode.value("TexCoords", SceneDebuggerMode::TexCoords);
        mode.value("HitType", SceneDebuggerMode::HitType);
        mode.value("InstanceID", SceneDebuggerMode::InstanceID);
        mode.value("MaterialID", SceneDebuggerMode::MaterialID);
        mode.value("GeometryID", SceneDebuggerMode::GeometryID);
        mode.value("BlasID", SceneDebuggerMode::BlasID);
        mode.value("InstancedGeometry", SceneDebuggerMode::InstancedGeometry);

        pybind11::class_<SceneDebugger, RenderPass, SceneDebugger::SharedPtr> pass(m, "SceneDebugger");
        pass.def_property(kMode, &SceneDebugger::getMode, &SceneDebugger::setMode);
    }
}

// Don't remove this. it's required for hot-reload to function properly
extern "C" FALCOR_API_EXPORT const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" FALCOR_API_EXPORT void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerPass(SceneDebugger::kInfo, SceneDebugger::create);
    Falcor::ScriptBindings::registerBinding(registerBindings);
}

SceneDebugger::SharedPtr SceneDebugger::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new SceneDebugger(dict));
}

SceneDebugger::SceneDebugger(const Dictionary& dict)
    : RenderPass(kInfo)
{
    if (!gpDevice->isFeatureSupported(Device::SupportedFeatures::RaytracingTier1_1))
    {
        throw RuntimeError("SceneDebugger: Raytracing Tier 1.1 is not supported by the current device");
    }

    // Parse dictionary.
    for (const auto& [key, value] : dict)
    {
        if (key == kMode) mParams.mode = (uint32_t)value;
        else if (key == kShowVolumes) mParams.showVolumes = value;
        else logWarning("Unknown field '{}' in a SceneDebugger dictionary.", key);
    }

    Program::Desc desc;
    desc.addShaderLibrary(kShaderFile).csEntry("main").setShaderModel(kShaderModel);
    mpDebugPass = ComputePass::create(desc, Program::DefineList(), false);
    mpFence = GpuFence::create();
}

Dictionary SceneDebugger::getScriptingDictionary()
{
    Dictionary d;
    d[kMode] = SceneDebuggerMode(mParams.mode);
    d[kShowVolumes] = mParams.showVolumes;
    return d;
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

void SceneDebugger::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    mpScene = pScene;
    mpMeshToBlasID = nullptr;

    if (mpScene)
    {
        // Prepare our programs for the scene.
        Shader::DefineList defines = mpScene->getSceneDefines();

        mpDebugPass->getProgram()->addDefines(defines);
        mpDebugPass->getProgram()->setTypeConformances(mpScene->getTypeConformances());
        mpDebugPass->setVars(nullptr); // Trigger recompile

        // Create lookup table for mesh to BLAS ID.
        auto blasIDs = mpScene->getMeshBlasIDs();
        if (!blasIDs.empty())
        {
            mpMeshToBlasID = Buffer::createStructured(sizeof(uint32_t), (uint32_t)blasIDs.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, blasIDs.data(), false);
        }

        // Create instance metadata.
        initInstanceInfo();

        // Bind variables.
        auto var = mpDebugPass->getRootVar()["CB"]["gSceneDebugger"];
        if (!mpPixelData)
        {
            mpPixelData = Buffer::createStructured(var["pixelData"], 1, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
            mpPixelDataStaging = Buffer::createStructured(var["pixelData"], 1, ResourceBindFlags::None, Buffer::CpuAccess::Read, nullptr, false);
        }
        var["pixelData"] = mpPixelData;
        var["meshToBlasID"] = mpMeshToBlasID;
        var["instanceInfo"] = mpInstanceInfo;
    }
}

void SceneDebugger::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    mPixelDataAvailable = false;
    const auto& pOutput = renderData[kOutput]->asTexture();

    if (mpScene == nullptr)
    {
        pRenderContext->clearUAV(pOutput->getUAV().get(), float4(0.f));
        return;
    }

    mpScene->setRaytracingShaderData(pRenderContext, mpDebugPass->getRootVar());

    ShaderVar var = mpDebugPass->getRootVar()["CB"]["gSceneDebugger"];
    var["params"].setBlob(mParams);
    var["output"] = pOutput;

    mpDebugPass->execute(pRenderContext, uint3(mParams.frameDim, 1));

    pRenderContext->copyResource(mpPixelDataStaging.get(), mpPixelData.get());
    pRenderContext->flush(false);
    mpFence->gpuSignal(pRenderContext->getLowLevelData()->getCommandQueue());

    mPixelDataAvailable = true;
    mParams.frameCount++;
}

void SceneDebugger::renderUI(Gui::Widgets& widget)
{
    widget.dropdown("Mode", kModeList, mParams.mode);
    widget.tooltip("Selects visualization mode");

    widget.checkbox("Clamp to [0,1]", mParams.clamp);
    widget.tooltip("Clamp pixel values to [0,1] before output.");

    if ((SceneDebuggerMode)mParams.mode == SceneDebuggerMode::FaceNormal ||
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
    widget.dummy("#spacer0", { 1, 20 });
    widget.var("Selected pixel", mParams.selectedPixel);

    renderPixelDataUI(widget);

    widget.dummy("#spacer1", { 1, 20 });
    widget.text("Scene: " + (mpScene ? mpScene->getPath().string() : "No scene loaded"));
}

void SceneDebugger::renderPixelDataUI(Gui::Widgets& widget)
{
    if (!mPixelDataAvailable) return;

    FALCOR_ASSERT(mpPixelDataStaging);
    mpFence->syncCpu();
    const PixelData& data = *reinterpret_cast<const PixelData*>(mpPixelDataStaging->map(Buffer::MapType::Read));

    std::ostringstream oss;

    switch ((HitType)data.hitType)
    {
    case HitType::Triangle:
        oss << "Mesh ID: " << data.geometryID << std::endl
            << "Mesh name: " << (mpScene->hasMesh(data.geometryID) ? mpScene->getMeshName(data.geometryID) : "unknown") << std::endl
            << "Instance ID: " << data.instanceID << std::endl
            << "Material ID: " << data.materialID << std::endl
            << "BLAS ID: " << data.blasID << std::endl;

        widget.text(oss.str());
        widget.dummy("#spacer2", { 1, 10 });

        // Show mesh details.
        if (auto g = widget.group("Mesh info"); g.open())
        {
            FALCOR_ASSERT(data.geometryID < mpScene->getMeshCount());
            const auto& mesh = mpScene->getMesh(data.geometryID);
            std::ostringstream oss;
            oss << "flags: " << std::hex << std::showbase << mesh.flags << std::dec << std::noshowbase << std::endl
                << "materialID: " << mesh.materialID << std::endl
                << "vertexCount: " << mesh.vertexCount << std::endl
                << "indexCount: " << mesh.indexCount << std::endl
                << "triangleCount: " << mesh.getTriangleCount() << std::endl
                << "vbOffset: " << mesh.vbOffset << std::endl
                << "ibOffset: " << mesh.ibOffset << std::endl
                << "skinningVbOffset: " << mesh.skinningVbOffset << std::endl
                << "prevVbOffset: " << mesh.prevVbOffset << std::endl
                << "use16BitIndices: " << mesh.use16BitIndices() << std::endl
                << "isFrontFaceCW: " << mesh.isFrontFaceCW() << std::endl;
            g.text(oss.str());
        }

        // Show mesh instance info.
        if (auto g = widget.group("Mesh instance info"); g.open())
        {
            FALCOR_ASSERT(data.instanceID < mpScene->getGeometryInstanceCount());
            const auto& instance = mpScene->getGeometryInstance(data.instanceID);
            std::ostringstream oss;
            oss << "flags: " << std::hex << std::showbase << instance.flags << std::dec << std::noshowbase << std::endl
                << "nodeID: " << instance.globalMatrixID << std::endl
                << "meshID: " << instance.geometryID << std::endl
                << "materialID: " << instance.materialID << std::endl
                << "vbOffset: " << instance.vbOffset << std::endl
                << "ibOffset: " << instance.ibOffset << std::endl
                << "isDynamic: " << instance.isDynamic() << std::endl;
            g.text(oss.str());

            // Print the list of scene graph nodes affecting this mesh instance.
            std::vector<uint32_t> nodes;
            auto nodeID = instance.globalMatrixID;
            while (nodeID != Scene::kInvalidNode)
            {
                nodes.push_back(nodeID);
                nodeID = mpScene->getParentNodeID(nodeID);
            }
            FALCOR_ASSERT(!nodes.empty());

            g.text("Scene graph (root first):");
            const auto& localMatrices = mpScene->getAnimationController()->getLocalMatrices();
            for (auto it = nodes.rbegin(); it != nodes.rend(); it++)
            {
                auto nodeID = *it;
                glm::mat4 mat = glm::transpose(localMatrices[nodeID]);
                if (auto nodeGroup = widget.group("ID " + std::to_string(nodeID)); nodeGroup.open())
                {
                    g.matrix("", mat);
                }
            }
        }

        // Show material info.
        if (auto g = widget.group("Material info"); g.open())
        {
            const auto& material = *mpScene->getMaterial(data.materialID);
            std::ostringstream oss;
            oss << "name: " << material.getName() << std::endl
                << "emissive: " << (material.isEmissive() ? "true" : "false") << std::endl
                << std::endl
                << "See Scene Settings->Materials for more details" << std::endl;
            g.text(oss.str());
        }
        break;
    case HitType::Curve:
        oss << "Curve ID: " << data.geometryID << std::endl
            << "Instance ID: " << data.instanceID << std::endl
            << "Material ID: " << data.materialID << std::endl
            << "BLAS ID: " << data.blasID << std::endl;

        widget.text(oss.str());
        widget.dummy("#spacer2", { 1, 10 });

        // Show mesh details.
        if (auto g = widget.group("Curve info"); g.open())
        {
            const auto& curve = mpScene->getCurve(data.geometryID);
            std::ostringstream oss;
            oss << "degree: " << curve.degree << std::endl
                << "vertexCount: " << curve.vertexCount << std::endl
                << "indexCount: " << curve.indexCount << std::endl
                << "vbOffset: " << curve.vbOffset << std::endl
                << "ibOffset: " << curve.ibOffset << std::endl;
            g.text(oss.str());
        }

        // Show material info.
        if (auto g = widget.group("Material info"); g.open())
        {
            const auto& material = *mpScene->getMaterial(data.materialID);
            std::ostringstream oss;
            oss << "name: " << material.getName() << std::endl
                << std::endl
                << "See Scene Settings->Materials for more details" << std::endl;
            g.text(oss.str());
        }
        break;
    case HitType::SDFGrid:
        oss << "SDF Grid ID: " << data.geometryID << std::endl
            << "Instance ID: " << data.instanceID << std::endl
            << "Material ID: " << data.materialID << std::endl
            << "BLAS ID: " << data.blasID << std::endl;
        widget.text(oss.str());
        widget.dummy("#spacer2", { 1, 10 });

        // Show SDF grid details.
        if (auto g = widget.group("SDF grid info"); g.open())
        {
            const SDFGrid::SharedPtr& pSDFGrid = mpScene->getSDFGrid(data.geometryID);
            std::ostringstream oss;
            oss << "gridWidth: " << pSDFGrid->getGridWidth() << std::endl;
            g.text(oss.str());
        }

        // Show material info.
        if (auto g = widget.group("Material info"); g.open())
        {
            const auto& material = *mpScene->getMaterial(data.materialID);
            std::ostringstream oss;
            oss << "name: " << material.getName() << std::endl
                << std::endl
                << "See Scene Settings->Materials for more details" << std::endl;
            g.text(oss.str());
        }
        break;
    default:
        oss << "Background pixel" << std::endl;
        break;
    }

    mpPixelDataStaging->unmap();
}

bool SceneDebugger::onMouseEvent(const MouseEvent& mouseEvent)
{
    if (mouseEvent.type == MouseEvent::Type::ButtonDown && mouseEvent.button == Input::MouseButton::Left)
    {
        float2 cursorPos = mouseEvent.pos * (float2)mParams.frameDim;
        mParams.selectedPixel = (uint2)glm::clamp(cursorPos, float2(0.f), float2(mParams.frameDim.x - 1, mParams.frameDim.y - 1));
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
    for (auto& counts : instanceCounts) counts.resize(mpScene->getGeometryCount());

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
    mpInstanceInfo = Buffer::createStructured(sizeof(InstanceInfo), (uint32_t)instanceInfo.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, instanceInfo.data(), false);
}
