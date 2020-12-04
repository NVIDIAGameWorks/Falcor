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
#include "SceneDebugger.h"

namespace
{
    const char kDesc[] = "Scene debugger for identifying asset issues.";
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
        { (uint32_t)SceneDebuggerMode::MeshID, "Mesh ID" },
        { (uint32_t)SceneDebuggerMode::MeshInstanceID, "Mesh instance ID" },
        { (uint32_t)SceneDebuggerMode::MaterialID, "Material ID" },
        { (uint32_t)SceneDebuggerMode::BlasID, "BLAS ID" },
        { (uint32_t)SceneDebuggerMode::CurveID, "Curve ID" },
        { (uint32_t)SceneDebuggerMode::CurveInstanceID, "Curve instance ID" },
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
        case SceneDebuggerMode::MeshID: return
            "Mesh ID in pseudocolor";
        case SceneDebuggerMode::MeshInstanceID: return
            "Mesh instance ID in pseudocolor";
        case SceneDebuggerMode::MaterialID: return
            "Material ID in pseudocolor";
        case SceneDebuggerMode::BlasID: return
            "Raytracing bottom-level acceleration structure (BLAS) ID in pseudocolor";
        case SceneDebuggerMode::CurveID: return
            "Curve ID in pseudocolor";
        case SceneDebuggerMode::CurveInstanceID: return
            "Curve instance ID in pseudocolor";
        case SceneDebuggerMode::InstancedGeometry: return
            "Green = instanced geometry\n"
            "Red = non-instanced geometry";
        default:
            should_not_get_here();
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
        mode.value("MeshID", SceneDebuggerMode::MeshID);
        mode.value("MeshInstanceID", SceneDebuggerMode::MeshInstanceID);
        mode.value("MaterialID", SceneDebuggerMode::MaterialID);
        mode.value("BlasID", SceneDebuggerMode::BlasID);
        mode.value("CurveID", SceneDebuggerMode::CurveID);
        mode.value("CurveInstanceID", SceneDebuggerMode::CurveInstanceID);
        mode.value("InstancedGeometry", SceneDebuggerMode::InstancedGeometry);
    }
}

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerClass("SceneDebugger", kDesc, SceneDebugger::create);
    Falcor::ScriptBindings::registerBinding(registerBindings);
}

SceneDebugger::SharedPtr SceneDebugger::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new SceneDebugger(dict));
}

SceneDebugger::SceneDebugger(const Dictionary& dict)
{
    if (!gpDevice->isFeatureSupported(Device::SupportedFeatures::RaytracingTier1_1))
    {
        throw std::exception("Raytracing Tier 1.1 is not supported by the current device");
    }

    // Parse dictionary.
    for (const auto& [key, value] : dict)
    {
        if (key == kMode) mParams.mode = (uint32_t)value;
        else if (key == kShowVolumes) mParams.showVolumes = value;
        else logWarning("Unknown field '" + key + "' in a SceneDebugger dictionary");
    }

    Program::Desc desc;
    desc.addShaderLibrary(kShaderFile).csEntry("main").setShaderModel(kShaderModel);
    mpDebugPass = ComputePass::create(desc, Program::DefineList(), false);
    mpFence = GpuFence::create();
}

std::string SceneDebugger::getDesc()
{
    return kDesc;
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

void SceneDebugger::compile(RenderContext* pContext, const CompileData& compileData)
{
    mParams.frameDim = compileData.defaultTexDims;
}

void SceneDebugger::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    mpScene = pScene;

    if (mpScene)
    {
        // Prepare our programs for the scene.
        Shader::DefineList defines = mpScene->getSceneDefines();

        // Disable discard and gradient operations.
        defines.add("_MS_DISABLE_ALPHA_TEST");
        defines.add("_DEFAULT_ALPHA_TEST");

        mpDebugPass->getProgram()->addDefines(defines);
        mpDebugPass->setVars(nullptr); // Trigger recompile

        // Create lookup table for mesh to BLAS ID.
        auto blasIDs = mpScene->getMeshBlasIDs();
        assert(!blasIDs.empty());
        mpMeshToBlasID = Buffer::createStructured(sizeof(uint32_t), (uint32_t)blasIDs.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, blasIDs.data(), false);

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
        var["meshInstanceInfo"] = mpMeshInstanceInfo;
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
    widget.text("Scene: " + (mpScene ? mpScene->getFilename() : "No scene loaded"));
}

void SceneDebugger::renderPixelDataUI(Gui::Widgets& widget)
{
    if (mPixelDataAvailable)
    {
        assert(mpPixelDataStaging);
        mpFence->syncCpu();
        const PixelData& data = *reinterpret_cast<const PixelData*>(mpPixelDataStaging->map(Buffer::MapType::Read));

        std::ostringstream oss;
        if (data.meshInstanceID != PixelData::kInvalidID)
        {
            oss << "Mesh ID: " << data.meshID << std::endl
                << "Mesh name: " << (mpScene->hasMesh(data.meshID) ? mpScene->getMeshName(data.meshID) : "unknown") << std::endl
                << "Mesh instance ID: " << data.meshInstanceID << std::endl
                << "Material ID: " << data.materialID << std::endl
                << "BLAS ID: " << data.blasID << std::endl;

            widget.text(oss.str());
            widget.dummy("#spacer2", { 1, 10 });

            // Show mesh details.
            if (auto g = widget.group("Mesh info"); g.open())
            {
                const auto& mesh = mpScene->getMesh(data.meshID);
                std::ostringstream oss;
                oss << "flags: " << mesh.flags << std::endl
                    << "vertexCount: " << mesh.vertexCount << std::endl
                    << "indexCount: " << mesh.indexCount << std::endl
                    << "triangleCount: " << mesh.getTriangleCount() << std::endl
                    << "vbOffset: " << mesh.vbOffset << std::endl
                    << "ibOffset: " << mesh.ibOffset << std::endl
                    << "use16BitIndices: " << mesh.use16BitIndices() << std::endl;
                g.text(oss.str());
                g.release();
            }

            // Show material info.
            if (auto g = widget.group("Material info"); g.open())
            {
                const auto& material = *mpScene->getMaterial(data.materialID);
                std::ostringstream oss;
                oss << "name: " << material.getName() << std::endl
                    << "emissive: " << (material.isEmissive() ? "true" : "false") << std::endl
                    << "doubleSided: " << (material.isDoubleSided() ? "true" : "false") << std::endl
                    << std::endl
                    << "See Scene Settings->Materials for more details" << std::endl;
                g.text(oss.str());
                g.release();
            }
        }
        else if (data.curveInstanceID != PixelData::kInvalidID)
        {
            oss << "Curve ID: " << data.curveID << std::endl
                << "Curve instance ID: " << data.curveInstanceID << std::endl
                << "Material ID: " << data.materialID << std::endl
                << "BLAS ID: " << data.blasID << std::endl;

            widget.text(oss.str());
            widget.dummy("#spacer2", { 1, 10 });

            // Show mesh details.
            if (auto g = widget.group("Curve info"); g.open())
            {
                const auto& curve = mpScene->getCurve(data.curveID);
                std::ostringstream oss;
                oss << "degree: " << curve.degree << std::endl
                    << "vertexCount: " << curve.vertexCount << std::endl
                    << "indexCount: " << curve.indexCount << std::endl
                    << "vbOffset: " << curve.vbOffset << std::endl
                    << "ibOffset: " << curve.ibOffset << std::endl;
                g.text(oss.str());
                g.release();
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
                g.release();
            }
        }
        else
        {
            oss << "Background pixel" << std::endl;
        }

        mpPixelDataStaging->unmap();
    }
}

bool SceneDebugger::onMouseEvent(const MouseEvent& mouseEvent)
{
    if (mouseEvent.type == MouseEvent::Type::LeftButtonDown)
    {
        float2 cursorPos = mouseEvent.pos * (float2)mParams.frameDim;
        mParams.selectedPixel = (uint2)glm::clamp(cursorPos, float2(0.f), float2(mParams.frameDim.x - 1, mParams.frameDim.y - 1));
    }

    return false;
}

void SceneDebugger::initInstanceInfo()
{
    const uint32_t instanceCount = mpScene ? mpScene->getMeshInstanceCount() : 0;

    // If there are no mesh instances. Just clear the buffer and return.
    if (instanceCount == 0)
    {
        mpMeshInstanceInfo = nullptr;
        return;
    }

    // Count number of instances of each mesh.
    const uint32_t meshCount = mpScene->getMeshCount();
    assert(meshCount > 0);

    std::vector<size_t> meshInstanceCounts(meshCount, 0);
    for (uint32_t instanceID = 0; instanceID < instanceCount; instanceID++)
    {
        uint32_t meshID = mpScene->getMeshInstance(instanceID).meshID;
        assert(meshID < meshCount);
        meshInstanceCounts[meshID]++;
    }

    // Setup instance metadata.
    std::vector<InstanceInfo> instanceInfo(instanceCount);
    for (uint32_t instanceID = 0; instanceID < instanceCount; instanceID++)
    {
        auto& info = instanceInfo[instanceID];

        uint32_t meshID = mpScene->getMeshInstance(instanceID).meshID;
        if (meshInstanceCounts[meshID] > 1)
        {
            info.flags |= (uint32_t)InstanceInfoFlags::IsInstanced;
        }
    }

    // Create GPU buffer.
    assert(!instanceInfo.empty());
    mpMeshInstanceInfo = Buffer::createStructured(sizeof(InstanceInfo), (uint32_t)instanceInfo.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, instanceInfo.data(), false);
}
