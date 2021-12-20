/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include "BSDFViewer.h"
#include "Rendering/Materials/BxDFConfig.slangh"

const RenderPass::Info BSDFViewer::kInfo { "BSDFViewer", "BSDF inspection utility." };


// Don't remove this. it's required for hot-reload to function properly
extern "C" FALCOR_API_EXPORT const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" FALCOR_API_EXPORT void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerPass(BSDFViewer::kInfo, BSDFViewer::create);
}

namespace
{
    const char kFileViewerPass[] = "RenderPasses/BSDFViewer/BSDFViewer.cs.slang";
    const char kOutput[] = "output";

    // Scripting options.
    const char kMaterialID[] = "materialID";
}

BSDFViewer::SharedPtr BSDFViewer::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new BSDFViewer(dict));
}

BSDFViewer::BSDFViewer(const Dictionary& dict)
    : RenderPass(kInfo)
{
    parseDictionary(dict);

    // Create a high-quality pseudorandom number generator.
    mpSampleGenerator = SampleGenerator::create(SAMPLE_GENERATOR_UNIFORM);

    // Set defines for the program. The defines are updated later if a scene is loaded.
    Program::DefineList defines;
    defines.add(mpSampleGenerator->getDefines());

    // Create program.
    mpViewerPass = ComputePass::create(kFileViewerPass, "main", defines, false);

    mpPixelDebug = PixelDebug::create();
    mpFence = GpuFence::create();
}

void BSDFViewer::parseDictionary(const Dictionary& dict)
{
    for (const auto& [key, value] : dict)
    {
        if (key == kMaterialID) mParams.materialID = value;
        else logWarning("Unknown field '" + key + "' in BSDFViewer dictionary");
    }
}

Dictionary BSDFViewer::getScriptingDictionary()
{
    Dictionary d;
    d[kMaterialID] = mParams.materialID;
    return d;
}

RenderPassReflection BSDFViewer::reflect(const CompileData& compileData)
{
    RenderPassReflection r;
    r.addOutput(kOutput, "Output buffer").format(ResourceFormat::RGBA32Float).bindFlags(ResourceBindFlags::UnorderedAccess);
    return r;
}

void BSDFViewer::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    mParams.frameDim = compileData.defaultTexDims;

    // Place a square viewport centered in the frame.
    uint32_t extent = std::min(mParams.frameDim.x, mParams.frameDim.y);
    uint32_t xOffset = (mParams.frameDim.x - extent) / 2;
    uint32_t yOffset = (mParams.frameDim.y - extent) / 2;

    mParams.viewportOffset = float2(xOffset, yOffset);
    mParams.viewportScale = float2(1.f / extent);
}

void BSDFViewer::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    mpScene = pScene;
    mpEnvMap = nullptr;
    mMaterialList.clear();
    mPixelDataValid = mPixelDataAvailable = false;

    if (mpScene != nullptr)
    {
        // Configure program for the scene.
        mpViewerPass->getProgram()->addDefines(mpScene->getSceneDefines());
        mpViewerPass->getProgram()->setTypeConformances(mpScene->getTypeConformances());

        // Compile program and bind the scene.
        mpViewerPass->setVars(nullptr); // Trigger vars creation
        mpViewerPass["gScene"] = mpScene->getParameterBlock();

        // Load and bind environment map.
        if (const auto &pEnvMap = mpScene->getEnvMap()) loadEnvMap(pEnvMap->getFilename());
        mParams.useEnvMap = mpEnvMap != nullptr;

        // Setup material ID.
        uint32_t materialCount = mpScene->getMaterialCount();
        if (materialCount == 0) RuntimeError("BSDFViewer: No support for scenes without materials");
        if (mParams.materialID >= materialCount)
        {
            mParams.materialID = materialCount - 1;
            logWarning("BSDFViewer: materialID is out of range. Clamping to ID " + std::to_string(mParams.materialID) + ".");
        }

        // Prepare UI list of materials.
        mMaterialList.reserve(materialCount);
        for (uint32_t i = 0; i < materialCount; i++)
        {
            auto mtl = mpScene->getMaterial(i);
            std::string name = std::to_string(i) + ": " + mtl->getName();
            mMaterialList.push_back({ i, name });
        }
        assert(mMaterialList.size() > 0);
    }
}

void BSDFViewer::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Update refresh flag if options that affect the output have changed.
    if (mOptionsChanged)
    {
        auto& dict = renderData.getDictionary();
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    auto pOutput = renderData[kOutput]->asTexture();
    if (!mpScene)
    {
        pRenderContext->clearUAV(pOutput->getUAV().get(), uint4(0));
        return;
    }

    // Read back pixel data from the previous frame if available.
    // This ensures parameters get updated even if the UI wasn't rendered.
    readPixelData();

    // Set compile-time constants.
    if (mParams.useDisneyDiffuse) mpViewerPass->addDefine("DiffuseBrdf", "DiffuseBrdfDisney");
    else mpViewerPass->removeDefine("DiffuseBrdf");
    if (mParams.useSeparableMaskingShadowing) mpViewerPass->addDefine("SpecularMaskingFunction", "SpecularMaskingFunctionSmithGGXSeparable");
    else mpViewerPass->removeDefine("SpecularMaskingFunction");

    // Setup constants.
    mParams.cameraViewportScale = std::tan(glm::radians(mParams.cameraFovY / 2.f)) * mParams.cameraDistance;

    // Set resources.
    mpSampleGenerator->setShaderData(mpViewerPass->getVars()->getRootVar());
    mpViewerPass["gOutput"] = pOutput;
    mpViewerPass["PerFrameCB"]["gParams"].setBlob(mParams);

    if (!mpPixelDataBuffer)
    {
        mpPixelDataBuffer = Buffer::createStructured(mpViewerPass->getProgram().get(), "gPixelData", 1, ResourceBindFlags::UnorderedAccess);
        mpPixelStagingBuffer = Buffer::createStructured(mpViewerPass->getProgram().get(), "gPixelData", 1, ResourceBindFlags::None, Buffer::CpuAccess::Read);
    }
    mpViewerPass["gPixelData"] = mpPixelDataBuffer;

    mpPixelDebug->beginFrame(pRenderContext, renderData.getDefaultTextureDims());
    mpPixelDebug->prepareProgram(mpViewerPass->getProgram(), mpViewerPass->getRootVar());

    // Execute pass.
    mpViewerPass->execute(pRenderContext, uint3(mParams.frameDim, 1));

    // Copy pixel data to staging buffer for readback.
    // This is to avoid a full flush and the associated perf warning.
    pRenderContext->copyResource(mpPixelStagingBuffer.get(), mpPixelDataBuffer.get());
    pRenderContext->flush(false);
    mpFence->gpuSignal(pRenderContext->getLowLevelData()->getCommandQueue());
    mPixelDataAvailable = true;
    mPixelDataValid = false;

    mpPixelDebug->endFrame(pRenderContext);
    mParams.frameCount++;
}

void BSDFViewer::readPixelData()
{
    if (mPixelDataAvailable)
    {
        mpFence->syncCpu();
        assert(mpPixelStagingBuffer);
        mPixelData = *static_cast<const PixelData*>(mpPixelStagingBuffer->map(Buffer::MapType::Read));
        mpPixelStagingBuffer->unmap();

        mPixelDataAvailable = false;
        mPixelDataValid = true;

        // Update parameters from values at selected pixel.
        mParams.texCoords = mPixelData.texC;
    }
}

void BSDFViewer::renderUI(Gui::Widgets& widget)
{
    if (!mpScene)
    {
        widget.text("No scene loaded");
        return;
    }

    bool dirty = false;

    dirty |= widget.checkbox("Enable BSDF slice viewer", mParams.enableSliceViewer);
    widget.tooltip("Run BSDF slice viewer.\nOtherise the default mode shows a shaded sphere of the specified material.", true);

    if (mParams.enableSliceViewer)
    {
        widget.text("The current mode shows a slice of the BSDF.\n"
                    "The x-axis is theta_h (angle between H and normal)\n"
                    "and y-axis is theta_d (angle between H and wi/wo),\n"
                    "both in [0,pi/2] with origin in the lower/left.");
    }
    else
    {
        widget.text("The current mode shows a shaded unit sphere.\n"
                    "The coordinate frame is right-handed with xy\n"
                    "pointing right/up and +z towards the viewer.\n"
                    " ");
    }

    if (auto mtlGroup = widget.group("Material", true))
    {
        mtlGroup.tooltip("Choose material in the dropdown below.\n\n"
            "Left/right arrow keys step to the previous/next material in the list.", true);

        assert(mMaterialList.size() > 0);
        dirty |= mtlGroup.dropdown("Materials", mMaterialList, mParams.materialID);

        auto type = mpScene->getMaterial(mParams.materialID)->getType();
        mtlGroup.text("Material type: " + to_string(type));

        dirty |= mtlGroup.checkbox("Normal mapping", mParams.useNormalMapping);
        dirty |= mtlGroup.checkbox("Fixed tex coords", mParams.useFixedTexCoords);
        dirty |= mtlGroup.var("Tex coords", mParams.texCoords, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), 0.01f);
    }

    if (auto bsdfGroup = widget.group("BSDF", true))
    {
        dirty |= bsdfGroup.checkbox("Enable diffuse", mParams.enableDiffuse);
        dirty |= bsdfGroup.checkbox("Enable specular", mParams.enableSpecular, true);

        dirty |= bsdfGroup.checkbox("Use Disney' diffuse BRDF", mParams.useDisneyDiffuse);
        bsdfGroup.tooltip("When enabled uses the original Disney diffuse BRDF, otherwise use Falcor's default (Frostbite's version).", true);
        dirty |= bsdfGroup.checkbox("Use separable masking-shadowing", mParams.useSeparableMaskingShadowing);
        bsdfGroup.tooltip("Use the separable form of Smith's masking-shadowing function which is used by the original Disney BRDF, otherwise use Falcor's default (the correlated form).", true);

        dirty |= bsdfGroup.checkbox("Use importance sampling", mParams.useImportanceSampling);
        bsdfGroup.tooltip("When enabled uses BSDF importance sampling, otherwise hemispherical cosine-weighted sampling for verification purposes.", true);
        dirty |= bsdfGroup.checkbox("Use pdf", mParams.usePdf);
        bsdfGroup.tooltip("When enabled evaluates BRDF * NdotL / pdf explicitly for verification purposes.\nOtherwise the weight computed by the importance sampling is used.", true);

        if (mParams.enableSliceViewer)
        {
            bsdfGroup.dummy("#space1", float2(1, 8));
            bsdfGroup.text("Slice viewer settings:");

            dirty |= bsdfGroup.checkbox("Multiply BSDF slice by NdotL", mParams.applyNdotL);
            bsdfGroup.tooltip("Note: This setting Only affects the BSDF slice viewer. NdotL is always enabled in lighting mode.", true);
        }
    }

    if (auto lightGroup = widget.group("Light", true))
    {
        dirty |= lightGroup.var("Light intensity", mParams.lightIntensity, 0.f, std::numeric_limits<float>::max(), 0.01f, false, "%.4f");
        dirty |= lightGroup.rgbColor("Light color", mParams.lightColor);
        lightGroup.tooltip("Not used when environment map is enabled.", true);

        dirty |= lightGroup.checkbox("Show ground plane", mParams.useGroundPlane);
        lightGroup.tooltip("When the ground plane is enabled, incident illumination from the lower hemisphere is zero.", true);

        // Directional lighting
        dirty |= lightGroup.checkbox("Directional light", mParams.useDirectionalLight);
        lightGroup.tooltip("When enabled a single directional light source is used, otherwise the light is omnidirectional.", true);

        if (mParams.useDirectionalLight)
        {
            mParams.useEnvMap = false;
            dirty |= lightGroup.var("Light direction", mParams.lightDir, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), 0.01f, false, "%.4f");
        }

        // Envmap lighting
        if (mpEnvMap)
        {
            dirty |= lightGroup.checkbox(("Environment map: " + mpEnvMap->getFilename()).c_str(), mParams.useEnvMap);
            lightGroup.tooltip("When enabled the specified environment map is used as light source. Enabling this option turns off directional lighting.", true);

            if (mParams.useEnvMap)
            {
                mParams.useDirectionalLight = false;
            }
        }
        else
        {
            lightGroup.text("Environment map: N/A");
        }

        if (lightGroup.button("Load environment map"))
        {
            std::string filename;
            if (openFileDialog(Bitmap::getFileDialogFilters(), filename))
            {
                // TODO: RenderContext* should maybe be a parameter to renderUI()?
                if (loadEnvMap(filename))
                {
                    mParams.useDirectionalLight = false;
                    mParams.useEnvMap = true;
                    dirty = true;
                }
            }
        }
    }

    if (auto cameraGroup = widget.group("Camera", true))
    {
        dirty |= cameraGroup.checkbox("Orthographic camera", mParams.orthographicCamera);

        if (!mParams.orthographicCamera)
        {
            dirty |= cameraGroup.var("Viewing distance", mParams.cameraDistance, 1.01f, std::numeric_limits<float>::max(), 0.01f, false, "%.2f");
            cameraGroup.tooltip("This is the camera's distance to origin in projective mode. The scene has radius 1.0 so the minimum camera distance has to be > 1.0", true);

            dirty |= cameraGroup.var("Vertical FOV (degrees)", mParams.cameraFovY, 1.f, 179.f, 1.f, false, "%.2f");
            cameraGroup.tooltip("The allowed range is [1,179] degrees to avoid numerical issues.", true);
        }
    }

    if (auto pixelGroup = widget.group("Pixel data", true))
    {
        // Read back data from the current frame when UI is shown.
        readPixelData();

        pixelGroup.var("Pixel", mParams.selectedPixel);

        if (mPixelDataValid)
        {
            pixelGroup.var("texC", mPixelData.texC, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), 0.f, false, "%.4f");
            pixelGroup.var("T", mPixelData.T, -1.f, 1.f, 0.f, false, "%.4f");
            pixelGroup.var("B", mPixelData.B, -1.f, 1.f, 0.f, false, "%.4f");
            pixelGroup.var("N", mPixelData.N, -1.f, 1.f, 0.f, false, "%.4f");
            pixelGroup.var("wi", mPixelData.wi, -1.f, 1.f, 0.f, false, "%.4f");
            pixelGroup.tooltip("Incident direction (view dir)", true);
            pixelGroup.var("wo", mPixelData.wo, -1.f, 1.f, 0.f, false, "%.4f");
            pixelGroup.tooltip("Outgoing direction (light dir)", true);
            pixelGroup.var("output", mPixelData.output, 0.f, std::numeric_limits<float>::max(), 0.f, false, "%.4f");

            pixelGroup.text("BSDF properties:");
            pixelGroup.var("emissive", mPixelData.emissive, 0.f, 1.f, 0.f, false, "%.4f");
            pixelGroup.var("roughness", mPixelData.roughness, 0.f, 1.f, 0.f, false, "%.4f");
            pixelGroup.var("diffuseReflectionAlbedo", mPixelData.diffuseReflectionAlbedo, 0.f, std::numeric_limits<float>::max(), 0.f, false, "%.4f");
            pixelGroup.var("diffuseTransmissionAlbedo", mPixelData.diffuseTransmissionAlbedo, 0.f, std::numeric_limits<float>::max(), 0.f, false, "%.4f");
            pixelGroup.var("specularReflectionAlbedo", mPixelData.specularReflectionAlbedo, 0.f, std::numeric_limits<float>::max(), 0.f, false, "%.4f");
            pixelGroup.var("specularTransmissionAlbedo", mPixelData.specularTransmissionAlbedo, 0.f, std::numeric_limits<float>::max(), 0.f, false, "%.4f");
            pixelGroup.var("specularReflectance", mPixelData.specularReflectance, 0.f, std::numeric_limits<float>::max(), 0.f, false, "%.4f");
            pixelGroup.checkbox("isTransmissive", mPixelData.isTransmissive);
        }
        else
        {
            pixelGroup.text("No data available");
        }
    }

    if (auto loggingGroup = widget.group("Logging", false))
    {
        mpPixelDebug->renderUI(widget);
    }

    if (dirty)
    {
        mOptionsChanged = true;
    }
}

bool BSDFViewer::onMouseEvent(const MouseEvent& mouseEvent)
{
    if (mouseEvent.type == MouseEvent::Type::LeftButtonDown)
    {
        mParams.selectedPixel = glm::clamp((int2)(mouseEvent.pos * (float2)mParams.frameDim), { 0,0 }, (int2)mParams.frameDim - 1);
    }

    return mpPixelDebug->onMouseEvent(mouseEvent);
}

bool BSDFViewer::onKeyEvent(const KeyboardEvent& keyEvent)
{
    if (keyEvent.type == KeyboardEvent::Type::KeyPressed)
    {
        if (keyEvent.key == KeyboardEvent::Key::Left || keyEvent.key == KeyboardEvent::Key::Right)
        {
            uint32_t id = mParams.materialID;
            uint32_t lastId = mMaterialList.size() > 0 ? (uint32_t)mMaterialList.size() - 1 : 0;
            if (keyEvent.key == KeyboardEvent::Key::Left) id = id > 0 ? id - 1 : lastId;
            else if (keyEvent.key == KeyboardEvent::Key::Right) id = id < lastId ? id + 1 : 0;

            if (id != mParams.materialID) mOptionsChanged = true; // Triggers reset of accumulation
            mParams.materialID = id;
            return true;
        }
    }
    return false;
}

bool BSDFViewer::loadEnvMap(const std::string& filename)
{
    mpEnvMap = EnvMap::create(filename);
    if (!mpEnvMap)
    {
        logWarning("Failed to load environment map from " + filename);
        return false;
    }

    auto pVars = mpViewerPass->getVars();
    mpEnvMap->setShaderData(pVars["PerFrameCB"]["gEnvMap"]);

    return true;
}
