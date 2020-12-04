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
#include "BSDFViewer.h"
#include "Experimental/Scene/Material/BxDFConfig.slangh"

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerClass("BSDFViewer", BSDFViewer::sDesc, BSDFViewer::create);
}

namespace
{
    const char kFileViewerPass[] = "RenderPasses/BSDFViewer/BSDFViewer.cs.slang";
    const char kOutput[] = "output";
}

const char* BSDFViewer::sDesc = "BSDF Viewer";

BSDFViewer::SharedPtr BSDFViewer::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new BSDFViewer(dict));
}

BSDFViewer::BSDFViewer(const Dictionary& dict)
{
    // Create a high-quality pseudorandom number generator.
    mpSampleGenerator = SampleGenerator::create(SAMPLE_GENERATOR_UNIFORM);

    // Defines to disable discard and gradient operations in Falcor's material system.
    Program::DefineList defines =
    {
        {"_MS_DISABLE_ALPHA_TEST", ""},
        {"_DEFAULT_ALPHA_TEST", ""},
        {"SCENE_MATERIAL_COUNT", "1"},
        {"SCENE_GRID_COUNT", "0"},
    };

    defines.add(mpSampleGenerator->getDefines());

    // Create programs.
    mpViewerPass = ComputePass::create(kFileViewerPass, "main", defines);

    // Create readback buffer.
    mPixelDataBuffer = Buffer::createStructured(mpViewerPass->getProgram().get(), "gPixelData", 1u, ResourceBindFlags::UnorderedAccess);

    mpPixelDebug = PixelDebug::create();
}

Dictionary BSDFViewer::getScriptingDictionary()
{
    return Dictionary();
}

RenderPassReflection BSDFViewer::reflect(const CompileData& compileData)
{
    RenderPassReflection r;
    r.addOutput(kOutput, "Output buffer").format(ResourceFormat::RGBA32Float).bindFlags(ResourceBindFlags::UnorderedAccess);
    return r;
}

void BSDFViewer::compile(RenderContext* pContext, const CompileData& compileData)
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
    mParams.materialID = 0;

    if (pScene == nullptr)
    {
        mParams.useSceneMaterial = false;
        mParams.useEnvMap = false;
    }
    else
    {
        mParams.useSceneMaterial = true;

        // Bind the scene to our program.
        mpViewerPass->getProgram()->addDefines(mpScene->getSceneDefines());
        mpViewerPass->setVars(nullptr); // Trigger vars creation
        mpViewerPass["gScene"] = mpScene->getParameterBlock();

        // Load and bind environment map.
        if (const auto &pEnvMap = mpScene->getEnvMap()) loadEnvMap(pEnvMap->getFilename());
        mParams.useEnvMap = mpEnvMap != nullptr;

        // Prepare UI list of materials.
        mMaterialList.reserve(mpScene->getMaterialCount());
        for (uint32_t i = 0; i < mpScene->getMaterialCount(); i++)
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

    // Set compile-time constants.
    mpViewerPass->addDefine("_USE_LEGACY_SHADING_CODE", mParams.useLegacyBSDF ? "1" : "0");

    if (mParams.useDisneyDiffuse) mpViewerPass->addDefine("DiffuseBrdf", "DiffuseBrdfDisney");
    else mpViewerPass->removeDefine("DiffuseBrdf");
    if (mParams.useSeparableMaskingShadowing) mpViewerPass->addDefine("SpecularMaskingFunction", "SpecularMaskingFunctionSmithGGXSeparable");
    else mpViewerPass->removeDefine("SpecularMaskingFunction");

    // Setup constants.
    mParams.cameraViewportScale = std::tan(glm::radians(mParams.cameraFovY / 2.f)) * mParams.cameraDistance;

    // Set resources.
    if (!mpSampleGenerator->setShaderData(mpViewerPass->getVars()->getRootVar())) throw std::exception("Failed to bind sample generator");
    mpViewerPass["gOutput"] = renderData[kOutput]->asTexture();
    mpViewerPass["gPixelData"] = mPixelDataBuffer;
    mpViewerPass["PerFrameCB"]["gParams"].setBlob(mParams);

    mpPixelDebug->beginFrame(pRenderContext, renderData.getDefaultTextureDims());
    mpPixelDebug->prepareProgram(mpViewerPass->getProgram(), mpViewerPass->getRootVar());

    // Execute pass.
    mpViewerPass->execute(pRenderContext, uint3(mParams.frameDim, 1));

    mpPixelDebug->endFrame(pRenderContext);

    mPixelDataValid = false;
    if (mParams.readback)
    {
        const PixelData* pData = static_cast<const PixelData*>(mPixelDataBuffer->map(Buffer::MapType::Read));
        mPixelData = *pData;
        mPixelDataBuffer->unmap();
        mPixelDataValid = true;

        // Copy values from selected pixel.
        mParams.texCoords = mPixelData.texC;
    }

    mParams.frameCount++;
}

void BSDFViewer::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;

    dirty |= widget.checkbox("Enable BSDF slice viewer", mParams.sliceViewer);
    widget.tooltip("Run BSDF slice viewer.\nOtherise the default mode shows a shaded sphere of the specified material.", true);

    if (mParams.sliceViewer)
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
        bool prevMode = mParams.useSceneMaterial;
        mtlGroup.checkbox("Use scene material", mParams.useSceneMaterial);
        mtlGroup.tooltip("Choose material in the dropdown below.\n\n"
            "Left/right arrow keys step to the previous/next material in the list.", true);

        if (!mpScene) mParams.useSceneMaterial = false;
        dirty |= ((bool)mParams.useSceneMaterial != prevMode);

        if (mParams.useSceneMaterial)
        {
            assert(mMaterialList.size() > 0);
            dirty |= mtlGroup.dropdown("Materials", mMaterialList, mParams.materialID);

            dirty |= mtlGroup.checkbox("Normal mapping", mParams.useNormalMapping);
            dirty |= mtlGroup.checkbox("Fixed tex coords", mParams.useFixedTexCoords);
            dirty |= mtlGroup.var("Tex coords", mParams.texCoords, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), 0.01f);
        }
        else
        {
            dirty |= mtlGroup.rgbColor("Base color", mParams.baseColor);
            dirty |= mtlGroup.var("Roughness", mParams.linearRoughness, 0.f, 1.f, 1e-2f);
            dirty |= mtlGroup.var("Metallic", mParams.metallic, 0.f, 1.f, 1e-2f);
            dirty |= mtlGroup.var("IoR", mParams.IoR, 1.f, std::numeric_limits<float>::max(), 1e-2f);
        }
    }

    if (auto bsdfGroup = widget.group("BSDF", true))
    {
        dirty |= bsdfGroup.checkbox("Use legacy BSDF code", mParams.useLegacyBSDF);

        dirty |= bsdfGroup.checkbox("Enable diffuse", mParams.enableDiffuse);
        dirty |= bsdfGroup.checkbox("Enable specular", mParams.enableSpecular, true);

        dirty |= bsdfGroup.checkbox("Use Disney' diffuse BRDF", mParams.useDisneyDiffuse);
        bsdfGroup.tooltip("When enabled uses the original Disney diffuse BRDF, otherwise use Falcor's default (Frostbite's version).", true);
        dirty |= bsdfGroup.checkbox("Use separable masking-shadowing", mParams.useSeparableMaskingShadowing);
        bsdfGroup.tooltip("Use the separable form of Smith's masking-shadowing function which is used by the original Disney BRDF, otherwise use Falcor's default (the correlated form).", true);

        dirty |= bsdfGroup.checkbox("Use BRDF sampling", mParams.useBrdfSampling);
        bsdfGroup.tooltip("When enabled uses BSDF importance sampling, otherwise hemispherical cosine-weighted sampling for verification purposes.", true);
        dirty |= bsdfGroup.checkbox("Use pdf", mParams.usePdf);
        bsdfGroup.tooltip("When enabled evaluates BRDF * NdotL / pdf explicitly for verification purposes.\nOtherwise the weight computed by the importance sampling is used.", true);

        if (mParams.sliceViewer)
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

    mParams.readback = false;

    if (auto pixelGroup = widget.group("Pixel data", true))
    {
        mParams.readback = mParams.useSceneMaterial && !mParams.useFixedTexCoords; // Enable readback if necessary

        pixelGroup.var("Pixel", mParams.selectedPixel);

        if (mPixelDataValid)
        {
            pixelGroup.var("texC", mPixelData.texC, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), 0.f, false, "%.4f");
            pixelGroup.var("baseColor", mPixelData.baseColor, 0.f, 1.f, 0.f, false, "%.4f");
            pixelGroup.var("diffuse", mPixelData.diffuse, 0.f, 1.f, 0.f, false, "%.4f");
            pixelGroup.var("specular", mPixelData.specular, 0.f, 1.f, 0.f, false, "%.4f");
            pixelGroup.var("roughness", mPixelData.linearRoughness, 0.f, 1.f, 0.f, false, "%.4f");
            pixelGroup.tooltip("This is the unmapped roughness parameters as specified in the content creation tool.", true);
            pixelGroup.var("metallic", mPixelData.metallic, 0.f, 1.f, 0.f, false, "%.4f");
            pixelGroup.var("T", mPixelData.T, -1.f, 1.f, 0.f, false, "%.4f");
            pixelGroup.var("B", mPixelData.B, -1.f, 1.f, 0.f, false, "%.4f");
            pixelGroup.var("N", mPixelData.N, -1.f, 1.f, 0.f, false, "%.4f");
            pixelGroup.var("wo", mPixelData.wo, -1.f, 1.f, 0.f, false, "%.4f");
            pixelGroup.var("wi", mPixelData.wi, -1.f, 1.f, 0.f, false, "%.4f");
            pixelGroup.var("output", mPixelData.output, 0.f, std::numeric_limits<float>::max(), 0.f, false, "%.4f");
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

    //widget.dummy("#space3", float2(1, 16));
    //dirty |= widget.checkbox("Debug switch", mParams.debugSwitch0);

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
