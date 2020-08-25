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
#include "PixelInspectorPass.h"
#include "PixelInspectorData.slang"
#include "RenderGraph/RenderPassHelpers.h"

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerClass("PixelInspectorPass", "Per pixel surface attributes inspector", PixelInspectorPass::create);
}

namespace
{
    const char kShaderFile[] = "RenderPasses/PixelInspectorPass/PixelInspector.cs.slang";

    const ChannelList kInputChannels =
    {
        { "posW",           "gWorldPosition",               "world space position",         true /* optional */ },
        { "normW",          "gWorldShadingNormal",          "world space normal",           true /* optional */ },
        { "tangentW",       "gWorldTangent",                "world space tangent",          true /* optional */ },
        { "faceNormalW",    "gWorldFaceNormal",             "face normal in world space",   true /* optional */ },
        { "texC",           "gTextureCoordinate",           "texture coordinates",          true /* optional */ },
        { "diffuseOpacity", "gMaterialDiffuseOpacity",      "diffuse color and opacity",    true /* optional */ },
        { "specRough",      "gMaterialSpecularRoughness",   "specular color and roughness", true /* optional */ },
        { "emissive",       "gMaterialEmissive",            "emissive color",               true /* optional */ },
        { "matlExtra",      "gMaterialExtraParams",         "additional material data",     true /* optional */ },
        { "linColor",       "gLinearColor",                 "color pre tone-mapping",       true /* optional */ },
        { "outColor",       "gOutputColor",                 "color post tone-mapping",      true /* optional */ },
        { "visBuffer",      "gVisBuffer",                   "Visibility buffer",            true /* optional */, ResourceFormat::RGBA32Uint },
    };
    const char kOutputChannel[] = "gPixelDataBuffer";
}

PixelInspectorPass::SharedPtr PixelInspectorPass::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new PixelInspectorPass);
}

PixelInspectorPass::PixelInspectorPass()
{
    for (auto it : kInputChannels)
    {
        mAvailableInputs[it.name] = false;
    }

    mpProgram = ComputeProgram::createFromFile(kShaderFile, "main", Program::DefineList(), Shader::CompilerFlags::TreatWarningsAsErrors);
    assert(mpProgram);

    mpVars = ComputeVars::create(mpProgram->getReflector());
    mpState = ComputeState::create();
    mpState->setProgram(mpProgram);
    assert(mpVars && mpState);

    mpPixelDataBuffer = Buffer::createStructured(mpProgram.get(), kOutputChannel, 1);
}

std::string PixelInspectorPass::getDesc()
{
    return
        "Inspect geometric and material properties at a given pixel.\n"
        "\n"
        "Left-mouse click on a pixel to select it\n";
}

RenderPassReflection PixelInspectorPass::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    for (auto it : kInputChannels)
    {
        auto& f = reflector.addInput(it.name, it.desc).format(it.format);
        if (it.optional) f.flags(RenderPassReflection::Field::Flags::Optional);
    }
    return reflector;
}

void PixelInspectorPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    for (auto it : kInputChannels)
    {
        mAvailableInputs[it.name] = renderData[it.name] != nullptr;
    }

    if (!mpScene) return;

    // Set the camera
    const Camera::SharedPtr& pCamera = mpScene->getCamera();
    pCamera->setShaderData(mpVars["PerFrameCB"]["gCamera"]);

    if (pCamera->getApertureRadius() > 0.f)
    {
        // TODO: Take view dir as optional input. For now issue warning if DOF is enabled.
        logWarning("Depth-of-field is enabled, but PixelInspectorPass assumes a pinhole camera. Expect the view vector to be inaccurate.");
    }

    const float2 cursorPosition = mUseContinuousPicking ? mCursorPosition : mSelectedCursorPosition;
    const uint2 resolution = renderData.getDefaultTextureDims();
    mSelectedPixel = glm::min((uint2)(cursorPosition * ((float2)resolution)), resolution - 1u);

    // Fill in the constant buffer.
    mpVars["PerFrameCB"]["gResolution"] = resolution;
    mpVars["PerFrameCB"]["gSelectedPixel"] = mSelectedPixel;

    // Bind all input buffers.
    for (auto it : kInputChannels)
    {
        if (mAvailableInputs[it.name])
        {
            Texture::SharedPtr pSrc = renderData[it.name]->asTexture();
            mpVars[it.texname] = pSrc;

            // If the texture has a different resolution, we need to scale the sampling coordinates accordingly.
            const uint2 srcResolution = uint2(pSrc->getWidth(), pSrc->getHeight());
            const bool needsScaling = mScaleInputsToWindow && srcResolution != resolution;
            const uint2 scaledCoord = (uint2)(((float2)(srcResolution * mSelectedPixel)) / ((float2)resolution));
            mpVars["PerFrameCB"][std::string(it.texname) + "Coord"] = needsScaling ? scaledCoord : mSelectedPixel;

            mIsInputInBounds[it.name] = glm::all(glm::lessThanEqual(mSelectedPixel, srcResolution));
        }
        else
        {
            mpVars->setTexture(it.texname, nullptr);
        }
    }

    // Bind the output buffer.
    mpVars[kOutputChannel] = mpPixelDataBuffer;

    // Run the inspector program.
    pRenderContext->dispatch(mpState.get(), mpVars.get(), { 1u, 1u, 1u });
}

void PixelInspectorPass::renderUI(Gui::Widgets& widget)
{
    PixelData pixelData = *reinterpret_cast<const PixelData*>(mpPixelDataBuffer->map(Buffer::MapType::Read));
    mpPixelDataBuffer->unmap();

    // Display the coordinates for the pixel at which information is retrieved.
    widget.var("Looking at pixel", (int2&)mSelectedPixel, 0);

    widget.checkbox("Scale inputs to window size", mScaleInputsToWindow);
    widget.checkbox("Continuously inspect pixels", mUseContinuousPicking);
    widget.tooltip("If continuously inspecting pixels, you will always see the data for the pixel currently under your mouse.\n"
        "Otherwise, left-mouse click on a pixel to select it.", true);

    const auto displayValues = [&pixelData, &widget, this](const std::vector<std::string>& inputNames, const std::vector<std::string>& values, const std::function<void(PixelData&)>& displayValues)
    {
        bool areAllInputsAvailable = true;
        for (const std::string& inputName : inputNames) areAllInputsAvailable = areAllInputsAvailable && mAvailableInputs[inputName];

        if (areAllInputsAvailable)
        {
            bool areAllInputsInBounds = true;
            for (const std::string& inputName : inputNames) areAllInputsInBounds = areAllInputsInBounds && mIsInputInBounds[inputName];

            if (areAllInputsInBounds)
            {
                displayValues(pixelData);
            }
            else
            {
                for (const std::string& value : values)
                {
                    const std::string text = value + ": out of bounds";
                    widget.text(text.c_str());
                }
            }
            return true;
        }
        return false;
    };

    // Display output data.
    if (auto outputGroup = widget.group("Output data", true))
    {
        bool displayedData = displayValues({ "linColor" }, { "Linear color", "Luminance (cd/m2)" }, [&outputGroup](PixelData& pixelData) {
            outputGroup.var("Linear color", pixelData.linearColor, 0.f, std::numeric_limits<float>::max(), 0.001f, false, "%.6f");
            outputGroup.var("Luminance (cd/m2)", pixelData.luminance, 0.f, std::numeric_limits<float>::max(), 0.001f, false, "%.6f");
        });

        displayedData |= displayValues({ "outColor" }, { "Output color" }, [&outputGroup](PixelData& pixelData) {
            outputGroup.var("Output color", pixelData.outputColor, 0.f, 1.f, 0.001f, false, "%.6f");
        });

        if (displayedData == false) outputGroup.text("No input data");
    }

    if (auto geometryGroup = widget.group("Geometry data", true))
    {
        // Display geometry data
        displayValues({ "posW" }, { "World position" }, [&geometryGroup](PixelData& pixelData) {
            geometryGroup.var("World position", pixelData.posW, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(), 0.001f, false, "%.6f");
        });

        displayValues({ "normW" }, { "Shading normal" }, [&geometryGroup](PixelData& pixelData) {
            geometryGroup.var("Shading normal", pixelData.normal, -1.f, 1.f, 0.001f, false, "%.6f");
        });

        displayValues({ "tangentW" }, { "Shading tangent" }, [&geometryGroup](PixelData& pixelData) {
            geometryGroup.var("Shading tangent", pixelData.tangent, -1.f, 1.f, 0.001f, false, "%.6f");
        });

        displayValues({ "normW", "tangentW" }, { "Shading bitangent" }, [&geometryGroup](PixelData& pixelData) {
            geometryGroup.var("Shading bitangent", pixelData.bitangent, -1.f, 1.f, 0.001f, false, "%.6f");
        });

        displayValues({ "faceNormalW" }, { "Face normal" }, [&geometryGroup](PixelData& pixelData) {
            geometryGroup.var("Face normal", pixelData.faceNormal, -1.f, 1.f, 0.001f, false, "%.6f");
        });

        displayValues({ "texC" }, { "Texture coords" }, [&geometryGroup](PixelData& pixelData) {
            float2 texCoords = float2(pixelData.texCoordU, pixelData.texCoordV);
            geometryGroup.var("Texture coords", texCoords, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(), 0.001f, false, "%.6f");
        });

        geometryGroup.var("View vector", pixelData.view, -1.f, 1.f, 0.001f, false, "%.6f");

        displayValues({ "normW" }, { "NdotV" }, [&geometryGroup](PixelData& pixelData) {
            geometryGroup.var("NdotV", pixelData.NdotV, -1.f, 1.f, 0.001f, false, "%.6f");
        });
    }

    if (auto materialGroup = widget.group("Material data", true))
    {
        // Display material data
        bool displayedData = displayValues({ "diffuseOpacity" }, { "Diffuse color", "Opacity" }, [&materialGroup](PixelData& pixelData) {
            materialGroup.var("Diffuse color", pixelData.diffuse, 0.f, 1.f, 0.001f, false, "%.6f");
            materialGroup.var("Opacity", pixelData.opacity, 0.f, 1.f, 0.001f, false, "%.6f");
        });

        displayedData |= displayValues({ "specRough" }, { "Specular color", "GGX Alpha", "Roughness" }, [&materialGroup](PixelData& pixelData) {
            materialGroup.var("Specular color", pixelData.specular, 0.f, 1.f, 0.001f, false, "%.6f");
            materialGroup.var("GGX Alpha", pixelData.ggxAlpha, 0.f, 1.f, 0.001f, false, "%.6f");
            materialGroup.var("Roughness", pixelData.linearRoughness, 0.f, 1.f, 0.001f, false, "%.6f");
        });

        displayedData |= displayValues({ "emissive" }, { "Emissive" }, [&materialGroup](PixelData& pixelData) {
            materialGroup.var("Emissive", pixelData.emissive, 0.f, std::numeric_limits<float>::max(), 0.001f, false, "%.6f");
        });

        displayedData |= displayValues({ "matlExtra" }, { "IoR", "Double sided" }, [&materialGroup](PixelData& pixelData) {
            materialGroup.var("IoR", pixelData.IoR, 0.f, std::numeric_limits<float>::max(), 0.001f, false, "%.6f");
            materialGroup.checkbox("Double sided", (int&)pixelData.doubleSided);
        });

        if (displayedData == false) materialGroup.text("No input data");
    }

    // Display visibility data
    if (auto visGroup = widget.group("Visibility data", true))
    {
        if (mAvailableInputs["visBuffer"])
        {
            visGroup.var("MeshInstanceID", pixelData.meshInstanceID);
            visGroup.var("TriangleIndex", pixelData.triangleIndex);
            visGroup.var("Barycentrics", pixelData.barycentrics);

            if (mpScene && pixelData.meshInstanceID != PixelData::kInvalidIndex)
            {
                auto instanceData = mpScene->getMeshInstance(pixelData.meshInstanceID);
                uint32_t matrixID = instanceData.globalMatrixID;
                glm::mat4 M = mpScene->getAnimationController()->getGlobalMatrices()[matrixID];

                visGroup.text("Transform:");
                visGroup.var("##col0", M[0]);
                visGroup.var("##col1", M[1]);
                visGroup.var("##col2", M[2]);
                visGroup.var("##col3", M[3]);

                bool flipped = instanceData.flags & (uint32_t)MeshInstanceFlags::Flipped;
                visGroup.checkbox("Flipped winding", flipped);
            }
        }
        else
        {
            visGroup.text("No visibility data available");
        }
    }
}

void PixelInspectorPass::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    mpScene = pScene;
}

bool PixelInspectorPass::onMouseEvent(const MouseEvent& mouseEvent)
{
    if (mouseEvent.type == MouseEvent::Type::Move)
    {
        mCursorPosition = mouseEvent.pos;
    }
    else if (mouseEvent.type == MouseEvent::Type::LeftButtonDown)
    {
        mSelectedCursorPosition = mouseEvent.pos;
    }

    return false;
}
