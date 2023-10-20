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
#include "PixelInspectorPass.h"
#include "PixelInspectorData.slang"
#include "RenderGraph/RenderPassHelpers.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, PixelInspectorPass>();
}

namespace
{
const char kShaderFile[] = "RenderPasses/PixelInspectorPass/PixelInspector.cs.slang";

const ChannelList kInputChannels = {
    // clang-format off
    { "posW",           "gWorldPosition",               "world space position"                              },
    { "normW",          "gWorldShadingNormal",          "world space normal",           true /* optional */ },
    { "tangentW",       "gWorldTangent",                "world space tangent",          true /* optional */ },
    { "faceNormalW",    "gWorldFaceNormal",             "face normal in world space",   true /* optional */ },
    { "texC",           "gTextureCoord",                "Texture coordinate",           true /* optional */ },
    { "texGrads",       "gTextureGrads",                "Texture gradients",            true /* optional */ },
    { "mtlData",        "gMaterialData",                "Material data"                                     },
    { "linColor",       "gLinearColor",                 "color pre tone-mapping",       true /* optional */ },
    { "outColor",       "gOutputColor",                 "color post tone-mapping",      true /* optional */ },
    { "vbuffer",        "gVBuffer",                     "Visibility buffer",            true /* optional */ },
    // clang-format on
};
const char kOutputChannel[] = "gPixelDataBuffer";
} // namespace

PixelInspectorPass::PixelInspectorPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    for (auto it : kInputChannels)
    {
        mAvailableInputs[it.name] = false;
    }

    mpState = ComputeState::create(mpDevice);
}

RenderPassReflection PixelInspectorPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    addRenderPassInputs(reflector, kInputChannels);

    return reflector;
}

void PixelInspectorPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    for (auto it : kInputChannels)
    {
        mAvailableInputs[it.name] = renderData[it.name] != nullptr;
    }

    if (!mpScene)
        return;

    // Check for scene changes that require shader recompilation.
    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::RecompileNeeded) ||
        is_set(mpScene->getUpdates(), Scene::UpdateFlags::GeometryChanged))
    {
        recreatePrograms();
    }

    // Create program.
    if (!mpProgram)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderFile).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());
        desc.setCompilerFlags(SlangCompilerFlags::TreatWarningsAsErrors);

        mpProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
        mpState->setProgram(mpProgram);
    }

    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    mpProgram->addDefines(getValidResourceDefines(kInputChannels, renderData));

    if (!mpVars)
    {
        mpVars = ProgramVars::create(mpDevice, mpProgram->getReflector());
        mpPixelDataBuffer = mpDevice->createStructuredBuffer(mpVars->getRootVar()[kOutputChannel], 1);
    }

    auto var = mpVars->getRootVar();

    // Bind the scene.
    mpScene->bindShaderData(var["gScene"]);

    if (mpScene->getCamera()->getApertureRadius() > 0.f)
    {
        // TODO: Take view dir as optional input. For now issue warning if DOF is enabled.
        logWarning("Depth-of-field is enabled, but PixelInspectorPass assumes a pinhole camera. Expect the view vector to be inaccurate.");
    }

    const float2 cursorPosition = mUseContinuousPicking ? mCursorPosition : mSelectedCursorPosition;
    const uint2 resolution = renderData.getDefaultTextureDims();
    mSelectedPixel = min((uint2)(cursorPosition * ((float2)resolution)), resolution - 1u);

    // Fill in the constant buffer.
    var["PerFrameCB"]["gResolution"] = resolution;
    var["PerFrameCB"]["gSelectedPixel"] = mSelectedPixel;

    // Bind all input buffers.
    for (auto it : kInputChannels)
    {
        if (mAvailableInputs[it.name])
        {
            ref<Texture> pSrc = renderData.getTexture(it.name);
            var[it.texname] = pSrc;

            // If the texture has a different resolution, we need to scale the sampling coordinates accordingly.
            const uint2 srcResolution = uint2(pSrc->getWidth(), pSrc->getHeight());
            const bool needsScaling = mScaleInputsToWindow && any(srcResolution != resolution);
            const uint2 scaledCoord = (uint2)(((float2)(srcResolution * mSelectedPixel)) / ((float2)resolution));
            var["PerFrameCB"][std::string(it.texname) + "Coord"] = needsScaling ? scaledCoord : mSelectedPixel;

            mIsInputInBounds[it.name] = all(mSelectedPixel <= srcResolution);
        }
        else
        {
            var[it.texname].setTexture(nullptr);
        }
    }

    // Bind the output buffer.
    FALCOR_ASSERT(mpPixelDataBuffer);
    var[kOutputChannel] = mpPixelDataBuffer;

    // Run the inspector program.
    pRenderContext->dispatch(mpState.get(), mpVars.get(), {1u, 1u, 1u});
}

void PixelInspectorPass::renderUI(Gui::Widgets& widget)
{
    if (!mpScene)
    {
        widget.textWrapped("No scene loaded, no data available!");
        return;
    }

    FALCOR_ASSERT(mpPixelDataBuffer);
    PixelData pixelData = mpPixelDataBuffer->getElement<PixelData>(0);

    // Display the coordinates for the pixel at which information is retrieved.
    widget.var("Looking at pixel", (int2&)mSelectedPixel, 0);

    widget.checkbox("Scale inputs to window size", mScaleInputsToWindow);
    widget.checkbox("Continuously inspect pixels", mUseContinuousPicking);
    widget.tooltip(
        "If continuously inspecting pixels, you will always see the data for the pixel currently under your mouse.\n"
        "Otherwise, left-mouse click on a pixel to select it.",
        true
    );

    const auto displayValues = [&pixelData, &widget, this](
                                   const std::vector<std::string>& inputNames,
                                   const std::vector<std::string>& values,
                                   const std::function<void(PixelData&)>& displayValues
                               )
    {
        bool areAllInputsAvailable = true;
        for (const std::string& inputName : inputNames)
            areAllInputsAvailable = areAllInputsAvailable && mAvailableInputs[inputName];

        if (areAllInputsAvailable)
        {
            bool areAllInputsInBounds = true;
            for (const std::string& inputName : inputNames)
                areAllInputsInBounds = areAllInputsInBounds && mIsInputInBounds[inputName];

            if (areAllInputsInBounds)
            {
                displayValues(pixelData);
            }
            else
            {
                for (const std::string& value : values)
                {
                    const std::string text = value + ": out of bounds";
                    widget.text(text);
                }
            }
            return true;
        }
        return false;
    };

    // Display output data.
    if (auto outputGroup = widget.group("Output data", true))
    {
        bool displayedData = displayValues(
            {"linColor"},
            {"Linear color", "Luminance (cd/m2)"},
            [&outputGroup](PixelData& pixelData)
            {
                outputGroup.var("Linear color", pixelData.linearColor, 0.f, std::numeric_limits<float>::max(), 0.001f, false, "%.6f");
                outputGroup.var("Luminance (cd/m2)", pixelData.luminance, 0.f, std::numeric_limits<float>::max(), 0.001f, false, "%.6f");
            }
        );

        displayedData |= displayValues(
            {"outColor"},
            {"Output color"},
            [&outputGroup](PixelData& pixelData)
            { outputGroup.var("Output color", pixelData.outputColor, 0.f, 1.f, 0.001f, false, "%.6f"); }
        );

        if (displayedData == false)
            outputGroup.text("No input data");
    }

    // Display geometry data.
    if (auto geometryGroup = widget.group("Geometry data", true))
    {
        displayValues(
            {"posW"},
            {"World position"},
            [&geometryGroup](PixelData& pixelData)
            {
                geometryGroup.var(
                    "World position",
                    pixelData.posW,
                    std::numeric_limits<float>::lowest(),
                    std::numeric_limits<float>::max(),
                    0.001f,
                    false,
                    "%.6f"
                );
            }
        );

        displayValues(
            {"normW"},
            {"Shading normal"},
            [&geometryGroup](PixelData& pixelData)
            { geometryGroup.var("Shading normal", pixelData.normal, -1.f, 1.f, 0.001f, false, "%.6f"); }
        );

        displayValues(
            {"tangentW"},
            {"Shading tangent"},
            [&geometryGroup](PixelData& pixelData)
            { geometryGroup.var("Shading tangent", pixelData.tangent, -1.f, 1.f, 0.001f, false, "%.6f"); }
        );

        displayValues(
            {"normW", "tangentW"},
            {"Shading bitangent"},
            [&geometryGroup](PixelData& pixelData)
            { geometryGroup.var("Shading bitangent", pixelData.bitangent, -1.f, 1.f, 0.001f, false, "%.6f"); }
        );

        displayValues(
            {"faceNormalW"},
            {"Face normal"},
            [&geometryGroup](PixelData& pixelData)
            { geometryGroup.var("Face normal", pixelData.faceNormal, -1.f, 1.f, 0.001f, false, "%.6f"); }
        );

        geometryGroup.var("View vector", pixelData.view, -1.f, 1.f, 0.001f, false, "%.6f");

        displayValues(
            {"texC"},
            {"Texture coords"},
            [&geometryGroup](PixelData& pixelData)
            {
                geometryGroup.var(
                    "Texture coord",
                    pixelData.texCoord,
                    std::numeric_limits<float>::lowest(),
                    std::numeric_limits<float>::max(),
                    0.001f,
                    false,
                    "%.6f"
                );
            }
        );

        geometryGroup.checkbox("Front facing", pixelData.frontFacing);
    }

    // Display material data.
    if (auto materialGroup = widget.group("Material data", true))
    {
        const std::vector<std::string> requiredInputs = {"posW", "texC", "mtlData"};

        bool displayedData = false;

        displayedData |= displayValues(
            requiredInputs,
            {"Material ID"},
            [&materialGroup](PixelData& pixelData) { materialGroup.var("Material ID", pixelData.materialID); }
        );

        displayedData |= displayValues(
            requiredInputs,
            {"Double sided"},
            [&materialGroup](PixelData& pixelData) { materialGroup.checkbox("Double sided", pixelData.doubleSided); }
        );

        displayedData |= displayValues(
            requiredInputs,
            {"Opacity"},
            [&materialGroup](PixelData& pixelData) { materialGroup.var("Opacity", pixelData.opacity, 0.f, 1.f, 0.001f, false, "%.6f"); }
        );

        displayedData |= displayValues(
            requiredInputs,
            {"IoR (outside)"},
            [&materialGroup](PixelData& pixelData)
            { materialGroup.var("IoR (outside)", pixelData.IoR, 0.f, std::numeric_limits<float>::max(), 0.001f, false, "%.6f"); }
        );

        displayedData |= displayValues(
            requiredInputs,
            {"Emission"},
            [&materialGroup](PixelData& pixelData)
            { materialGroup.var("Emission", pixelData.emission, 0.f, std::numeric_limits<float>::max(), 0.001f, false, "%.6f"); }
        );

        displayedData |= displayValues(
            requiredInputs,
            {"Roughness"},
            [&materialGroup](PixelData& pixelData)
            { materialGroup.var("Roughness", pixelData.roughness, 0.f, std::numeric_limits<float>::max(), 0.001f, false, "%.6f"); }
        );

        displayedData |= displayValues(
            requiredInputs,
            {"GuideNormal"},
            [&materialGroup](PixelData& pixelData)
            { materialGroup.var("GuideNormal", pixelData.guideNormal, 0.f, std::numeric_limits<float>::max(), 0.001f, false, "%.6f"); }
        );

        displayedData |= displayValues(
            requiredInputs,
            {"DiffuseReflectionAlbedo"},
            [&materialGroup](PixelData& pixelData)
            {
                materialGroup.var(
                    "DiffuseReflectionAlbedo",
                    pixelData.diffuseReflectionAlbedo,
                    0.f,
                    std::numeric_limits<float>::max(),
                    0.001f,
                    false,
                    "%.6f"
                );
            }
        );

        displayedData |= displayValues(
            requiredInputs,
            {"DiffuseTransmissionAlbedo"},
            [&materialGroup](PixelData& pixelData)
            {
                materialGroup.var(
                    "DiffuseTransmissionAlbedo",
                    pixelData.diffuseTransmissionAlbedo,
                    0.f,
                    std::numeric_limits<float>::max(),
                    0.001f,
                    false,
                    "%.6f"
                );
            }
        );

        displayedData |= displayValues(
            requiredInputs,
            {"SpecularReflectionAlbedo"},
            [&materialGroup](PixelData& pixelData)
            {
                materialGroup.var(
                    "SpecularReflectionAlbedo",
                    pixelData.specularReflectionAlbedo,
                    0.f,
                    std::numeric_limits<float>::max(),
                    0.001f,
                    false,
                    "%.6f"
                );
            }
        );

        displayedData |= displayValues(
            requiredInputs,
            {"SpecularTransmissionAlbedo"},
            [&materialGroup](PixelData& pixelData)
            {
                materialGroup.var(
                    "SpecularTransmissionAlbedo",
                    pixelData.specularTransmissionAlbedo,
                    0.f,
                    std::numeric_limits<float>::max(),
                    0.001f,
                    false,
                    "%.6f"
                );
            }
        );

        displayedData |= displayValues(
            requiredInputs,
            {"SpecularReflectance"},
            [&materialGroup](PixelData& pixelData) {
                materialGroup.var(
                    "SpecularReflectance", pixelData.specularReflectance, 0.f, std::numeric_limits<float>::max(), 0.001f, false, "%.6f"
                );
            }
        );

        displayedData |= displayValues(
            requiredInputs,
            {"IsTransmissive"},
            [&materialGroup](PixelData& pixelData) { materialGroup.checkbox("IsTransmissive", pixelData.isTransmissive); }
        );

        if (displayedData == false)
            materialGroup.text("No input data");
    }

    // Display visibility data.
    if (auto visGroup = widget.group("Visibility data", true))
    {
        if (mAvailableInputs["vbuffer"])
        {
            bool validHit = mpScene && pixelData.instanceID != PixelData::kInvalidIndex;

            std::string hitType = "None";
            if (validHit)
            {
                switch ((HitType)pixelData.hitType)
                {
                case HitType::Triangle:
                    hitType = "Triangle";
                    break;
                case HitType::Curve:
                    hitType = "Curve";
                    break;
                default:
                    hitType = "Unknown";
                    FALCOR_ASSERT(false);
                }
            }

            visGroup.text("HitType: " + hitType);
            visGroup.var("InstanceID", pixelData.instanceID);
            visGroup.var("PrimitiveIndex", pixelData.primitiveIndex);
            visGroup.var("Barycentrics", pixelData.barycentrics);

            if (validHit && (HitType)pixelData.hitType == HitType::Triangle)
            {
                auto instanceData = mpScene->getGeometryInstance(pixelData.instanceID);
                uint32_t matrixID = instanceData.globalMatrixID;
                float4x4 M = mpScene->getAnimationController()->getGlobalMatrices()[matrixID];

                visGroup.text("Transform:");
                visGroup.matrix("##mat", M);

                bool flipped = instanceData.flags & (uint32_t)GeometryInstanceFlags::TransformFlipped;
                bool objectCW = instanceData.flags & (uint32_t)GeometryInstanceFlags::IsObjectFrontFaceCW;
                bool worldCW = instanceData.flags & (uint32_t)GeometryInstanceFlags::IsWorldFrontFaceCW;
                visGroup.checkbox("TransformFlipped", flipped);
                visGroup.checkbox("IsObjectFrontFaceCW", objectCW);
                visGroup.checkbox("IsWorldFrontFaceCW", worldCW);
            }
        }
        else
        {
            visGroup.text("No visibility data available");
        }
    }
}

void PixelInspectorPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;

    recreatePrograms();
}

void PixelInspectorPass::recreatePrograms()
{
    mpProgram = nullptr;
    mpVars = nullptr;
    mpPixelDataBuffer = nullptr;
}

bool PixelInspectorPass::onMouseEvent(const MouseEvent& mouseEvent)
{
    if (mouseEvent.type == MouseEvent::Type::Move)
    {
        mCursorPosition = mouseEvent.pos;
    }
    else if (mouseEvent.type == MouseEvent::Type::ButtonDown && mouseEvent.button == Input::MouseButton::Left)
    {
        mSelectedCursorPosition = mouseEvent.pos;
    }

    return false;
}
