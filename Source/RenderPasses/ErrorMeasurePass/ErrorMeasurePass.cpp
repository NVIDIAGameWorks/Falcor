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
#include "ErrorMeasurePass.h"
#include "RenderGraph/RenderPassLibrary.h"
#include <sstream>

const RenderPass::Info ErrorMeasurePass::kInfo { "ErrorMeasurePass", "Measures error with respect to a reference image." };

namespace
{
    const std::string kErrorComputationShaderFile = "RenderPasses/ErrorMeasurePass/ErrorMeasurer.cs.slang";
    const std::string kConstantBufferName = "PerFrameCB";

    // Input channels
    const std::string kInputChannelWorldPosition = "WorldPosition";
    const std::string kInputChannelSourceImage = "Source";
    const std::string kInputChannelReferenceImage = "Reference";

    // Output channel
    const std::string kOutputChannelImage = "Output";

    // Serialized parameters
    const std::string kReferenceImagePath = "ReferenceImagePath";
    const std::string kMeasurementsFilePath = "MeasurementsFilePath";
    const std::string kIgnoreBackground = "IgnoreBackground";
    const std::string kComputeSquaredDifference = "ComputeSquaredDifference";
    const std::string kComputeAverage = "ComputeAverage";
    const std::string kUseLoadedReference = "UseLoadedReference";
    const std::string kReportRunningError = "ReportRunningError";
    const std::string kRunningErrorSigma = "RunningErrorSigma";
    const std::string kSelectedOutputId = "SelectedOutputId";
}

static void regErrorMeasurePass(pybind11::module& m)
{
    pybind11::enum_<ErrorMeasurePass::OutputId> op(m, "OutputId");
    op.value("Source", ErrorMeasurePass::OutputId::Source);
    op.value("Reference", ErrorMeasurePass::OutputId::Reference);
    op.value("Difference", ErrorMeasurePass::OutputId::Difference);
}

// Don't remove this. it's required for hot-reload to function properly
extern "C" FALCOR_API_EXPORT const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" FALCOR_API_EXPORT void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerPass(ErrorMeasurePass::kInfo, ErrorMeasurePass::create);
    ScriptBindings::registerBinding(regErrorMeasurePass);
}

const Gui::RadioButtonGroup ErrorMeasurePass::sOutputSelectionButtons =
{
    { (uint32_t)OutputId::Source, "Source", true },
    { (uint32_t)OutputId::Reference, "Reference", true },
    { (uint32_t)OutputId::Difference, "Difference", true }
};

const Gui::RadioButtonGroup ErrorMeasurePass::sOutputSelectionButtonsSourceOnly =
{
    { (uint32_t)OutputId::Source, "Source", true }
};

ErrorMeasurePass::SharedPtr ErrorMeasurePass::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new ErrorMeasurePass(dict));
}

ErrorMeasurePass::ErrorMeasurePass(const Dictionary& dict)
    : RenderPass(kInfo)
{
    for (const auto& [key, value] : dict)
    {
        if (key == kReferenceImagePath) mReferenceImagePath = value.operator std::filesystem::path();
        else if (key == kMeasurementsFilePath) mMeasurementsFilePath = value.operator std::filesystem::path();
        else if (key == kIgnoreBackground) mIgnoreBackground = value;
        else if (key == kComputeSquaredDifference) mComputeSquaredDifference = value;
        else if (key == kComputeAverage) mComputeAverage = value;
        else if (key == kUseLoadedReference) mUseLoadedReference = value;
        else if (key == kReportRunningError) mReportRunningError = value;
        else if (key == kRunningErrorSigma) mRunningErrorSigma = value;
        else if (key == kSelectedOutputId) mSelectedOutputId = value;
        else
        {
            logWarning("Unknown field '{}' in ErrorMeasurePass dictionary.", key);
        }
    }

    // Load/create files (if specified in config).
    loadReference();
    openMeasurementsFile();

    mpParallelReduction = ComputeParallelReduction::create();
    mpErrorMeasurerPass = ComputePass::create(kErrorComputationShaderFile);
}

Dictionary ErrorMeasurePass::getScriptingDictionary()
{
    Dictionary dict;
    dict[kReferenceImagePath] = mReferenceImagePath;
    dict[kMeasurementsFilePath] = mMeasurementsFilePath;
    dict[kIgnoreBackground] = mIgnoreBackground;
    dict[kComputeSquaredDifference] = mComputeSquaredDifference;
    dict[kComputeAverage] = mComputeAverage;
    dict[kUseLoadedReference] = mUseLoadedReference;
    dict[kReportRunningError] = mReportRunningError;
    dict[kRunningErrorSigma] = mRunningErrorSigma;
    dict[kSelectedOutputId] = mSelectedOutputId;
    return dict;
}

RenderPassReflection ErrorMeasurePass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kInputChannelSourceImage, "Source image");
    reflector.addInput(kInputChannelReferenceImage, "Reference image (optional)").flags(RenderPassReflection::Field::Flags::Optional);
    reflector.addInput(kInputChannelWorldPosition, "World-space position").flags(RenderPassReflection::Field::Flags::Optional);
    // TODO: when compile() is available, match the format of the source image?
    reflector.addOutput(kOutputChannelImage, "Output image").format(ResourceFormat::RGBA32Float);
    return reflector;
}

void ErrorMeasurePass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    Texture::SharedPtr pSourceImageTexture = renderData.getTexture(kInputChannelSourceImage);
    Texture::SharedPtr pOutputImageTexture = renderData.getTexture(kOutputChannelImage);

    // Create the texture for the difference image if this is our first
    // time through or if the source image resolution has changed.
    const uint32_t width = pSourceImageTexture->getWidth(), height = pSourceImageTexture->getHeight();
    if (!mpDifferenceTexture || mpDifferenceTexture->getWidth() != width ||
        mpDifferenceTexture->getHeight() != height)
    {
        mpDifferenceTexture = Texture::create2D(width, height, ResourceFormat::RGBA32Float, 1, 1, nullptr,
                                                Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
        FALCOR_ASSERT(mpDifferenceTexture);
    }

    mMeasurements.valid = false;

    Texture::SharedPtr pReference = getReference(renderData);
    if (!pReference)
    {
        // We don't have a reference image, so just copy the source image to the output.
        pRenderContext->blit(pSourceImageTexture->getSRV(), pOutputImageTexture->getRTV());
        return;
    }

    runDifferencePass(pRenderContext, renderData);
    runReductionPasses(pRenderContext, renderData);

    switch (mSelectedOutputId)
    {
    case OutputId::Source:
        pRenderContext->blit(pSourceImageTexture->getSRV(), pOutputImageTexture->getRTV());
        break;
    case OutputId::Reference:
        pRenderContext->blit(pReference->getSRV(), pOutputImageTexture->getRTV());
        break;
    case OutputId::Difference:
        pRenderContext->blit(mpDifferenceTexture->getSRV(), pOutputImageTexture->getRTV());
        break;
    default:
        throw RuntimeError("ErrorMeasurePass: Unhandled OutputId case");
    }

    saveMeasurementsToFile();
}

void ErrorMeasurePass::runDifferencePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Bind textures.
    Texture::SharedPtr pSourceTexture = renderData.getTexture(kInputChannelSourceImage);
    Texture::SharedPtr pWorldPositionTexture = renderData.getTexture(kInputChannelWorldPosition);
    mpErrorMeasurerPass["gReference"] = getReference(renderData);
    mpErrorMeasurerPass["gSource"] = pSourceTexture;
    mpErrorMeasurerPass["gWorldPosition"] = pWorldPositionTexture;
    mpErrorMeasurerPass["gResult"] = mpDifferenceTexture;

    // Set constant buffer parameters.
    const uint2 resolution = uint2(pSourceTexture->getWidth(), pSourceTexture->getHeight());
    mpErrorMeasurerPass[kConstantBufferName]["gResolution"] = resolution;
    // If the world position texture is unbound, then don't do the background pixel check.
    mpErrorMeasurerPass[kConstantBufferName]["gIgnoreBackground"] = (uint32_t)(mIgnoreBackground && pWorldPositionTexture);
    mpErrorMeasurerPass[kConstantBufferName]["gComputeDiffSqr"] = (uint32_t)mComputeSquaredDifference;
    mpErrorMeasurerPass[kConstantBufferName]["gComputeAverage"] = (uint32_t)mComputeAverage;

    // Run the compute shader.
    mpErrorMeasurerPass->execute(pRenderContext, resolution.x, resolution.y);
}

void ErrorMeasurePass::runReductionPasses(RenderContext* pRenderContext, const RenderData& renderData)
{
    float4 error;
    mpParallelReduction->execute(pRenderContext, mpDifferenceTexture, ComputeParallelReduction::Type::Sum, &error);

    const float pixelCountf = static_cast<float>(mpDifferenceTexture->getWidth() * mpDifferenceTexture->getHeight());
    mMeasurements.error = error / pixelCountf;
    mMeasurements.avgError = (mMeasurements.error.x + mMeasurements.error.y + mMeasurements.error.z) / 3.f;
    mMeasurements.valid = true;

    if (mRunningAvgError < 0)
    {
        // The running error values are invalid. Start them off with the current frame's error.
        mRunningError = mMeasurements.error;
        mRunningAvgError = mMeasurements.avgError;
    }
    else
    {
        mRunningError = mRunningErrorSigma * mRunningError + (1 - mRunningErrorSigma) * mMeasurements.error;
        mRunningAvgError = mRunningErrorSigma * mRunningAvgError + (1 - mRunningErrorSigma) * mMeasurements.avgError;
    }
}

void ErrorMeasurePass::renderUI(Gui::Widgets& widget)
{
    const auto getFilename = [](const std::filesystem::path& path)
    {
        return path.empty() ? "N/A" : path.filename().string();
    };

    // Create a button for loading the reference image.
    if (widget.button("Load reference"))
    {
        FileDialogFilterVec filters;
        filters.push_back({ "exr", "High Dynamic Range" });
        filters.push_back({ "pfm", "Portable Float Map" });
        std::filesystem::path path;
        if (openFileDialog(filters, path))
        {
            mReferenceImagePath = path;
            loadReference();
        }
    }

    // Create a button for defining the measurements output file.
    if (widget.button("Set output data file", true))
    {
        FileDialogFilterVec filters;
        filters.push_back({ "csv", "CSV Files" });
        std::filesystem::path path;
        if (saveFileDialog(filters, path))
        {
            mMeasurementsFilePath = path;
            openMeasurementsFile();
        }
    }

    // Radio buttons to select the output.
    widget.text("Show:");
    if (mMeasurements.valid)
    {
        widget.radioButtons(sOutputSelectionButtons, reinterpret_cast<uint32_t&>(mSelectedOutputId));
        widget.tooltip("Press 'O' to change output mode; hold 'Shift' to reverse the cycling.\n\n"
                         "Note: Difference is computed based on current - reference value.", true);
    }
    else
    {
        uint32_t dummyId = 0;
        widget.radioButtons(sOutputSelectionButtonsSourceOnly, dummyId);
    }

    widget.checkbox("Ignore background", mIgnoreBackground);
    widget.tooltip("Do not include background pixels in the error measurements.\n"
                      "This option requires the optional input '" + std::string(kInputChannelWorldPosition) + "' to be bound", true);
    widget.checkbox("Compute L2 error (rather than L1)", mComputeSquaredDifference);
    widget.checkbox("Compute RGB average", mComputeAverage);
    widget.tooltip("When enabled, the average error over the RGB components is computed when creating the difference image.\n"
        "The average is computed after squaring the differences when L2 error is selected.");

    widget.checkbox("Use loaded reference image", mUseLoadedReference);
    widget.tooltip("Take the reference from the loaded image instead or the input channel.\n\n"
                     "If the chosen reference doesn't exist, the error measurements are disabled.", true);
    // Display the filename of the reference file.
    const std::string referenceText = "Reference: " + getFilename(mReferenceImagePath);
    widget.text(referenceText);
    if (!mReferenceImagePath.empty())
    {
        widget.tooltip(mReferenceImagePath.string());
    }

    // Display the filename of the measurement file.
    const std::string outputText = "Output: " + getFilename(mMeasurementsFilePath);
    widget.text(outputText);
    if (!mMeasurementsFilePath.empty())
    {
        widget.tooltip(mMeasurementsFilePath.string());
    }

    // Print numerical error (scalar and RGB).
    if (widget.checkbox("Report running error", mReportRunningError) && mReportRunningError)
    {
        // The checkbox was enabled; mark the running error values invalid so that they start fresh.
        mRunningAvgError = -1.f;
    }
    widget.tooltip("Exponential moving average, sigma = " + std::to_string(mRunningErrorSigma));
    if (mMeasurements.valid)
    {
        // Use stream so we can control formatting.
        std::ostringstream oss;
        oss << std::scientific;
        oss << (mComputeSquaredDifference ? "MSE (avg): " : "L1 error (avg): ") <<
          (mReportRunningError ? mRunningAvgError : mMeasurements.avgError) << std::endl;
        oss << (mComputeSquaredDifference ? "MSE (rgb): " : "L1 error (rgb): ") <<
          (mReportRunningError ? mRunningError.r : mMeasurements.error.r) << ", " <<
          (mReportRunningError ? mRunningError.g : mMeasurements.error.g) << ", " <<
          (mReportRunningError ? mRunningError.b : mMeasurements.error.b);
        widget.text(oss.str());
    }
    else
    {
        widget.text("Error: N/A");
    }
}

bool ErrorMeasurePass::onKeyEvent(const KeyboardEvent& keyEvent)
{
    if (keyEvent.type == KeyboardEvent::Type::KeyPressed && keyEvent.key == Input::Key::O)
    {
        int32_t ofs = keyEvent.hasModifier(Input::Modifier::Shift) ? -1 : 1;
        int32_t index = (int32_t)mSelectedOutputId;
        index = (index + ofs + (int32_t)OutputId::Count) % (int32_t)OutputId::Count;
        mSelectedOutputId = (OutputId)index;
        return true;
    }

    return false;
}

void ErrorMeasurePass::loadReference()
{
    if (mReferenceImagePath.empty()) return;

    // TODO: it would be nice to also be able to take the reference image as an input.
    mpReferenceTexture = Texture::createFromFile(mReferenceImagePath, false /* no MIPs */, false /* linear color */);
    if (!mpReferenceTexture)
    {
        reportError(fmt::format("Failed to load texture from '{}'", mReferenceImagePath));
        mReferenceImagePath.clear();
    }

    mUseLoadedReference = mpReferenceTexture != nullptr;
    mRunningAvgError = -1.f;   // Mark running error values as invalid.
}

Texture::SharedPtr ErrorMeasurePass::getReference(const RenderData& renderData) const
{
    return mUseLoadedReference ? mpReferenceTexture : renderData.getTexture(kInputChannelReferenceImage);
}

void ErrorMeasurePass::openMeasurementsFile()
{
    if (mMeasurementsFilePath.empty()) return;

    mMeasurementsFile = std::ofstream(mMeasurementsFilePath, std::ios::trunc);
    if (!mMeasurementsFile)
    {
        reportError(fmt::format("Failed to open file '{}'.", mMeasurementsFilePath));
        mMeasurementsFilePath.clear();
    }
    else
    {
        if (mComputeSquaredDifference)
        {
            mMeasurementsFile << "avg_L2_error,red_L2_error,green_L2_error,blue_L2_error" << std::endl;
        }
        else
        {
            mMeasurementsFile << "avg_L1_error,red_L1_error,green_L1_error,blue_L1_error" << std::endl;
        }
        mMeasurementsFile << std::scientific;
    }
}

void ErrorMeasurePass::saveMeasurementsToFile()
{
    if (!mMeasurementsFile) return;

    FALCOR_ASSERT(mMeasurements.valid);
    mMeasurementsFile << mMeasurements.avgError << ",";
    mMeasurementsFile << mMeasurements.error.r << ',' << mMeasurements.error.g << ',' << mMeasurements.error.b;
    mMeasurementsFile << std::endl;
}
