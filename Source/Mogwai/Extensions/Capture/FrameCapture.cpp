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
#include "Falcor.h"
#include "FrameCapture.h"
#include "Utils/Scripting/ScriptWriter.h"
#include <filesystem>

namespace Mogwai
{
    namespace
    {
        const std::string kScriptVar = "frameCapture";
        const std::string kPrintFrames = "print";
        const std::string kFrames = "frames";
        const std::string kAddFrames = "addFrames";
        const std::string kUI = "ui";
        const std::string kOutputs = "outputs";
        const std::string kCapture = "capture";

        template<typename T>
        std::vector<typename T::value_type::first_type> getFirstOfPair(const T& pair)
        {
            std::vector<typename T::value_type::first_type> v;
            v.reserve(pair.size());
            for (auto p : pair) v.push_back(p.first);
            return v;
        }
    }

    MOGWAI_EXTENSION(FrameCapture);

    FrameCapture::UniquePtr FrameCapture::create(Renderer* pRenderer)
    {
        return UniquePtr(new FrameCapture(pRenderer));
    }

    FrameCapture::FrameCapture(Renderer* pRenderer)
        : CaptureTrigger(pRenderer, "Frame Capture")
    {
        mpImageProcessing = ImageProcessing::create();
    }

    void FrameCapture::renderUI(Gui* pGui)
    {
        if (mShowUI)
        {
            auto w = Gui::Window(pGui, mName.c_str(), mShowUI, {}, { 800, 400 });

            CaptureTrigger::renderUI(w);

            w.checkbox("Capture All Outputs", mCaptureAllOutputs);
            w.tooltip("Capture all available outputs instead of the marked ones only.");

            if (w.button("Capture Current Frame")) capture();
        }
    }

    void FrameCapture::registerScriptBindings(pybind11::module& m)
    {
        using namespace pybind11::literals;

        CaptureTrigger::registerScriptBindings(m);

        pybind11::class_<FrameCapture, CaptureTrigger> frameCapture(m, "FrameCapture");

        // Members
        frameCapture.def(kAddFrames.c_str(), pybind11::overload_cast<const RenderGraph*, const uint64_vec&>(&FrameCapture::addFrames), "graph"_a, "frames"_a);
        frameCapture.def(kAddFrames.c_str(), pybind11::overload_cast<const std::string&, const uint64_vec&>(&FrameCapture::addFrames), "name"_a, "frames"_a);

        auto printGraph = [](FrameCapture* pFC, RenderGraph* pGraph) { pybind11::print(pFC->graphFramesStr(pGraph)); };
        frameCapture.def(kPrintFrames.c_str(), printGraph, "graph"_a);
        frameCapture.def(kCapture.c_str(), &FrameCapture::capture);
        auto printAllGraphs = [](FrameCapture* pFC)
        {
            std::string s;
            for (const auto& g : pFC->mGraphRanges) {s += "'" + g.first->getName() + "':\n" + pFC->graphFramesStr(g.first) + "\n";}
            pybind11::print(s.empty() ? "Empty" : s);
        };
        frameCapture.def(kPrintFrames.c_str(), printAllGraphs);

        // Settings
        auto getUI = [](FrameCapture* pFC) { return pFC->mShowUI; };
        auto setUI = [](FrameCapture* pFC, bool show) { pFC->mShowUI = show; };
        frameCapture.def_property(kUI.c_str(), getUI, setUI);

        frameCapture.def_property("captureAllOutputs",
            [](FrameCapture* pFC){ return pFC->mCaptureAllOutputs;},
            [](FrameCapture* pFC, bool all){ pFC->mCaptureAllOutputs = all; });
    }

    std::string FrameCapture::getScriptVar() const
    {
        return kScriptVar;
    }

    std::string FrameCapture::getScript(const std::string& var) const
    {
        std::string s;

        s += "# Frame Capture\n";
        s += CaptureTrigger::getScript(var);

        for (const auto& g : mGraphRanges)
        {
            s += ScriptWriter::makeMemberFunc(var, kAddFrames, g.first->getName(), getFirstOfPair(g.second));
        }
        return s;
    }

    void FrameCapture::triggerFrame(RenderContext* pRenderContext, RenderGraph* pGraph, uint64_t frameID)
    {
        std::vector<std::string> unmarkedOutputs;

        if (mCaptureAllOutputs)
        {
            // Mark all outputs and (re)execute the graph.
            unmarkedOutputs = pGraph->getUnmarkedOutputs();
            for (const auto& output : unmarkedOutputs) pGraph->markOutput(output);
            pGraph->execute(pRenderContext);
        }

        for (uint32_t i = 0 ; i < pGraph->getOutputCount() ; i++)
        {
            captureOutput(pRenderContext, pGraph, i);
        }

        if (mCaptureAllOutputs && !unmarkedOutputs.empty())
        {
            for (const auto& output : unmarkedOutputs) pGraph->unmarkOutput(output);
            pGraph->compile(pRenderContext);
        }
    }

    void FrameCapture::captureOutput(RenderContext* pRenderContext, RenderGraph* pGraph, const uint32_t outputIndex)
    {
        const std::string outputName = pGraph->getOutputName(outputIndex);
        const std::string basename = getOutputNamePrefix(outputName) + std::to_string(gpFramework->getGlobalClock().getFrame());

        const Texture::SharedPtr pOutput = pGraph->getOutput(outputIndex)->asTexture();
        if (!pOutput) throw RuntimeError("Graph output {} is not a texture", outputName);

        const ResourceFormat format = pOutput->getFormat();
        const uint32_t channels = getFormatChannelCount(format);

        for (auto mask : pGraph->getOutputMasks(outputIndex))
        {
            // Determine output color channels and filename suffix.
            std::string suffix;
            uint32_t outputChannels = 0;

            switch (mask)
            {
            case TextureChannelFlags::Red: suffix = ".R"; outputChannels = 1; break;
            case TextureChannelFlags::Green: suffix = ".G"; outputChannels = 1; break;
            case TextureChannelFlags::Blue: suffix = ".B"; outputChannels = 1; break;
            case TextureChannelFlags::Alpha: suffix = ".A"; outputChannels = 1; break;
            case TextureChannelFlags::RGB: /* No suffix */ outputChannels = 3; break;
            case TextureChannelFlags::RGBA: suffix = ".RGBA"; outputChannels = 4; break;
            default:
                logWarning("Graph output {} mask {:#x} is not supported. Skipping.", outputName, (uint32_t)mask);
                continue;
            }

            // Copy relevant channels into new texture if necessary.
            Texture::SharedPtr pTex = pOutput;
            if (outputChannels == 1 && channels > 1)
            {
                // Determine output format.
                ResourceFormat outputFormat = ResourceFormat::Unknown;
                uint bits = getNumChannelBits(format, mask);

                switch (getFormatType(format))
                {
                case FormatType::Unorm:
                case FormatType::UnormSrgb:
                    if (bits == 8) outputFormat = ResourceFormat::R8Unorm;
                    else if (bits == 16) outputFormat = ResourceFormat::R16Unorm;
                    break;
                case FormatType::Snorm:
                    if (bits == 8) outputFormat = ResourceFormat::R8Snorm;
                    else if (bits == 16) outputFormat = ResourceFormat::R16Snorm;
                    break;
                case FormatType::Uint:
                    if (bits == 8) outputFormat = ResourceFormat::R8Uint;
                    else if (bits == 16) outputFormat = ResourceFormat::R16Uint;
                    else if (bits == 32) outputFormat = ResourceFormat::R32Uint;
                    break;
                case FormatType::Sint:
                    if (bits == 8) outputFormat = ResourceFormat::R8Int;
                    else if (bits == 16) outputFormat = ResourceFormat::R16Int;
                    else if (bits == 32) outputFormat = ResourceFormat::R32Int;
                    break;
                case FormatType::Float:
                    if (bits == 16) outputFormat = ResourceFormat::R16Float;
                    else if (bits == 32) outputFormat = ResourceFormat::R32Float;
                    break;
                }

                if (outputFormat == ResourceFormat::Unknown)
                {
                    logWarning("Graph output {} mask {:#x} failed to determine output format. Skipping.", outputName, (uint32_t)mask);
                    continue;
                }

                // If extracting a single R, G or B channel from an SRGB format we may lose some precision in the conversion
                // to a singel channel non-SRGB format of the same bit depth. Issue a warning for this case for now.
                // The alternative would be to convert to a higher-precision monochrome format like R32Float,
                // but then the output image will be in a floating-point format which may be undesirable too.
                if (is_set(mask, TextureChannelFlags::RGB) && isSrgbFormat(format))
                {
                    logWarning("Graph output {} mask {:#x} extracting single RGB channel from SRGB format may lose precision.", outputName, (uint32_t)mask);
                }

                // Copy color channel into temporary texture.
                pTex = Texture::create2D(pOutput->getWidth(), pOutput->getHeight(), outputFormat, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
                mpImageProcessing->copyColorChannel(pRenderContext, pOutput->getSRV(0, 1, 0, 1), pTex->getUAV(), mask);
            }

            // Write output image.
            auto ext = Bitmap::getFileExtFromResourceFormat(pTex->getFormat());
            auto fileformat = Bitmap::getFormatFromFileExtension(ext);
            std::string filename = basename + suffix + "." + ext;
            Bitmap::ExportFlags flags = Bitmap::ExportFlags::None;
            if (mask == TextureChannelFlags::RGBA) flags |= Bitmap::ExportFlags::ExportAlpha;

            pTex->captureToFile(0, 0, filename, fileformat, flags);
        }
    }

    void FrameCapture::addFrames(const RenderGraph* pGraph, const uint64_vec& frames)
    {
        for (auto f : frames) addRange(pGraph, f, 1);
    }

    void FrameCapture::addFrames(const std::string& graphName, const uint64_vec& frames)
    {
        auto pGraph = mpRenderer->getGraph(graphName).get();
        if (!pGraph) throw RuntimeError("Can't find a graph named '{}'", graphName);
        this->addFrames(pGraph, frames);
    }

    std::string FrameCapture::graphFramesStr(const RenderGraph* pGraph)
    {
        const auto& ranges = mGraphRanges[pGraph];
        std::string s("\t");
        s += kFrames + " = " + ScriptBindings::repr(getFirstOfPair(ranges));
        return s;
    }

    void FrameCapture::capture()
    {
        auto pGraph = mpRenderer->getActiveGraph();
        if (!pGraph) return;
        uint64_t frameID = gpFramework->getGlobalClock().getFrame();
        triggerFrame(gpDevice->getRenderContext(), pGraph, frameID);
    }
}
