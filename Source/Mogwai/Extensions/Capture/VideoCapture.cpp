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
#include "stdafx.h"
#include "VideoCapture.h"

namespace Mogwai
{
    namespace
    {
        const std::string kScriptVar = "vc";
        const std::string kUI = "ui";
        const std::string kCodec = "codec";
        const std::string kFps = "fps";
        const std::string kBitrate = "bitrate";
        const std::string kGopSize = "gopSize";
        const std::string kRanges = "ranges";
        const std::string kAddRanges = "addRanges";
        const std::string kPrint = "print";
        const std::string kOutputs = "outputs";

        Texture::SharedPtr createTextureForBlit(const Texture* pSource)
        {
            assert(pSource->getType() == Texture::Type::Texture2D);
            return Texture::create2D(pSource->getWidth(), pSource->getHeight(), ResourceFormat::RGBA8UnormSrgb, 1, 1, nullptr, Texture::BindFlags::RenderTarget);
        }
    }

    MOGWAI_EXTENSION(VideoCapture);

    VideoCapture::VideoCapture(Renderer* pRenderer) : CaptureTrigger(pRenderer, "Video Capture")
    {
        mpEncoderUI = VideoEncoderUI::create();
    }

    VideoCapture::UniquePtr VideoCapture::create(Renderer* pRenderer)
    {
        return UniquePtr(new VideoCapture(pRenderer));
    }

    void VideoCapture::renderUI(Gui* pGui)
    {
        if (mShowUI)
        {
            auto w = Gui::Window(pGui, "Video Capture", mShowUI, { 800, 400 });
            CaptureTrigger::renderUI(w);
            w.separator();
            mpEncoderUI->render(w, true);
        }
    }

    void VideoCapture::beginRange(RenderGraph* pGraph, const Range& r)
    {
        VideoEncoder::Desc d;
        d.bitrateMbps = mpEncoderUI->getBitrate();
        d.codec = mpEncoderUI->getCodec();
        d.fps = mpEncoderUI->getFPS();
        d.gopSize = mpEncoderUI->getGopSize();

        for (uint32_t i = 0 ; i < pGraph->getOutputCount() ; i++)
        {
            const auto& outputName = pGraph->getOutputName(i);
            Texture::SharedPtr pTex = pGraph->getOutput(i)->asTexture();
            if (!pTex || pTex->getType() != Texture::Type::Texture2D)
            {
                logError("Can't video capture " + outputName + ". The output is not a Texture2D");
                continue;
            }

            EncodeData encoder;
            auto texFormat = pTex->getFormat();
            if (VideoEncoder::isFormatSupported(texFormat) == false)
            {
                auto res = msgBox("Trying to record graph output " + outputName + " but the resource format is not supported by the video encoder.\nWould you like to capture the output as an RGBA8Srgb resource?\n\nFor HDR textures, this operation will clamp the results", MsgBoxType::YesNo);
                if(res == MsgBoxButton::No) continue;
                encoder.pBlitTex = createTextureForBlit(pTex.get());
                pTex = encoder.pBlitTex;
            }

            d.height = pTex->getHeight();
            d.width = pTex->getWidth();
            d.format = pTex->getFormat();
            d.filename = getOutputNamePrefix(outputName) + std::to_string(r.first) + "." + std::to_string(r.second) + "." + VideoEncoder::getSupportedContainerForCodec(d.codec)[0].ext;
            encoder.output = outputName;
            encoder.pEncoder = VideoEncoder::create(d);
            mEncoders.push_back(std::move(encoder));
        }
    }

    void VideoCapture::endRange(RenderGraph* pGraph, const Range& r)
    {
        for (const auto& e : mEncoders) e.pEncoder->endCapture();
    }

    void VideoCapture::triggerFrame(RenderContext* pCtx, RenderGraph* pGraph, uint64_t frameID)
    {
        for (const auto& e : mEncoders)
        {
            Texture::SharedPtr pTex = std::dynamic_pointer_cast<Texture>(pGraph->getOutput(e.output));
            if (e.pBlitTex)
            {
                pCtx->blit(pTex->getSRV(0, 1, 0, 1), e.pBlitTex->getRTV(0, 0, 1));
                pTex = e.pBlitTex;
            }

            e.pEncoder->appendFrame(pCtx->readTextureSubresource(pTex.get(), 0).data());
        }
    }

    void VideoCapture::scriptBindings(Bindings& bindings)
    {
        CaptureTrigger::scriptBindings(bindings);
        auto& m = bindings.getModule();

        pybind11::class_<VideoCapture, CaptureTrigger> videoCapture(m, "VideoCapture");

        bindings.addGlobalObject(kScriptVar, this, "Video Capture Helpers");

        // UI
        auto getUI = [](VideoCapture* pVC) { return pVC->mShowUI; };
        auto setUI = [](VideoCapture* pVC, bool show) { pVC->mShowUI = show; };
        videoCapture.def_property(kUI.c_str(), getUI, setUI);

        // Settings
        auto getCodec = [](VideoCapture* pVC) {return pVC->mpEncoderUI->getCodec(); };
        auto setCodec = [](VideoCapture* pVC, VideoEncoder::Codec c) {pVC->mpEncoderUI->setCodec(c); return pVC; };
        videoCapture.def_property(kCodec.c_str(), getCodec, setCodec);

        auto getFPS = [](VideoCapture* pVC) {return pVC->mpEncoderUI->getFPS(); };
        auto setFPS = [](VideoCapture* pVC, uint32_t fps) {pVC->mpEncoderUI->setFPS(fps); return pVC; };
        videoCapture.def_property(kFps.c_str(), getFPS, setFPS);

        auto getBitrate = [](VideoCapture* pVC) {return pVC->mpEncoderUI->getBitrate(); };
        auto setBitrate = [](VideoCapture* pVC, float bitrate) {pVC->mpEncoderUI->setBitrate(bitrate); return pVC; };
        videoCapture.def_property(kBitrate.c_str(), getBitrate, setBitrate);

        auto getGopSize = [](VideoCapture* pVC) {return pVC->mpEncoderUI->getGopSize(); };
        auto setGopSize = [](VideoCapture* pVC, uint32_t gop) {pVC->mpEncoderUI->setGopSize(gop); return pVC; };
        videoCapture.def_property(kGopSize.c_str(), getGopSize, setGopSize);

        // Ranges
        videoCapture.def(kRanges.c_str(), pybind11::overload_cast<const RenderGraph*, const range_vec&>(&VideoCapture::addRanges)); // PYTHONDEPRECATED
        videoCapture.def(kRanges.c_str(), pybind11::overload_cast<const std::string&, const range_vec&>(&VideoCapture::addRanges)); // PYTHONDEPRECATED
        videoCapture.def(kAddRanges.c_str(), pybind11::overload_cast<const RenderGraph*, const range_vec&>(&VideoCapture::addRanges), "graph"_a, "ranges"_a);
        videoCapture.def(kAddRanges.c_str(), pybind11::overload_cast<const std::string&, const range_vec&>(&VideoCapture::addRanges), "name"_a, "ranges"_a);

        auto printGraph = [](VideoCapture* pVC, RenderGraph* pGraph) { pybind11::print(pVC->graphRangesStr(pGraph)); };
        videoCapture.def(kPrint.c_str(), printGraph, "graph"_a);

        auto printAllGraphs = [](VideoCapture* pVC)
        {
            std::string s;
            for (const auto& g : pVC->mGraphRanges) { s += "'" + g.first->getName() + "':\n" + pVC->graphRangesStr(g.first) + "\n"; }
            pybind11::print(s.empty() ? "Empty" : s);
        };
        videoCapture.def(kPrint.c_str(), printAllGraphs);
    }

    std::string VideoCapture::getScript()
    {
        if (mGraphRanges.empty()) return "";

        std::string s("# Video Capture\n");
        s += CaptureTrigger::getScript(kScriptVar);
        s += Scripting::makeSetProperty(kScriptVar, kCodec, mpEncoderUI->getCodec());
        s += Scripting::makeSetProperty(kScriptVar, kFps, mpEncoderUI->getFPS());
        s += Scripting::makeSetProperty(kScriptVar, kBitrate, mpEncoderUI->getBitrate());
        s += Scripting::makeSetProperty(kScriptVar, kGopSize, mpEncoderUI->getGopSize());

        for (const auto& g : mGraphRanges)
        {
            s += Scripting::makeMemberFunc(kScriptVar, kAddRanges, g.first->getName(), g.second);
        }
        return s;
    }

    void VideoCapture::addRanges(const RenderGraph* pGraph, const range_vec& ranges)
    {
        for (auto r : ranges) addRange(pGraph, r.first, r.second);
    }

    void VideoCapture::addRanges(const std::string& graphName, const range_vec& ranges)
    {
        auto pGraph = mpRenderer->getGraph(graphName).get();
        if (!pGraph) throw std::runtime_error("Can't find a graph named '" + graphName + "'");
        this->addRanges(pGraph, ranges);
    }

    std::string VideoCapture::graphRangesStr(const RenderGraph* pGraph)
    {
        const auto& g = mGraphRanges[pGraph];
        std::string s("\t");
        s += kRanges + " = " + ScriptBindings::repr(g);
        return s;
    }
}

