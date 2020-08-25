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
#pragma once
#include "../../Mogwai.h"
#include "CaptureTrigger.h"
#include "Utils/Video/VideoEncoderUI.h"
#include "Utils/Video/VideoEncoder.h"

namespace Mogwai
{
    class VideoCapture : public CaptureTrigger
    {
    public:
        static UniquePtr create(Renderer* pRenderer);
        virtual void renderUI(Gui* pGui) override;
        virtual void beginRange(RenderGraph* pGraph, const Range& r) override;
        virtual void endRange(RenderGraph* pGraph, const Range& r) override;
        virtual void scriptBindings(Bindings& bindings) override;
        virtual std::string getScript() override;
        virtual void triggerFrame(RenderContext* pCtx, RenderGraph* pGraph, uint64_t frameID) override;

    private:
        VideoCapture(Renderer* pRenderer);

        void addRanges(const RenderGraph* pGraph, const range_vec& ranges);
        void addRanges(const std::string& graphName, const range_vec& ranges);
        std::string graphRangesStr(const RenderGraph* pGraph);

        VideoEncoderUI::UniquePtr mpEncoderUI;

        struct EncodeData
        {
            std::string output;
            VideoEncoder::UniquePtr pEncoder;
            Texture::SharedPtr pBlitTex;
        };
        std::vector<EncodeData> mEncoders;
    };
}
