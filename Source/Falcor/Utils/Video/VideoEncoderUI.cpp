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
#include "VideoEncoderUI.h"
#include "Core/Assert.h"
#include "Utils/UI/Gui.h"

namespace Falcor
{
    static const Gui::DropdownList kCodecID =
    {
        { (uint32_t)VideoEncoder::Codec::Raw, std::string("Uncompressed") },
        { (uint32_t)VideoEncoder::Codec::H264, std::string("H.264") },
        { (uint32_t)VideoEncoder::Codec::HEVC, std::string("HEVC(H.265)") },
        { (uint32_t)VideoEncoder::Codec::MPEG2, std::string("MPEG2") },
        { (uint32_t)VideoEncoder::Codec::MPEG4, std::string("MPEG4") }
    };

    VideoEncoderUI::UniquePtr VideoEncoderUI::create(CallbackStart startCaptureCB, CallbackEnd endCaptureCB)
    {
        return UniquePtr(new VideoEncoderUI(startCaptureCB, endCaptureCB));
    }

    VideoEncoderUI::VideoEncoderUI(CallbackStart startCaptureCB, CallbackEnd endCaptureCB)
        : mStartCB(startCaptureCB)
        , mEndCB(endCaptureCB)
    {
    }

    void VideoEncoderUI::render(Gui::Window& w, bool codecOnly)
    {
        mCapturing ? endCaptureUI(w, codecOnly) : startCaptureUI(w, codecOnly);
    }

    void VideoEncoderUI::setCaptureState(bool state)
    {
        mCapturing = state;
    }

    void VideoEncoderUI::startCaptureUI(Gui::Window& w, bool codecOnly)
    {
        {
            auto g = w.group("Codec Options");
            g.dropdown("Codec", kCodecID, (uint32_t&)mCodec);
            g.var("Video FPS", mFPS, 0u, 240u, 1);
            g.var("Bitrate (Mbps)", mBitrate, 0.f, FLT_MAX, 0.01f);
            g.var("GOP Size", mGopSize, 0u, 100000u, 1);
        }

        if (codecOnly) return;

        {
            auto g = w.group("Capture Options", true);

            g.checkbox("Capture UI", mCaptureUI);
            g.tooltip("Check this box if you want the GUI recorded");

            g.checkbox("Reset rendering", mResetOnFirstFrame);
            g.tooltip("Check this box if you want the rendering to be reset for the first frame, for example to reset temporal accumulation");

            g.checkbox("Use Time-Range", mUseTimeRange);
            if (mUseTimeRange)
            {
                auto g = w.group("Time Range", true);
                g.var("Start Time", mStartTime, 0.f, FLT_MAX, 0.001f);
                g.var("End Time", mEndTime, 0.f, FLT_MAX, 0.001f);
            }
        }

        if (mStartCB && w.button("Start Recording")) startCapture();
        if (mEndCB && w.button("Cancel", true)) endCapture();
    }

    void VideoEncoderUI::startCapture()
    {
        if (!mCapturing)
        {
            if (saveFileDialog(VideoEncoder::getSupportedContainerForCodec(mCodec), mPath))
            {
                FALCOR_ASSERT(mStartCB);
                mCapturing = mStartCB();
            }
        }
    }

    void VideoEncoderUI::endCaptureUI(Gui::Window& w, bool codecOnly)
    {
        if (mEndCB)
        {
            if (w.button("End Recording"))
            {
                endCapture();
            }
        }
    }

    void VideoEncoderUI::endCapture()
    {
        if (mCapturing)
        {
            FALCOR_ASSERT(mEndCB);
            mEndCB();
            mCapturing = false;
        }
    }
}
