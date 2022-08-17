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
#pragma once
#include "VideoEncoder.h"
#include "Core/Macros.h"
#include "Utils/UI/Gui.h"
#include <filesystem>
#include <functional>
#include <memory>

namespace Falcor
{
    class FALCOR_API VideoEncoderUI
    {
    public:
        using UniquePtr = std::unique_ptr<VideoEncoderUI>;
        using UniqueConstPtr = std::unique_ptr<const VideoEncoderUI>;
        using CallbackStart = std::function<bool(void)>;
        using CallbackEnd = std::function<void(void)>;

        ~VideoEncoderUI() = default;

        /** Create video encoder UI.
            \param[in] startCaptureCB Optional function called at start of capture.
            \param[in] startCaptureCB Optional function called at end of capture.
            \return New object, or throws an exception if creation failed.
        */
        static UniquePtr create(CallbackStart startCaptureCB = nullptr, CallbackEnd endCaptureCB = nullptr);

        void render(Gui::Window& w, bool codecOnly = false);
        void setCaptureState(bool state);

        VideoEncoder::Codec getCodec() const { return mCodec; }
        uint32_t getFPS() const { return mFPS; }
        float getBitrate() const { return mBitrate; }
        uint32_t getGopSize() const { return mGopSize; }

        VideoEncoderUI& setCodec(VideoEncoder::Codec c) { mCodec = c; return *this; }
        VideoEncoderUI& setFPS(uint32_t fps) { mFPS = fps; return *this; }
        VideoEncoderUI& setBitrate(float bitrate) { mBitrate = bitrate; return *this; }
        VideoEncoderUI& setGopSize(uint32_t gopSize) { mGopSize = gopSize; return *this; }

        bool useTimeRange() const { return mUseTimeRange; }
        bool captureUI() const { return mCaptureUI; }
        float getStartTime() const { return mStartTime; }
        float getEndTime() const { return mEndTime; }
        const std::filesystem::path& getPath() const { return mPath; }

    private:
        VideoEncoderUI(CallbackStart startCaptureCB, CallbackEnd endCaptureCB);

        void startCapture();
        void endCapture();
        void startCaptureUI(Gui::Window& w, bool codecOnly);
        void endCaptureUI(Gui::Window& w, bool codecOnly);

        bool mCapturing = false;
        CallbackStart mStartCB = nullptr;
        CallbackEnd mEndCB = nullptr;

        uint32_t mFPS = 60;
        VideoEncoder::Codec mCodec = VideoEncoder::Codec::Raw;

        bool mUseTimeRange = false;
        bool mCaptureUI = false;
        bool mResetOnFirstFrame = false;
        float mStartTime = 0;
        float mEndTime = 4.f;

        std::filesystem::path mPath;
        float mBitrate = 30.f;
        uint32_t mGopSize = 10;
    };
}
