/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/
#include "Framework.h"
#include "VideoEncoderUI.h"
#include "Utils/Platform/OS.h"
#include "Utils/Gui.h"

namespace Falcor
{
    static const Gui::DropdownList kCodecID = 
    {
        { (int32_t)VideoEncoder::CodecID::RawVideo, std::string("Uncompressed") },
        { (int32_t)VideoEncoder::CodecID::H264, std::string("H.264") },
        { (int32_t)VideoEncoder::CodecID::HEVC, std::string("HEVC(H.265)") },
        { (int32_t)VideoEncoder::CodecID::MPEG2, std::string("MPEG2") },
        { (int32_t)VideoEncoder::CodecID::MPEG4, std::string("MPEG4") }
    };

    VideoEncoderUI::UniquePtr VideoEncoderUI::create(uint32_t topLeftX, uint32_t topLeftY, uint32_t width, uint32_t height, Callback startCaptureCB, Callback endCaptureCB)
    {
        return UniquePtr(new VideoEncoderUI(topLeftX, topLeftY, width, height, startCaptureCB, endCaptureCB));
    }

    VideoEncoderUI::VideoEncoderUI(uint32_t topLeftX, uint32_t topLeftY, uint32_t width, uint32_t height, Callback startCaptureCB, Callback endCaptureCB) : mStartCB(startCaptureCB), mEndCB(endCaptureCB)
    {
        mWindowDims.x = topLeftX;
        mWindowDims.y = topLeftY;
        mWindowDims.width = width;
        mWindowDims.height = height;
    }

    void VideoEncoderUI::render(Gui* pGui)
    {
        if (mCapturing)
        {
            endCaptureUI(pGui);
        }
        else
        {
            startCaptureUI(pGui);
        }
    }

    void VideoEncoderUI::startCaptureUI(Gui* pGui)
    {
        pGui->pushWindow("Video Capture", mWindowDims.width, mWindowDims.height, mWindowDims.x, mWindowDims.y);
        pGui->addDropdown("Codec", kCodecID, (uint32_t&)mCodec);
        pGui->addIntVar("Video FPS", (int32_t&)mFPS, 0, 240, 1);

        if(pGui->beginGroup("Codec Options"))
        {
            pGui->addFloatVar("Bitrate (Mbps)", mBitrate, 0, FLT_MAX, 0.01f);
            pGui->addIntVar("GOP Size", (int32_t&)mGopSize, 0, 100000, 1);
            pGui->endGroup();
        }

        pGui->addCheckBox("Capture UI", mCaptureUI);
        pGui->addTooltip("Check this box if you want the GUI recorded");
        pGui->addCheckBox("Use Time-Range", mUseTimeRange);
        if(mUseTimeRange)
        {
            if (pGui->beginGroup("Time Range"))
            {
                pGui->addFloatVar("Start Time", mStartTime, 0, FLT_MAX, 0.001f);
                pGui->addFloatVar("End Time", mEndTime, 0, FLT_MAX, 0.001f);
                pGui->endGroup();
            }
        }

        if (pGui->addButton("Start Recording"))
        {
            startCapture();
        }
        if (pGui->addButton("Cancel", true))
        {
            mEndCB();
        }
        pGui->popWindow();
    }

    VideoEncoderUI::~VideoEncoderUI() = default;

    void VideoEncoderUI::startCapture()
    {
        if(saveFileDialog(VideoEncoder::getSupportedContainerForCodec(mCodec).c_str(), mFilename))
        {
            if(!mUseTimeRange)
            {
                mCaptureUI = true;
            }

            mCapturing = true;

            // Call the users callback
            mStartCB();
        }
    }

    void VideoEncoderUI::endCaptureUI(Gui* pGui)
    {
        if (mCaptureUI)
        {
            pGui->pushWindow("Video Capture");
            if (pGui->addButton("End Recording"))
            {
                mEndCB();
                mCapturing = false;
            }
            pGui->popWindow();
        }

    }
}