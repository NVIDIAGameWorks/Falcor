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
#include "Core/Macros.h"
#include "Core/API/Formats.h"
#include "Core/Platform/OS.h"
#include <filesystem>
#include <memory>

struct AVFormatContext;
struct AVStream;
struct AVFrame;
struct SwsContext;
struct AVCodecContext;

namespace Falcor
{
    class FALCOR_API VideoEncoder
    {
    public:
        using UniquePtr = std::unique_ptr<VideoEncoder>;
        using UniqueConstPtr = std::unique_ptr<const VideoEncoder>;

        enum class Codec : int32_t
        {
            Raw,
            H264,
            HEVC,
            MPEG2,
            MPEG4,
        };

        struct Desc
        {
            uint32_t fps = 60;
            uint32_t width = 0;
            uint32_t height = 0;
            float bitrateMbps = 4;
            uint32_t gopSize = 10;
            Codec codec = Codec::Raw;
            ResourceFormat format = ResourceFormat::BGRA8UnormSrgb;
            bool flipY = false;
            std::filesystem::path path;
        };

        ~VideoEncoder();

        /** Create a video encoder.
            \param[in] desc Encoder settings.
            \return A new encoder object, or returns nullptr on error. For example, due unsupported encoder settings.
        */
        static UniquePtr create(const Desc& desc);

        void appendFrame(const void* pData);
        void endCapture();

        static bool isFormatSupported(ResourceFormat format);
        static FileDialogFilterVec getSupportedContainerForCodec(Codec codec);

    private:
        VideoEncoder(const std::filesystem::path& path);
        bool init(const Desc& desc);

        AVFormatContext* mpOutputContext = nullptr;
        AVStream*        mpOutputStream  = nullptr;
        AVFrame*         mpFrame         = nullptr;
        SwsContext*      mpSwsContext    = nullptr;
        AVCodecContext*  mpCodecContext = nullptr;

        const std::filesystem::path mPath;
        ResourceFormat mFormat;
        uint32_t mRowPitch = 0;
        std::unique_ptr<uint8_t[]> mpFlippedImage; // Used in case the image memory layout if bottom->top
    };
}
