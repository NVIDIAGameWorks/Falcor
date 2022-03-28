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
#include "stdafx.h"
#include "VideoEncoder.h"

extern "C"
{
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
}

namespace Falcor
{
    namespace
    {
        AVPixelFormat getPictureFormatFromCodec(AVCodecID codec)
        {
            switch (codec)
            {
            case AV_CODEC_ID_RAWVIDEO:
                return AV_PIX_FMT_BGR24;
            case AV_CODEC_ID_H264:
            case AV_CODEC_ID_HEVC:
            case AV_CODEC_ID_MPEG2VIDEO:
                return AV_PIX_FMT_YUV422P;
            case AV_CODEC_ID_MPEG4:
                return AV_PIX_FMT_YUV420P;
            default:
                FALCOR_UNREACHABLE();
                return AV_PIX_FMT_NONE;
            }
        }

        AVPixelFormat getPictureFormatFromFalcorFormat(ResourceFormat format)
        {
            switch (format)
            {
            case ResourceFormat::RGBA8Unorm:
            case ResourceFormat::RGBA8UnormSrgb:
                return AV_PIX_FMT_RGBA;
            case ResourceFormat::BGRA8Unorm:
            case ResourceFormat::BGRA8UnormSrgb:
                return AV_PIX_FMT_BGRA;
            default:
                return AV_PIX_FMT_NONE;
            }
        }

        AVCodecID getCodecID(VideoEncoder::Codec codec)
        {
            switch (codec)
            {
            case VideoEncoder::Codec::Raw:
                return AV_CODEC_ID_RAWVIDEO;
            case VideoEncoder::Codec::H264:
                return AV_CODEC_ID_H264;
            case VideoEncoder::Codec::HEVC:
                return AV_CODEC_ID_HEVC;
            case VideoEncoder::Codec::MPEG2:
                return AV_CODEC_ID_MPEG2VIDEO;
            case VideoEncoder::Codec::MPEG4:
                return AV_CODEC_ID_MPEG4;
            default:
                FALCOR_UNREACHABLE();
                return AV_CODEC_ID_NONE;
            }
        }

        static bool error(const std::filesystem::path& path, const std::string& msg)
        {
            reportError(fmt::format("Error when creating video capture file '{}'.\n{}", path, msg));
            return false;
        }

        AVCodecContext* createCodecContext(AVFormatContext* pCtx, uint32_t width, uint32_t height, uint32_t fps, float bitrateMbps, uint32_t gopSize, AVCodecID codecID, AVCodec* pCodec)
        {
            // Initialize the codec context
            AVCodecContext* pCodecCtx = avcodec_alloc_context3(pCodec);
            pCodecCtx->codec_id = codecID;
            pCodecCtx->bit_rate = (int)(bitrateMbps * 1000 * 1000);
            pCodecCtx->width = width;
            pCodecCtx->height = height;
            pCodecCtx->time_base = { 1, (int)fps };
            pCodecCtx->gop_size = gopSize;
            pCodecCtx->pix_fmt = getPictureFormatFromCodec(codecID);

            // Some formats want stream headers to be separate
            if (pCtx->oformat->flags & AVFMT_GLOBALHEADER)
            {
                pCodecCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
            }

            return pCodecCtx;
        }

        AVStream* createVideoStream(AVFormatContext* pCtx, uint32_t fps, AVCodecID codecID, const std::filesystem::path& path, AVCodec*& pCodec)
        {
            // Get the encoder
            pCodec = avcodec_find_encoder(codecID);
            if (pCodec == nullptr)
            {
                error(path, std::string("Can't find ") + avcodec_get_name(codecID) + " encoder.");
                return nullptr;
            }

            // create the video stream
            AVStream* pStream = avformat_new_stream(pCtx, nullptr);
            if (pStream == nullptr)
            {
                error(path, "Failed to create video stream.");
                return nullptr;
            }
            pStream->id = pCtx->nb_streams - 1;
            pStream->time_base = { 1, (int)fps };
            return pStream;
        }

        AVFrame* allocateFrame(int format, uint32_t width, uint32_t height, const std::filesystem::path& path)
        {
            AVFrame* pFrame = av_frame_alloc();
            if (pFrame == nullptr)
            {
                error(path, "Video frame allocation failed.");
                return nullptr;
            }

            pFrame->format = format;
            pFrame->width = width;
            pFrame->height = height;
            pFrame->pts = 0;

            // Allocate the buffer for the encoded image
            if (av_frame_get_buffer(pFrame, 32) < 0)
            {
                error(path, "Can't allocate destination picture");
                return nullptr;
            }

            return pFrame;
        }

        bool openVideo(AVCodec* pCodec, AVCodecContext* pCodecCtx, AVFrame*& pFrame, const std::filesystem::path& path)
        {
            AVDictionary* param = nullptr;

            if (pCodecCtx->codec_id == AV_CODEC_ID_H264)
            {
                // H.264 defaults to lossless currently. This should be changed in the future.
                av_dict_set(&param, "qp", "0", 0);
                /*
                Change options to trade off compression efficiency against encoding speed. If you specify a preset, the changes it makes will be applied before all other parameters are applied.
                You should generally set this option to the slowest you can bear.
                Values available: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow, placebo.
                */
                av_dict_set(&param, "preset", "veryslow", 0);
            }

            // Open the codec
            if (avcodec_open2(pCodecCtx, pCodec, &param) < 0)
            {
                return error(path, "Can't open video codec.");
            }
            av_dict_free(&param);

            // create a frame
            pFrame = allocateFrame(pCodecCtx->pix_fmt, pCodecCtx->width, pCodecCtx->height, path);
            if (pFrame == nullptr)
            {
                return false;
            }
            return true;
        }
    }

    VideoEncoder::VideoEncoder(const std::filesystem::path& path)
        : mPath(path)
    {
    }

    VideoEncoder::~VideoEncoder()
    {
        endCapture();
    }

    VideoEncoder::UniquePtr VideoEncoder::create(const Desc& desc)
    {
        UniquePtr pVC = UniquePtr(new VideoEncoder(desc.path));
        FALCOR_ASSERT(pVC);

        // Initialize the encoder. This may fail, in which case we return nullptr.
        if (pVC->init(desc) == false)
        {
            pVC = nullptr;
        }
        return pVC;
    }

    bool VideoEncoder::isFormatSupported(ResourceFormat format)
    {
        return getPictureFormatFromFalcorFormat(format) != AV_PIX_FMT_NONE;
    }

    bool VideoEncoder::init(const Desc& desc)
    {
        // av_register_all() is deprecated since 58.9.100, but Linux repos may not get a newer version, so this call cannot be completely removed.
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 9, 100)
        av_register_all();
#endif
        // Create the output context
        avformat_alloc_output_context2(&mpOutputContext, nullptr, nullptr, mPath.string().c_str());
        if(mpOutputContext == nullptr)
        {
            // The sample tries again, while explicitly requesting mpeg format. I chose not to do it, since it might lead to a container with a wrong extension
            return error(mPath, "File output format not recognized. Make sure you use a known file extension (avi/mpeg/mp4)");
        }

        // Get the output format of the container
        AVOutputFormat* pOutputFormat = mpOutputContext->oformat;
        FALCOR_ASSERT((pOutputFormat->flags & AVFMT_NOFILE) == 0); // Problem. We want a file.

        // Create the video codec
        AVCodec* pVideoCodec;
        mpOutputStream = createVideoStream(mpOutputContext, desc.fps, getCodecID(desc.codec), mPath, pVideoCodec);
        if(mpOutputStream == nullptr)
        {
            return false;
        }

        mpCodecContext = createCodecContext(mpOutputContext, desc.width, desc.height, desc.fps, desc.bitrateMbps, desc.gopSize, getCodecID(desc.codec), pVideoCodec);
        if(mpCodecContext == nullptr)
        {
            return false;
        }

        // Open the video stream
        if(openVideo(pVideoCodec, mpCodecContext, mpFrame, mPath) == false)
        {
            return false;
        }

        // copy the stream parameters to the muxer
        if(avcodec_parameters_from_context(mpOutputStream->codecpar, mpCodecContext) < 0)
        {
            return error(desc.path, "Could not copy the stream parameters\n");
        }

        av_dump_format(mpOutputContext, 0, mPath.string().c_str(), 1);

        // Open the output file
        FALCOR_ASSERT((pOutputFormat->flags & AVFMT_NOFILE) == 0); // No output file required. Not sure if/when this happens.
        if(avio_open(&mpOutputContext->pb, mPath.string().c_str(), AVIO_FLAG_WRITE) < 0)
        {
            return error(mPath, "Can't open output file.");
        }

        // Write the stream header
        if(avformat_write_header(mpOutputContext, nullptr) < 0)
        {
            return error(mPath, "Can't write file header.");
        }

        mFormat = desc.format;
        mRowPitch = getFormatBytesPerBlock(desc.format) * desc.width;
        if(desc.flipY)
        {
            mpFlippedImage = new uint8_t[desc.height * mRowPitch];
        }

        FALCOR_ASSERT(isFormatSupported(desc.format));
        mpSwsContext = sws_getContext(desc.width, desc.height, getPictureFormatFromFalcorFormat(desc.format), desc.width, desc.height, mpCodecContext->pix_fmt, SWS_POINT, nullptr, nullptr, nullptr);
        if(mpSwsContext == nullptr)
        {
            return error(mPath, "Failed to allocate SWScale context");
        }
        return true;
    }

    bool flush(AVCodecContext* pCodecContext, AVFormatContext* pOutputContext, AVStream* pOutputStream, const std::filesystem::path& path)
    {
        while(true)
        {
            // Allocate a packet
            std::unique_ptr<AVPacket, std::function<void(AVPacket*)>> pPacket(av_packet_alloc(), [] (AVPacket* pPacket) { av_packet_free(&pPacket); });

            int r = avcodec_receive_packet(pCodecContext, pPacket.get());
            if(r == AVERROR(EAGAIN) || r == AVERROR_EOF)
            {
                return true;
            }
            else if(r < 0)
            {
                error(path, "Can't retrieve packet");
                return false;
            }

            // rescale output packet timestamp values from codec to stream timebase
            av_packet_rescale_ts(pPacket.get(), pCodecContext->time_base, pOutputStream->time_base);
            pPacket->stream_index = pOutputStream->index;
            r = av_interleaved_write_frame(pOutputContext, pPacket.get());
            if(r < 0)
            {
                char msg[1024];
                av_make_error_string(msg, 1024, r);
                error(path, "Failed when writing encoded frame to file");
                return false;
            }
        }
    }

    void VideoEncoder::endCapture()
    {
        if(mpOutputContext)
        {
            // Flush the codex
            avcodec_send_frame(mpCodecContext, nullptr);
            flush(mpCodecContext, mpOutputContext, mpOutputStream, mPath);

            av_write_trailer(mpOutputContext);

            avio_closep(&mpOutputContext->pb);
            avcodec_free_context(&mpCodecContext);
            av_frame_free(&mpFrame);
            sws_freeContext(mpSwsContext);
            avformat_free_context(mpOutputContext);
            mpOutputContext = nullptr;
            mpOutputStream = nullptr;
        }
        safe_delete(mpFlippedImage);
    }

    void VideoEncoder::appendFrame(const void* pData)
    {
        if(mpFlippedImage)
        {
            // Flip the image
            for(int32_t h = 0; h < mpCodecContext->height; h++)
            {
                const uint8_t* pSrc = (uint8_t*)pData + h * mRowPitch;
                uint8_t* pDst = mpFlippedImage + (mpCodecContext->height - 1 - h) * mRowPitch;
                memcpy(pDst, pSrc, mRowPitch);
            }

            pData = mpFlippedImage;
        }

        uint8_t* src[AV_NUM_DATA_POINTERS] = {0};
        int32_t rowPitch[AV_NUM_DATA_POINTERS] = {0};
        src[0] = (uint8_t*)pData;
        rowPitch[0] = (int32_t)mRowPitch;

        // Scale and convert the image
        sws_scale(mpSwsContext, src, rowPitch, 0, mpCodecContext->height, mpFrame->data, mpFrame->linesize);

        // Encode the frame
        int r = avcodec_send_frame(mpCodecContext, mpFrame);
        mpFrame->pts++;
        if(r == AVERROR(EAGAIN))
        {
            if(flush(mpCodecContext, mpOutputContext, mpOutputStream, mPath) == false)
            {
                return;
            }
        }
        else if(r < 0)
        {
            error(mPath, "Can't send video frame");
            return;
        }
    }

    FileDialogFilterVec VideoEncoder::getSupportedContainerForCodec(Codec codec)
    {
        FileDialogFilterVec filters;
        const FileDialogFilter AVI{ "avi", "AVI (Audio Video Interleaved)"};
        const FileDialogFilter MP4{ "mp4", "MP4 (MPEG-4 Part 14)"};
        const FileDialogFilter MKV{ "mkv", "MKV (Matroska)\0*.mkv" };

        switch(codec)
        {
        case VideoEncoder::Codec::Raw:
            filters.push_back(AVI);
            break;
        case VideoEncoder::Codec::H264:
        case VideoEncoder::Codec::MPEG2:
        case VideoEncoder::Codec::MPEG4:
            filters.push_back(MP4);
            filters.push_back(MKV);
            filters.push_back(AVI);
            break;
        case VideoEncoder::Codec::HEVC:
            filters.push_back(MP4);
            filters.push_back(MKV);
            break;
        default:
            FALCOR_UNREACHABLE();
        }
        return filters;
    }

    FALCOR_SCRIPT_BINDING(VideoEncoder)
    {
        pybind11::enum_<VideoEncoder::Codec> codec(m, "Codec");
        codec.value("Raw", VideoEncoder::Codec::Raw);
        codec.value("MPEG4", VideoEncoder::Codec::MPEG4);
        codec.value("MPEG2", VideoEncoder::Codec::MPEG2);
        codec.value("H264", VideoEncoder::Codec::H264);
        codec.value("HEVC", VideoEncoder::Codec::HEVC);
    }
}
