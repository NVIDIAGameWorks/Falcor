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
#include "VideoEncoder.h"
#include "Utils/BinaryFileStream.h"

extern "C"
{
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
}

namespace Falcor
{
    AVPixelFormat getPictureFormatFromCodec(AVCodecID codec)
    {
        switch(codec)
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
            should_not_get_here();
            return AV_PIX_FMT_NONE;
        }
    }

    AVPixelFormat getPictureFormatFromFalcorFormat(ResourceFormat format)
    {
        switch(format)
        {
        case ResourceFormat::RGBA8Unorm:
        case ResourceFormat::RGBA8UnormSrgb:
            return AV_PIX_FMT_RGBA;
        case ResourceFormat::BGRA8Unorm:
        case ResourceFormat::BGRA8UnormSrgb:
            return AV_PIX_FMT_BGRA;
        default:
            should_not_get_here();
            return AV_PIX_FMT_NONE;
        }
    }

    AVCodecID getCodecID(VideoEncoder::CodecID codec)
    {
        switch(codec)
        {
        case VideoEncoder::CodecID::RawVideo:
            return AV_CODEC_ID_RAWVIDEO;
        case VideoEncoder::CodecID::H264:
            return AV_CODEC_ID_H264;
        case VideoEncoder::CodecID::HEVC:
            return AV_CODEC_ID_HEVC;
        case VideoEncoder::CodecID::MPEG2:
            return AV_CODEC_ID_MPEG2VIDEO;
        case VideoEncoder::CodecID::MPEG4:
            return AV_CODEC_ID_MPEG4;
        default:
            should_not_get_here();
            return AV_CODEC_ID_NONE;
        }
    }

    static bool error(const std::string& filename, const std::string& msg)
    {
        std::string s("Error when creating video capture file ");
        s += filename + ".\n" + msg;
        logError(msg);
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
        pCodecCtx->time_base = {1, (int)fps};
        pCodecCtx->gop_size = gopSize;
        pCodecCtx->pix_fmt = getPictureFormatFromCodec(codecID);

        // Some formats want stream headers to be separate
        if(pCtx->oformat->flags & AVFMT_GLOBALHEADER)
        {
            pCodecCtx->flags |= CODEC_FLAG_GLOBAL_HEADER;
        }

        return pCodecCtx;
    }

    AVStream* createVideoStream(AVFormatContext* pCtx, uint32_t fps, AVCodecID codecID, const std::string& filename, AVCodec*& pCodec)
    {
        // Get the encoder
        pCodec = avcodec_find_encoder(codecID);
        if(pCodec == nullptr)
        {
            error(filename, std::string("Can't find ") + avcodec_get_name(codecID) + " encoder.");
            return nullptr;
        }

        // create the video stream
        AVStream* pStream = avformat_new_stream(pCtx, nullptr);
        if(pStream == nullptr)
        {
            error(filename, "Failed to create video stream.");
            return nullptr;
        }
        pStream->id = pCtx->nb_streams - 1;
        pStream->time_base = {1, (int)fps};
        return pStream;
    }

    AVFrame* allocateFrame(int format, uint32_t width, uint32_t height, const std::string& filename)
    {
        AVFrame* pFrame = av_frame_alloc();
        if(pFrame == nullptr)
        {
            error(filename, "Video frame allocation failed.");
            return nullptr;
        }

        pFrame->format = format;
        pFrame->width = width;
        pFrame->height = height;
        pFrame->pts = 0;

        // Allocate the buffer for the encoded image
        if(av_frame_get_buffer(pFrame, 32) < 0)
        {
            error(filename, "Can't allocate destination picture");
            return nullptr;
        }
     
        return pFrame;
    }

    bool openVideo(AVCodec* pCodec, AVCodecContext* pCodecCtx, AVFrame*& pFrame, const std::string& filename)
    {
        AVDictionary* param = nullptr;

        if(pCodecCtx->codec_id == AV_CODEC_ID_H264)
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
        if(avcodec_open2(pCodecCtx, pCodec, &param) < 0)
        {
            return error(filename, "Can't open video codec.");
        }
        av_dict_free(&param);

        // create a frame
        pFrame = allocateFrame(pCodecCtx->pix_fmt, pCodecCtx->width, pCodecCtx->height, filename);
        if(pFrame == nullptr)
        {
            return false;
        }
        return true;
    }

    VideoEncoder::VideoEncoder(const std::string& filename) : mFilename(filename)
    {
    }

    VideoEncoder::~VideoEncoder()
    {
        endCapture();
    }

    VideoEncoder::UniquePtr VideoEncoder::create(const Desc& desc)
    {
        UniquePtr pVC = UniquePtr(new VideoEncoder(desc.filename));
        if(pVC == nullptr)
        {
            error(desc.filename, "Failed to create CVideoCapture object");
            return nullptr;
        }

        if(pVC->init(desc) == false)
        {
            pVC = nullptr;
        }
        return pVC;
    }

    bool VideoEncoder::init(const Desc& desc)
    {
        // Register the codecs
        av_register_all();

        // create the output context
        avformat_alloc_output_context2(&mpOutputContext, nullptr, nullptr, mFilename.c_str());
        if(mpOutputContext == nullptr)
        {
            // The sample tries again, while explicitly requesting mpeg format. I chose not to do it, since it might lead to a container with a wrong extension
            return error(mFilename, "File output format not recognized. Make sure you use a known file extension (avi/mpeg/mp4)");
        }

        // Get the output format of the container
        AVOutputFormat* pOutputFormat = mpOutputContext->oformat;
        assert((pOutputFormat->flags & AVFMT_NOFILE) == 0); // Problem. We want a file.

        // create the video codec
        AVCodec* pVideoCodec;
        mpOutputStream = createVideoStream(mpOutputContext, desc.fps, getCodecID(desc.codec), mFilename, pVideoCodec);
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
        if(openVideo(pVideoCodec, mpCodecContext, mpFrame, mFilename) == false)
        {
            return false;
        }

        // copy the stream parameters to the muxer
        if(avcodec_parameters_from_context(mpOutputStream->codecpar, mpCodecContext) < 0)
        {
            return error(desc.filename, "Could not copy the stream parameters\n");
        }

        av_dump_format(mpOutputContext, 0, mFilename.c_str(), 1);

        // Open the output file
        assert((pOutputFormat->flags & AVFMT_NOFILE) == 0); // No output file required. Not sure if/when this happens.
        if(avio_open(&mpOutputContext->pb, mFilename.c_str(), AVIO_FLAG_WRITE) < 0)
        {
            return error(mFilename, "Can't open output file.");
        }

        // Write the stream header
        if(avformat_write_header(mpOutputContext, nullptr) < 0)
        {
            return error(mFilename, "Can't write file header.");
        }

        mFormat = desc.format;
        mRowPitch = getFormatBytesPerBlock(desc.format) * desc.width;
        if(desc.flipY)
        {
            mpFlippedImage = new uint8_t[desc.height * mRowPitch];
        }

        mpSwsContext = sws_getContext(desc.width, desc.height, getPictureFormatFromFalcorFormat(desc.format), desc.width, desc.height, mpCodecContext->pix_fmt, SWS_POINT, nullptr, nullptr, nullptr);
        if(mpSwsContext == nullptr)
        {
            return error(mFilename, "Failed to allocate SWScale context");
        }
        return true;
    }

    bool flush(AVCodecContext* pCodecContext, AVFormatContext* pOutputContext, AVStream* pOutputStream, const std::string& filename)
    {
        while(true)
        {
            // Initialize the packet
            AVPacket packet = {0};
            av_init_packet(&packet);

            int r = avcodec_receive_packet(pCodecContext, &packet);
            if(r == AVERROR(EAGAIN) || r == AVERROR_EOF)
            {
                return true;
            }
            else if(r < 0)
            {
                error(filename, "Can't retrieve packet");
                return false;
            }

            // rescale output packet timestamp values from codec to stream timebase
            av_packet_rescale_ts(&packet, pCodecContext->time_base, pOutputStream->time_base);
            packet.stream_index = pOutputStream->index;
            r = av_interleaved_write_frame(pOutputContext, &packet);
            if(r < 0)
            {
                char msg[1024];
                av_make_error_string(msg, 1024, r);
                error(filename, "Failed when writing encoded frame to file");
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
            flush(mpCodecContext, mpOutputContext, mpOutputStream, mFilename);

            av_write_trailer(mpOutputContext);

            avio_closep(&mpOutputContext->pb);
            avcodec_free_context(&mpCodecContext);
            av_frame_free(&mpFrame);
            sws_freeContext(mpSwsContext);
            avformat_free_context(mpOutputContext);
            mpOutputContext = nullptr;
            mpOutputStream = nullptr;
        }
        safe_delete(mpFlippedImage)
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
            if(flush(mpCodecContext, mpOutputContext, mpOutputStream, mFilename) == false)
            {
                return;
            }
        }
        else if(r < 0)
        {
            error(mFilename, "Can't send video frame");
            return;
        }
    }

    const std::string VideoEncoder::getSupportedContainerForCodec(CodecID codec)
    {
        const std::string AVI = std::string("AVI (Audio Video Interleaved)") + '\0' + "*.avi" + '\0';
        const std::string MP4 = std::string("MP4 (MPEG-4 Part 14)") + '\0' + "*.mp4" + '\0';
        const std::string MKV = std::string("MKV (Matroska)\0*.mkv") + '\0' + "*.mkv" + '\0';

        std::string s;
        switch(codec)
        {
        case VideoEncoder::CodecID::RawVideo:
            s += AVI;
            break;
        case VideoEncoder::CodecID::H264:
        case VideoEncoder::CodecID::MPEG2:
        case VideoEncoder::CodecID::MPEG4:
            s += MP4 + MKV + AVI;
            break;
        case VideoEncoder::CodecID::HEVC:
            s += MP4 + MKV;
            break;
        default:
            should_not_get_here();
        }

        s += "\0";
        return s;
    }
}