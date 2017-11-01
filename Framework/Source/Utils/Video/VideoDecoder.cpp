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
#if 0
#include "Framework.h"
#include "VideoDecoder.h"
#include "Utils/Platform/OS.h"
#include "Utils/BinaryFileStream.h"
extern "C"
{
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
}

#include <cstdio>

namespace Falcor
{
    float VideoDecoder::rationalToFloat(const AVRational& r)
    {
        return ((float)r.num / (float)r.den);
    }

    VideoDecoder::UniquePtr VideoDecoder::create(const std::string& filename, uint32_t bufferedFrames, bool async)
    {
        auto pVideo = UniquePtr(new VideoDecoder());
        if(pVideo->load(filename, bufferedFrames, async) == false)
        {
            pVideo = nullptr;
        }

        return pVideo;
    }

    VideoDecoder::VideoDecoder()
    {
    }

    bool VideoDecoder::load(const std::string& filename, uint32_t bufferedFrames, bool async)
    {
        mFilename = filename;
        mVidBufferCount = bufferedFrames;

        if(async)
        {
            mAsyncDecoding = std::make_shared<std::future<void>>(std::async(std::launch::async, &VideoDecoder::bufferFrames, this));
        }
        else
            bufferFrames();

        return true;
    }

    VideoDecoder::~VideoDecoder()
    {
        // Flush the async operation
        if(mAsyncDecoding)
        {
            mAsyncDecoding->get();
        }

        av_free(mpFrame);

        // Close the codec
        avcodec_close(mpCodecCtx);
        // Close the video file
        avformat_close_input(&mpFormatCtx);

        for(auto& tex : (*mFrameTextures))
            if(tex) tex->evict(nullptr);
    }

    void FlipRGBFrame(AVFrame* pFrame, int H)
    {
        uint8_t* templine = new uint8_t[pFrame->linesize[0]];

        int hby2 = H / 2;

        for(int line = 0; line < hby2; line++)
        {
            auto* line1 = pFrame->data[0] + (line)* pFrame->linesize[0];
            auto* line2 = pFrame->data[0] + (H - 1 - line) * pFrame->linesize[0];

            memcpy(templine, line1, pFrame->linesize[0]);
            memcpy(line1, line2, pFrame->linesize[0]);
            memcpy(line2, templine, pFrame->linesize[0]);
        }

        delete[] templine;
    }

    void VideoDecoder::bufferFrames()
    {
        if(mAsyncDecoding)
        {
            setThreadPriority(getCurrentThread(), ThreadPriorityType::Low);
            setThreadAffinity(getCurrentThread(), (1<<5)|(1<<6)|(1<<7)|(1<<8));
        }

        if(mpFrame)
        {
            av_free(mpFrame);
            mpFrame = nullptr;
        }

        // Close the codec
        if(mpCodecCtx)
        {
            avcodec_close(mpCodecCtx);
            mpCodecCtx = nullptr;
        }
        // Close the video file
        if(mpFormatCtx)
        {
            avformat_close_input(&mpFormatCtx);
            mpFormatCtx = nullptr;
        }

        // Register the codecs
        avcodec_register_all();
        av_register_all();

        if(avformat_open_input(&mpFormatCtx, mFilename.c_str(), NULL, NULL) != 0)
        {
            printf("Cannot open file\n");
            return;
        }

        if(avformat_find_stream_info(mpFormatCtx, NULL) < 0)
        {
            printf("Couldn't find stream information.\n");
            return;
        }

        for(uint32_t i = 0; i < mpFormatCtx->nb_streams; i++)
        {
            auto& stream = mpFormatCtx->streams[i];
            if(stream->codec->codec_type == AVMEDIA_TYPE_VIDEO)
            {
                mVideoStream = i;
                break;
            }
        }

        if(mVideoStream == -1)
        {
            return;
        }

        auto& stream = mpFormatCtx->streams[mVideoStream];

        // Get a pointer to the codec context for the video stream
        mpCodecCtxOrig = stream->codec;

        mFPS = rationalToFloat(stream->avg_frame_rate);

        // Find the decoder for the video stream
        mpCodec = avcodec_find_decoder(mpCodecCtxOrig->codec_id);

        if(mpCodec == NULL)
        {
            printf("Unsupported codec!\n");
            return; // Codec not found
        }

        // Copy context
        mpCodecCtx = avcodec_alloc_context3(mpCodec);
        if(avcodec_copy_context(mpCodecCtx, mpCodecCtxOrig) != 0)
        {
            printf("Couldn't copy codec context\n");
            return; // Error copying codec context
        }

        // Open codec
        if(avcodec_open2(mpCodecCtx, mpCodec, nullptr) < 0)
        {
            return; // Could not open codec
        }

        // Allocate video frame
        mpFrame = av_frame_alloc();


        mFrames.clear();
        mFrames.reserve(mVidBufferCount);

        const bool async = mAsyncDecoding != nullptr;

        //printf("Buffering frames...\n");
        uint32_t frameIdx = 0;

        SwsContext *sws_ctx = sws_getContext(
            mpCodecCtx->width, mpCodecCtx->height, mpCodecCtx->pix_fmt,
            mpCodecCtx->width, mpCodecCtx->height, PIX_FMT_RGBA, SWS_BILINEAR, NULL, NULL, NULL);

        AVPacket packet;
        while(av_read_frame(mpFormatCtx, &packet) >= 0 && (frameIdx < mVidBufferCount))
        {
            if(packet.stream_index == mVideoStream)
            {
                int32_t isFrameDone;
                avcodec_decode_video2(mpCodecCtx, mpFrame, &isFrameDone, &packet);
                if(isFrameDone)
                {
                    // Create either a single frame, or all frames one-by-one on CPU
                    if(async || mFrames.empty())
                    {
                        // Inplace construction to avoid releasing of arrays
                        mFrames.push_back(Frame());
                        new(&mFrames.back()) Frame(mpCodecCtx);
                    }
                    Frame& frame = mFrames.back();

                    // Convert the image from its native format to RGB
                    sws_scale(sws_ctx, (uint8_t const * const *)mpFrame->data, mpFrame->linesize, 0, mpCodecCtx->height, frame.mpFrameRGB->data, frame.mpFrameRGB->linesize);
                    FlipRGBFrame(frame.mpFrameRGB, mpCodecCtx->height);
                    if(!async)
                        uploadToGPU(frameIdx);

                    frameIdx++;
                }
            }
            av_free_packet(&packet);
        }
        //printf("Done reading %d frames\n", frameIdx);

        mRealFrameCount = frameIdx;
    }

    void VideoDecoder::uploadToGPU(int frameStart)
    {
        if(!mFrameTextures)
            mFrameTextures = std::make_shared<TexturePool>();
        for(size_t i=0;i<mFrames.size();++i)
        {
            auto& frame = mFrames[i];
            if(frameStart + i >= mFrameTextures->size())
            {
                mFrameTextures->push_back(Texture::create2D(mpCodecCtx->width, mpCodecCtx->height, ResourceFormat::RGBA8UnormSrgb, 1, 1, frame.mpFrameRGB->data[0]));
                mFrameTextures->back()->makeResident(nullptr);
            }
            else
            {
                (*mFrameTextures)[frameStart + i]->uploadSubresourceData(frame.mpFrameRGB->data[0], frame.mFrameSize);
            }
        }
        mFrames.clear();
    }

    Texture::SharedPtr VideoDecoder::getTextureForNextFrame(float curTime)
    {
        // Flush the async operation, upload everything to video memory
        if(mAsyncDecoding)
        {
            mAsyncDecoding->get();
            mAsyncDecoding.reset();
            uploadToGPU();
        }
        int curFrame = ((int)floor(curTime * mFPS)) % mRealFrameCount;
        //SetBufferToTexture(m_VidBuffers[curFrame]);
        //printf("time, frame: %f, %d\n", curTime, curFrame);
        return (*mFrameTextures)[curFrame];
    }

    float VideoDecoder::getDuration()
    {
        // Flush the async operation, upload everything to video memory
        if(mAsyncDecoding)
        {
            mAsyncDecoding->get();
            mAsyncDecoding.reset();
            uploadToGPU();
        }

        return ((float)mRealFrameCount) / mFPS;
    }

    VideoDecoder::TexturePoolPtr VideoDecoder::getTexturePool()
    {
        return mFrameTextures;
    }

    void VideoDecoder::setTexturePool(TexturePoolPtr& texturePool)
    {
        mFrameTextures = texturePool;
    }

    VideoDecoder::Frame::Frame(AVCodecContext* codec)
    {
        mFrameSize = avpicture_get_size(PIX_FMT_RGBA, codec->width, codec->height);

        // Allocate an AVFrame structure
        mpFrameRGB = av_frame_alloc();
        if(mpFrameRGB == NULL)
        {
            printf("Cannot allocate frame. Possibly out of CPU memory!\n");
            return;
        }
        mCpuVidBuffer = (uint8_t*)av_malloc(mFrameSize*sizeof(uint8_t));

        avpicture_fill((AVPicture *)mpFrameRGB, mCpuVidBuffer, PIX_FMT_RGBA, codec->width, codec->height);
        //printf("frame size is %d Mbytes\n", mFrameSize / (1024 * 1024));
    }

    VideoDecoder::Frame::~Frame()
    {
        if(mCpuVidBuffer)
            av_free(mCpuVidBuffer);
        if(mpFrameRGB)
            av_free(mpFrameRGB);
    }
}
#endif