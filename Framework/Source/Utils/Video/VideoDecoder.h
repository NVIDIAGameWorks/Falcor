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
#pragma once
#include <string>
#include <future>
#include "API/Texture.h"

struct AVFormatContext;
struct AVStream;
struct AVFrame;
struct AVPicture;
struct SwsContext;
struct AVCodecContext;
struct AVCodec;
struct AVRational;

namespace Falcor
{        
    /** Simple video decoder for high-framerate and high-resolution
        playback of rendered videos. Currently decodes the first N frames
        as textures before playing.
    */
    class VideoDecoder
    {
    public:
        using UniquePtr = std::unique_ptr<VideoDecoder>;
        using UniqueConstPtr = std::unique_ptr<const VideoDecoder>;

        typedef std::vector<Texture::SharedPtr> TexturePool;
        typedef std::shared_ptr<TexturePool> TexturePoolPtr;

        /** Create a new VideoDecoder object
            \param[in] filename Input video file (with path)
            \param[in] bufferFrames The maximum number of input frames to buffer as Texture objects. Default is 300.
            \param[in] async Whether load operation should happen asynchronously
        */
        static UniquePtr create(const std::string& filename, uint32_t bufferedFrames = 300, bool async = false);
        ~VideoDecoder();

        /** Get a texture object for the frame at current time
            \param[in] curTime Time for which frame is sought
            \return Texture pointer to texture object
        */
        Texture::SharedPtr getTextureForNextFrame(float curTime);

        /** Return duration of video loaded (in seconds).
            This uses the actual number of frames. Only makes sense
            for videos shorter than the requested frame count.
            \return Duration of video loaded in seconds.
        */
        float getDuration();

        /** Returns reusable shared texture pool
        */
        TexturePoolPtr getTexturePool();

        /** Sets reusable shared texture pool
        */
        void setTexturePool(TexturePoolPtr& texturePool);

        /** Opens a file
            \param[in] filename Input video file (with path)
            \param[in] bufferFrames The maximum number of input frames to buffer as Texture objects. Default is 300.
            \param[in] async Whether load operation should happen asynchronously
        */
        bool load(const std::string& filename, uint32_t bufferedFrames, bool async);

    private:
        /** Holds a single video frame on CPU
        */
        struct Frame
        {
            Frame() {}
            Frame(AVCodecContext* codec);
            ~Frame();
            AVFrame*    mpFrameRGB = nullptr;
            uint8_t*    mCpuVidBuffer = nullptr;
            int         mFrameSize = 0;
        };

        VideoDecoder();

        void uploadToGPU(int frameStart = 0);

        std::string mFilename;

        AVFormatContext*                        mpFormatCtx       = nullptr;
        AVCodecContext*                         mpCodecCtxOrig    = nullptr;
        AVCodecContext*                         mpCodecCtx        = nullptr;
        AVCodec*                                mpCodec           = nullptr;
        AVFrame*                                mpFrame           = nullptr;
        std::vector<Frame>                      mFrames;

        float                                   mFPS              = 30;
        bool                                    mFlipY            = true;
        unsigned                                mVideoStream      = -1;
        uint32_t                                mVidBufferCount   = 300;
        uint32_t                                mRealFrameCount = 0;

        std::shared_ptr<std::future<void>>        mAsyncDecoding = nullptr;

        TexturePoolPtr                          mFrameTextures;

        // helper routines
        void  bufferFrames();
        float rationalToFloat(const AVRational& r);
    };
}