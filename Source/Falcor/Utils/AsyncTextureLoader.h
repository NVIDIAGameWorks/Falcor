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
#include <future>
#include "Falcor.h"

namespace Falcor
{
    /** Utility class to load textures asynchronously using multiple worker threads.
    */
    class dlldecl AsyncTextureLoader
    {
    public:
        /** Constructor.
            \param[in] threadCount Number of worker threads.
        */
        AsyncTextureLoader(size_t threadCount = std::thread::hardware_concurrency());

        /** Destructor.
            Blocks until all textures are loaded.
        */
        ~AsyncTextureLoader();

        /** Request loading a texture.
            \param[in] filename Filename of the image. Can also include a full path or relative path from a data directory.
            \param[in] generateMipLevels Whether the mip-chain should be generated.
            \param[in] loadAsSrgb Load the texture using sRGB format. Only valid for 3 or 4 component textures.
            \param[in] bindFlags The bind flags to create the texture with.
            \return A future to a new texture, or nullptr if the texture failed to load.
        */
        std::future<Texture::SharedPtr> loadFromFile(const std::string& filename, bool generateMipLevels, bool loadAsSrgb, Resource::BindFlags bindFlags = Resource::BindFlags::ShaderResource);

    private:
        void runWorkers(size_t threadCount);
        void terminateWorkers();

        struct Request
        {
            std::string filename;
            bool generateMipLevels;
            bool loadAsSrgb;
            Resource::BindFlags bindFlags;
            std::promise<Texture::SharedPtr> promise;
        };

        std::queue<Request> mRequestQueue;      ///< Texture loading request queue.
        std::condition_variable mCondition;     ///< Condition variable for workers to wait on.
        std::mutex mMutex;                      ///< Mutex for synchronizing access to shared resources.
        std::vector<std::thread> mThreads;      ///< Worker threads.
        bool mTerminate = false;                ///< Flag to terminate worker threads.
        bool mFlushPending = false;             ///< Flag to indicate a flush is pending.
        uint32_t mUploadCounter = 0;            ///< Counter to issue a flush every few uploads.
    };
}
