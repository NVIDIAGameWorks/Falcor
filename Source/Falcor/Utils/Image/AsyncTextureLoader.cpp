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
#include "AsyncTextureLoader.h"
#include "Core/API/Device.h"
#include "Utils/Threading.h"

namespace Falcor
{
    namespace
    {
        constexpr size_t kUploadsPerFlush = 16; ///< Number of texture uploads before issuing a flush (to keep upload heap from growing).
    }

    AsyncTextureLoader::AsyncTextureLoader(size_t threadCount)
    {
        runWorkers(threadCount);
    }

    AsyncTextureLoader::~AsyncTextureLoader()
    {
        terminateWorkers();

        gpDevice->flushAndSync();
    }

    std::future<Texture::SharedPtr> AsyncTextureLoader::loadFromFile(const std::filesystem::path& path, bool generateMipLevels, bool loadAsSrgb, Resource::BindFlags bindFlags, LoadCallback callback)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mLoadRequestQueue.push(LoadRequest{path, generateMipLevels, loadAsSrgb, bindFlags, callback });
        mCondition.notify_one();
        return mLoadRequestQueue.back().promise.get_future();
    }

    void AsyncTextureLoader::runWorkers(size_t threadCount)
    {
        // Create a barrier to synchronize worker threads before issuing a global flush.
        mFlushBarrier = std::make_shared<Barrier>(threadCount, [&]() {
            gpDevice->flushAndSync();
            mFlushPending = false;
            mUploadCounter = 0;
            });

        for (size_t i = 0; i < threadCount; ++i)
        {
            mThreads.emplace_back(&AsyncTextureLoader::runWorker, this);
        }
    }

    void AsyncTextureLoader::runWorker()
    {
        // This function is the entry point for worker threads.
        // The workers wait on the load request queue and load a texture when woken up.
        // To avoid the upload heap growing too large, we synchronize the threads and
        // issue a global GPU flush at regular intervals.

        while (true)
        {
            // Wait on condition until more work is ready.
            std::unique_lock<std::mutex> lock(mMutex);
            mCondition.wait(lock, [&]() { return mTerminate || !mLoadRequestQueue.empty() || mFlushPending; });

            // Sync thread if a flush is pending.
            if (mFlushPending)
            {
                lock.unlock();
                mFlushBarrier->wait();
                mCondition.notify_one();
                continue;
            }

            // Terminate thread unless there is more work to do.
            if (mTerminate && mLoadRequestQueue.empty() && !mFlushPending) break;

            // Go back waiting if queue is currently empty.
            if (mLoadRequestQueue.empty()) continue;

            // Pop next load request from queue.
            auto request = std::move(mLoadRequestQueue.front());
            mLoadRequestQueue.pop();

            lock.unlock();

            // Load the textures (this part is running in parallel).
            Texture::SharedPtr pTexture = Texture::createFromFile(request.path, request.generateMipLevels, request.loadAsSRGB, request.bindFlags);
            request.promise.set_value(pTexture);

            if (request.callback)
            {
                request.callback(pTexture);
            }

            lock.lock();

            // Issue a global flush if necessary.
            // TODO: It would be better to check the size of the upload heap instead.
            if (!mTerminate && pTexture != nullptr &&
                ++mUploadCounter >= kUploadsPerFlush)
            {
                mFlushPending = true;
                mCondition.notify_all();
            }

            mCondition.notify_one();
        }
    }

    void AsyncTextureLoader::terminateWorkers()
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mTerminate = true;
        }

        mCondition.notify_all();

        for (auto& thread : mThreads) thread.join();
    }
}
