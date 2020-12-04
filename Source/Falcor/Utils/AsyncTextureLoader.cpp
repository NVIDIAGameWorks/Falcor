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
#include "stdafx.h"
#include "AsyncTextureLoader.h"

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

    std::future<Texture::SharedPtr> AsyncTextureLoader::loadFromFile(const std::string& filename, bool generateMipLevels, bool loadAsSrgb, Resource::BindFlags bindFlags)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mRequestQueue.push(Request{filename, generateMipLevels, loadAsSrgb, bindFlags});
        mCondition.notify_one();
        return mRequestQueue.back().promise.get_future();
    }

    void AsyncTextureLoader::runWorkers(size_t threadCount)
    {
        // Create a barrier to synchronize worker threads before issuing a global flush.
        auto barrier = std::make_shared<Barrier>(threadCount, [&] () {
            gpDevice->flushAndSync();
            mFlushPending = false;
            mUploadCounter = 0;
        });

        // Start worker threads.
        for (size_t i = 0; i < threadCount; ++i)
        {
            mThreads.emplace_back([&, barrier] () {
                while (true)
                {
                    // Wait on condition until more work is ready.
                    std::unique_lock<std::mutex> lock(mMutex);
                    mCondition.wait(lock, [&] () { return mTerminate || !mRequestQueue.empty() || mFlushPending; });

                    // Sync thread if a flush is pending.
                    if (mFlushPending)
                    {
                        lock.unlock();
                        barrier->wait();
                        mCondition.notify_one();
                        continue;
                    }

                    // Terminate thread unless there is more work to do.
                    if (mTerminate && mRequestQueue.empty() && !mFlushPending) break;

                    // Go back waiting if queue is currently empty.
                    if (mRequestQueue.empty()) continue;

                    // Pop next loading request from queue.
                    auto request = std::move(mRequestQueue.front());
                    mRequestQueue.pop();

                    lock.unlock();

                    // Load the textures (this part is running in parallel).
                    Texture::SharedPtr pTexture = Texture::createFromFile(request.filename, request.generateMipLevels, request.loadAsSrgb, request.bindFlags);
                    request.promise.set_value(pTexture);

                    lock.lock();

                    // Issue a global flush if necessary.
                    // TODO: It would be better to check the size of the upload heap instead.
                    if (!mTerminate && ++mUploadCounter >= kUploadsPerFlush)
                    {
                        mFlushPending = true;
                        mCondition.notify_all();
                    }

                    mCondition.notify_one();
                }
            });
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
