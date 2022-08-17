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
#include "TextureManager.h"
#include "Core/API/Device.h"
#include "Utils/Logger.h"

// Temporarily disable asynchronous texture loader until Falcor supports parallel GPU work submission.
// Until then `TextureManager` should only called from the main thread.
#define DISABLE_ASYNC_TEXTURE_LOADER

namespace Falcor
{
    namespace
    {
        const size_t kMaxTextureHandleCount = std::numeric_limits<uint32_t>::max();
        static_assert(TextureManager::TextureHandle::kInvalidID >= kMaxTextureHandleCount);
    }

    TextureManager::SharedPtr TextureManager::create(size_t maxTextureCount, size_t threadCount)
    {
        return SharedPtr(new TextureManager(maxTextureCount, threadCount));
    }

    TextureManager::TextureManager(size_t maxTextureCount, size_t threadCount)
        : mMaxTextureCount(std::min(maxTextureCount, kMaxTextureHandleCount))
        , mAsyncTextureLoader(threadCount)
    {
    }

    TextureManager::~TextureManager()
    {
    }

    TextureManager::TextureHandle TextureManager::addTexture(const Texture::SharedPtr& pTexture)
    {
        FALCOR_ASSERT(pTexture);
        if (pTexture->getType() != Resource::Type::Texture2D || pTexture->getSampleCount() != 1)
        {
            throw ArgumentError("Only single-sample 2D textures can be added");
        }

        std::unique_lock<std::mutex> lock(mMutex);
        TextureHandle handle;

        if (auto it = mTextureToHandle.find(pTexture.get()); it != mTextureToHandle.end())
        {
            // Texture is already managed. Return its handle.
            handle = it->second;
        }
        else
        {
            // Texture is not already managed. Add new texture desc.
            TextureDesc desc = { TextureState::Loaded, pTexture };
            handle = addDesc(desc);

            // Add to texture-to-handle map.
            mTextureToHandle[pTexture.get()] = handle;

            // If texture was originally loaded from disk, add to key-to-handle map to avoid loading it again later if requested in loadTexture().
            // It's possible the user-provided texture has already been loaded by us. In that case, log a warning as the redundant load should be fixed.
            if (!pTexture->getSourcePath().empty())
            {
                bool hasMips = pTexture->getMipCount() > 1;
                bool isSrgb = isSrgbFormat(pTexture->getFormat());
                TextureKey textureKey(pTexture->getSourcePath().string(), hasMips, isSrgb, pTexture->getBindFlags());

                if (mKeyToHandle.find(textureKey) == mKeyToHandle.end())
                {
                    mKeyToHandle[textureKey] = handle;
                }
                else
                {
                    logWarning("TextureManager::addTexture() - Texture loaded from '{}' appears to be identical to an already loaded texture. This could be optimized by getting it from TextureManager.", pTexture->getSourcePath());
                }
            }
        }

        return handle;
    }

    TextureManager::TextureHandle TextureManager::loadTexture(const std::filesystem::path& path, bool generateMipLevels, bool loadAsSRGB, Resource::BindFlags bindFlags, bool async)
    {
        TextureHandle handle;

        // Find the full path to the texture.
        std::filesystem::path fullPath;
        if (!findFileInDataDirectories(path, fullPath))
        {
            logWarning("Can't find texture file '{}'.", path);
            return handle;
        }

        std::unique_lock<std::mutex> lock(mMutex);
        const TextureKey textureKey(fullPath, generateMipLevels, loadAsSRGB, bindFlags);

        if (auto it = mKeyToHandle.find(textureKey); it != mKeyToHandle.end())
        {
            // Texture is already managed. Return its handle.
            handle = it->second;
        }
        else
        {
#ifndef DISABLE_ASYNC_TEXTURE_LOADER
            mLoadRequestsInProgress++;

            // Texture is not already managed. Add new texture desc.
            TextureDesc desc = { TextureState::Referenced, nullptr };
            handle = addDesc(desc);

            // Add to key-to-handle map.
            mKeyToHandle[textureKey] = handle;

            // Function called by the async texture loader when loading finishes.
            // It's called by a worker thread so needs to acquire the mutex before changing any state.
            auto callback = [=](Texture::SharedPtr pTexture)
            {
                std::unique_lock<std::mutex> lock(mMutex);

                // Mark texture as loaded.
                auto& desc = getDesc(handle);
                desc.state = TextureState::Loaded;
                desc.pTexture = pTexture;

                // Add to texture-to-handle map.
                if (pTexture) mTextureToHandle[pTexture.get()] = handle;

                mLoadRequestsInProgress--;
                mCondition.notify_all();
            };

            // Issue load request to texture loader.
            mAsyncTextureLoader.loadFromFile(fullPath, generateMipLevels, loadAsSRGB, bindFlags, callback);
#else
            // Load texture from main thread.
            Texture::SharedPtr pTexture = Texture::createFromFile(fullPath, generateMipLevels, loadAsSRGB, bindFlags);

            // Add new texture desc.
            TextureDesc desc = { TextureState::Loaded, pTexture };
            handle = addDesc(desc);

            // Add to key-to-handle map.
            mKeyToHandle[textureKey] = handle;

            // Add to texture-to-handle map.
            if (pTexture) mTextureToHandle[pTexture.get()] = handle;

            mCondition.notify_all();
#endif
        }

        lock.unlock();

        if (!async)
        {
            waitForTextureLoading(handle);
        }

        return handle;
    }

    void TextureManager::waitForTextureLoading(const TextureHandle& handle)
    {
        if (!handle) return;

        // Acquire mutex and wait for texture state to change.
        std::unique_lock<std::mutex> lock(mMutex);
        mCondition.wait(lock, [&]() { return getDesc(handle).state == TextureState::Loaded; });

        gpDevice->flushAndSync();
    }

    void TextureManager::waitForAllTexturesLoading()
    {
        // Acquire mutex and wait for all in-progress requests to finish.
        std::unique_lock<std::mutex> lock(mMutex);
        mCondition.wait(lock, [&]() { return mLoadRequestsInProgress == 0; });

        gpDevice->flushAndSync();
    }

    void TextureManager::removeTexture(const TextureHandle& handle)
    {
        if (!handle) return;

        waitForTextureLoading(handle);

        std::lock_guard<std::mutex> lock(mMutex);

        // Get texture desc. If it's already cleared, we're done.
        auto& desc = getDesc(handle);
        if (!desc.isValid()) return;

        // Remove handle from maps.
        // Note not all handles exist in key-to-handle map so search for it. This can be optimized if needed.
        auto it = std::find_if(mKeyToHandle.begin(), mKeyToHandle.end(), [handle](const auto& keyVal) { return keyVal.second == handle; });
        if (it != mKeyToHandle.end()) mKeyToHandle.erase(it);

        if (desc.pTexture)
        {
            FALCOR_ASSERT(mTextureToHandle.find(desc.pTexture.get()) != mTextureToHandle.end());
            mTextureToHandle.erase(desc.pTexture.get());
        }

        // Clear texture desc.
        desc = {};

        // Return handle to the free list.
        mFreeList.push_back(handle);
    }

    TextureManager::TextureDesc TextureManager::getTextureDesc(const TextureHandle& handle) const
    {
        if (!handle) return {};

        std::lock_guard<std::mutex> lock(mMutex);
        FALCOR_ASSERT(handle && handle.id < mTextureDescs.size());
        return mTextureDescs[handle.id];
    }

    size_t TextureManager::getTextureDescCount() const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return mTextureDescs.size();
    }

    void TextureManager::setShaderData(const ShaderVar& var, const size_t descCount) const
    {
        std::lock_guard<std::mutex> lock(mMutex);

        if (mTextureDescs.size() > descCount)
        {
            throw RuntimeError("Descriptor array is too small");
        }

        Texture::SharedPtr nullTexture;
        for (size_t i = 0; i < mTextureDescs.size(); i++)
        {
            var[i] = mTextureDescs[i].pTexture;
        }
        for (size_t i = mTextureDescs.size(); i < descCount; i++)
        {
            var[i] = nullTexture;
        }
    }

    TextureManager::TextureHandle TextureManager::addDesc(const TextureDesc& desc)
    {
        TextureHandle handle;

        // Allocate new texture handle and insert desc.
        if (!mFreeList.empty())
        {
            handle = mFreeList.back();
            mFreeList.pop_back();
            getDesc(handle) = desc;
        }
        else
        {
            if (mTextureDescs.size() >= mMaxTextureCount)
            {
                throw RuntimeError("Out of texture handles");
            }
            handle = { static_cast<uint32_t>(mTextureDescs.size()) };
            mTextureDescs.emplace_back(desc);
        }

        return handle;
    }

    TextureManager::TextureDesc& TextureManager::getDesc(const TextureHandle& handle)
    {
        FALCOR_ASSERT(handle && handle.id < mTextureDescs.size());
        return mTextureDescs[handle.id];
    }
}
