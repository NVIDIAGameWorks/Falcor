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
#include "AsyncTextureLoader.h"
#include "Core/Macros.h"
#include "Core/API/Resource.h"
#include "Core/API/Texture.h"
#include "Core/Program/ShaderVar.h"
#include <condition_variable>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

namespace Falcor
{
    /** Multi-threaded texture manager.

        This class manages a collection of textures and implements
        asynchronous texture loading. All operations are thread-safe.

        Each managed texture is assigned a unique handle upon loading.
        This handle is used in shader code to reference the given texture
        in the array of GPU texture descriptors.
    */
    class FALCOR_API TextureManager
    {
    public:
        using SharedPtr = std::shared_ptr<TextureManager>;

        ~TextureManager();

        /** State of a managed texture.
        */
        enum class TextureState
        {
            Invalid,        ///< Invalid/unknown texture.
            Referenced,     ///< Texture is referenced, but not yet loaded.
            Loaded,         ///< Texture has finished loading.
        };

        /** Handle to a managed texture.
        */
        struct TextureHandle
        {
            uint32_t id = kInvalidID;
            static const uint32_t kInvalidID = std::numeric_limits<uint32_t>::max();

            uint32_t getID() const { return id; }
            bool isValid() const { return id != kInvalidID; }
            explicit operator bool() const { return isValid(); }
            bool operator==(const TextureHandle& other) const { return id == other.id; }
        };

        /** Struct describing a managed texture.
        */
        struct TextureDesc
        {
            TextureState state = TextureState::Invalid;     ///< Current state of the texture.
            Texture::SharedPtr pTexture;                    ///< Valid texture object when state is 'Loaded', or nullptr if loading failed.

            bool isValid() const { return state != TextureState::Invalid; }
        };

        /** Create a texture manager.
            \param[in] maxTextureCount Maximum number of textures that can be simultaneously managed.
            \param[in] threadCount Number of worker threads.
            \return A new object.
        */
        static SharedPtr create(size_t maxTextureCount, size_t threadCount = std::thread::hardware_concurrency());

        /** Add a texture to the manager.
            If the texture is already managed, its existing handle is returned.
            \param[in] pTexture The texture resource.
            \return Unique handle to the texture.
        */
        TextureHandle addTexture(const Texture::SharedPtr& pTexture);

        /** Requst loading a texture from file.
            This will add the texture to the set of managed textures. The function returns a handle immediately.
            If asynchronous loading is requested, the texture data will not be available until loading completes.
            The returned handle is valid for the entire lifetime of the texture, until removeTexture() is called.
            \param[in] path File path of the texture. This can be a full path or a relative path from a data directory.
            \param[in] generateMipLevels Whether the full mip-chain should be generated.
            \param[in] loadAsSRGB Load the texture as sRGB format if supported, otherwise linear color.
            \param[in] bindFlags The bind flags for the texture resource.
            \param[in] async Load asynchronously, otherwise the function blocks until the texture data is loaded.
            \return Unique handle to the texture, or an invalid handle if the texture can't be found.
        */
        TextureHandle loadTexture(const std::filesystem::path& path, bool generateMipLevels, bool loadAsSRGB, Resource::BindFlags bindFlags = Resource::BindFlags::ShaderResource, bool async = true);

        /** Wait for a requested texture to load.
            If the handle is valid, the call blocks until the texture is loaded (or failed to load).
            \param[in] handle Texture handle.
        */
        void waitForTextureLoading(const TextureHandle& handle);

        /** Waits for all currently requested textures to be loaded.
        */
        void waitForAllTexturesLoading();

        /** Remove a texture.
            \param[in] handle Texture handle.
        */
        void removeTexture(const TextureHandle& handle);

        /** Get a loaded texture. Call getTextureDesc() for more info.
            \param[in] handle Texture handle.
            \return Texture if loaded, or nullptr if handle doesn't exist or texture isn't yet loaded.
        */
        Texture::SharedPtr getTexture(const TextureHandle& handle) const { return getTextureDesc(handle).pTexture; }

        /** Get a texture desc.
            \param[in] handle Texture handle.
            \return Texture desc, or invalid desc if handle is invalid.
        */
        TextureDesc getTextureDesc(const TextureHandle& handle) const;

        /** Get texture desc count.
            \return Number of texture descs.
        */
        size_t getTextureDescCount() const;

        /** Bind all textures into a shader var.
            The shader var should refer to a Texture2D descriptor array of fixed size.
            The array must be large enough, otherwise an exception is thrown.
            This restriction will go away when unbounded descriptor arrays are supported (see #1321).
            \param[in] var Shader var for descriptor array.
            \param[in] descCount Size of descriptor array.
        */
        void setShaderData(const ShaderVar& var, const size_t descCount) const;

    private:
        TextureManager(size_t maxTextureCount, size_t threadCount);

        /** Key to uniquely identify a managed texture.
        */
        struct TextureKey
        {
            std::filesystem::path fullPath;
            bool generateMipLevels;
            bool loadAsSRGB;
            Resource::BindFlags bindFlags;

            TextureKey(const std::filesystem::path& path, bool mips, bool srgb, Resource::BindFlags flags)
                : fullPath(path), generateMipLevels(mips), loadAsSRGB(srgb), bindFlags(flags)
            {}

            bool operator<(const TextureKey& rhs) const
            {
                if (fullPath != rhs.fullPath) return fullPath < rhs.fullPath;
                else if (generateMipLevels != rhs.generateMipLevels) return generateMipLevels < rhs.generateMipLevels;
                else if (loadAsSRGB != rhs.loadAsSRGB) return loadAsSRGB < rhs.loadAsSRGB;
                else return bindFlags < rhs.bindFlags;
            }
        };

        TextureHandle addDesc(const TextureDesc& desc);
        TextureDesc& getDesc(const TextureHandle& handle);

        mutable std::mutex mMutex;                                  ///< Mutex for synchronizing access to shared resources.
        std::condition_variable mCondition;                         ///< Condition variable to wait on for loading to finish.

        // Internal state. Do not access outside of critical section.
        std::vector<TextureDesc> mTextureDescs;                     ///< Array of all texture descs, indexed by handle ID.
        std::vector<TextureHandle> mFreeList;                       ///< List of unused handles.
        std::map<TextureKey, TextureHandle> mKeyToHandle;           ///< Map from texture key to handle.
        std::map<const Texture*, TextureHandle> mTextureToHandle;   ///< Map from texture ptr to handle.

        AsyncTextureLoader mAsyncTextureLoader;                     ///< Utility for asynchronous texture loading.
        size_t mLoadRequestsInProgress = 0;                         ///< Number of load requests currently in progress.

        const size_t mMaxTextureCount;                              ///< Maximum number of textures that can be simultaneously managed.
    };
}
