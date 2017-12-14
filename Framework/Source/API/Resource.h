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
#include "ResourceViews.h"
#include <unordered_map>

namespace Falcor
{
    class CopyContext;

    class Resource : public std::enable_shared_from_this<Resource>
    {
    public:
        using ApiHandle = ResourceHandle;
        /** These flags are hints the driver to what pipeline stages the resource will be bound to.
        */
        enum class BindFlags : uint32_t
        {
            None = 0x0,             ///< The resource will not be bound the pipeline. Use this to create a staging resource
            Vertex = 0x1,           ///< The resource will be bound as a vertex-buffer
            Index = 0x2,            ///< The resource will be bound as a index-buffer
            Constant = 0x4,         ///< The resource will be bound as a constant-buffer
            StreamOutput = 0x8,     ///< The resource will be bound to the stream-output stage as an output buffer
            ShaderResource = 0x10,  ///< The resource will be bound as a shader-resource
            UnorderedAccess = 0x20, ///< The resource will be bound as an UAV
            RenderTarget = 0x40,    ///< The resource will be bound as a render-target
            DepthStencil = 0x80,    ///< The resource will be bound as a depth-stencil buffer
            IndirectArg = 0x100     ///< The resource will be bound as an indirect argument buffer
        };

        /** Resource types. Notice there are no array types. Array are controlled using the array size parameter on texture creation.
        */
        enum class Type
        {
            Buffer,                 ///< Buffer. Can be bound to all shader-stages
            Texture1D,              ///< 1D texture. Can be bound as render-target, shader-resource and UAV
            Texture2D,              ///< 2D texture. Can be bound as render-target, shader-resource and UAV
            Texture3D,              ///< 3D texture. Can be bound as render-target, shader-resource and UAV
            TextureCube,            ///< Texture-cube. Can be bound as render-target, shader-resource and UAV
            Texture2DMultisample,   ///< 2D multi-sampled texture. Can be bound as render-target, shader-resource and UAV
        };

        /** Resource state. Keeps track of how the resource was last used
        */
        enum class State : uint32_t
        {
            Undefined,
            PreInitialized,
            Common,
            VertexBuffer,
            ConstantBuffer,
            IndexBuffer,
            RenderTarget,
            UnorderedAccess,
            DepthStencil,
            ShaderResource,
            StreamOut,
            IndirectArg,
            CopyDest,
            CopySource,
            ResolveDest,
            ResolveSource,
            Present,
            GenericRead,
            Predication,
        };

        using SharedPtr = std::shared_ptr<Resource>;
        using SharedConstPtr = std::shared_ptr<const Resource>;

        /** Default value used in create*() methods
        */
        static const uint32_t kMaxPossible = RenderTargetView::kMaxPossible;

        virtual ~Resource() = 0;

        /** Get the bind flags
        */
        BindFlags getBindFlags() const { return mBindFlags; }

        /** Get the current state
        */
        State getState() const { return mState; }

        /** Get the resource type
        */
        Type getType() const { return mType; }

        /** Get the API handle
        */
        ApiHandle getApiHandle() const { return mApiHandle; }

        /** Get a shader-resource view.
            \param[in] firstArraySlice The first array slice of the view
            \param[in] arraySize The array size. If this is equal to Texture#kMaxPossible, will create a view ranging from firstArraySlice to the texture's array size
            \param[in] mostDetailedMip The most detailed mip level of the view
            \param[in] mipCount The number of mip-levels to bind. If this is equal to Texture#kMaxPossible, will create a view ranging from mostDetailedMip to the texture's mip levels count
        */
        ShaderResourceView::SharedPtr getSRV(uint32_t mostDetailedMip = 0, uint32_t mipCount = kMaxPossible, uint32_t firstArraySlice = 0, uint32_t arraySize = kMaxPossible) const;

        /** Get a render-target view.
            \param[in] mipLevel The requested mip-level
            \param[in] firstArraySlice The first array slice of the view
            \param[in] arraySize The array size. If this is equal to Texture#kMaxPossible, will create a view ranging from firstArraySlice to the texture's array size
        */
        RenderTargetView::SharedPtr getRTV(uint32_t mipLevel = 0, uint32_t firstArraySlice = 0, uint32_t arraySize = kMaxPossible) const;

        /** Get a depth stencil view.
            \param[in] mipLevel The requested mip-level
            \param[in] firstArraySlice The first array slice of the view
            \param[in] arraySize The array size. If this is equal to Texture#kMaxPossible, will create a view ranging from firstArraySlice to the texture's array size
        */
        DepthStencilView::SharedPtr getDSV(uint32_t mipLevel = 0, uint32_t firstArraySlice = 0, uint32_t arraySize = kMaxPossible) const;

        /** Get an unordered access view.
            \param[in] mipLevel The requested mip-level
            \param[in] firstArraySlice The first array slice of the view
            \param[in] arraySize The array size. If this is equal to Texture#kMaxPossible, will create a view ranging from firstArraySlice to the texture's array size
        */
        UnorderedAccessView::SharedPtr getUAV(uint32_t mipLevel = 0, uint32_t firstArraySlice = 0, uint32_t arraySize = kMaxPossible) const;

        struct ViewInfoHashFunc
        {
            std::size_t operator()(const ResourceViewInfo& v) const
            {
                return ((std::hash<uint32_t>()(v.firstArraySlice)
                    ^ (std::hash<uint32_t>()(v.arraySize) << 1)) >> 1)
                    ^ (std::hash<uint32_t>()(v.mipCount) << 1)
                    ^ (std::hash<uint32_t>()(v.mostDetailedMip) << 3);
            }
        };

        /** Invalidate and release all of the resource views
        */
        void invalidateViews() const;

    protected:
        friend class CopyContext;

        Resource(Type type, BindFlags bindFlags) : mType(type), mBindFlags(bindFlags) {}

        Type mType;
        BindFlags mBindFlags;
        mutable State mState = State::Undefined;
        ApiHandle mApiHandle;

        mutable std::unordered_map<ResourceViewInfo, ShaderResourceView::SharedPtr, ViewInfoHashFunc> mSrvs;
        mutable std::unordered_map<ResourceViewInfo, RenderTargetView::SharedPtr, ViewInfoHashFunc> mRtvs;
        mutable std::unordered_map<ResourceViewInfo, DepthStencilView::SharedPtr, ViewInfoHashFunc> mDsvs;
        mutable std::unordered_map<ResourceViewInfo, UnorderedAccessView::SharedPtr, ViewInfoHashFunc> mUavs;
    };

    enum_class_operators(Resource::BindFlags);

    const std::string to_string(Resource::Type);
    const std::string to_string(Resource::BindFlags);
    const std::string to_string(Resource::State);
}