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
#include "ResourceViews.h"
#include <unordered_map>

namespace Falcor
{
    class Texture;
    class Buffer;
    class ParameterBlock;

    class dlldecl Resource : public std::enable_shared_from_this<Resource>
    {
    public:
        using ApiHandle = ResourceHandle;
        using BindFlags = ResourceBindFlags;

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
            PixelShader,
            NonPixelShader,
#ifdef FALCOR_D3D12
            AccelerationStructure,
#endif
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

        bool isStateGlobal() const { return mState.isGlobal; }

        /** Get the current state. This is only valid if isStateGlobal() returns true
        */
        State getGlobalState() const;

        /** Get a subresource state
        */
        State getSubresourceState(uint32_t arraySlice, uint32_t mipLevel) const;

        /** Get the resource type
        */
        Type getType() const { return mType; }

        /** Get the API handle
        */
        const ApiHandle& getApiHandle() const { return mApiHandle; }

        /** Creates a shared resource API handle.
        */
        SharedResourceApiHandle createSharedApiHandle();

        struct ViewInfoHashFunc
        {
            std::size_t operator()(const ResourceViewInfo& v) const
            {
                return ((std::hash<uint32_t>()(v.firstArraySlice) ^ (std::hash<uint32_t>()(v.arraySize) << 1)) >> 1)
                    ^ (std::hash<uint32_t>()(v.mipCount) << 1)
                    ^ (std::hash<uint32_t>()(v.mostDetailedMip) << 3)
                    ^ (std::hash<uint32_t>()(v.firstElement) << 5)
                    ^ (std::hash<uint32_t>()(v.elementCount) << 7);
            }
        };

        /** Get the size of the resource
        */
        size_t getSize() const { return mSize; }

        /** Invalidate and release all of the resource views
        */
        void invalidateViews() const;

        /** Set the resource name
        */
        void setName(const std::string& name) { mName = name; apiSetName(); }

        /** Get the resource name
        */
        const std::string& getName() const { return mName; }

        /** Get a SRV/UAV for the entire resource.
            Buffer and Texture have overloads which allow you to create a view into part of the resource
        */
        virtual ShaderResourceView::SharedPtr getSRV() = 0;
        virtual UnorderedAccessView::SharedPtr getUAV() = 0;

        /** Conversions to derived classes
        */
        std::shared_ptr<Texture> asTexture() { return this ? std::dynamic_pointer_cast<Texture>(shared_from_this()) : nullptr; }
        std::shared_ptr<Buffer> asBuffer() { return this ? std::dynamic_pointer_cast<Buffer>(shared_from_this()) : nullptr; }

    protected:
        friend class CopyContext;

        Resource(Type type, BindFlags bindFlags, uint64_t size) : mType(type), mBindFlags(bindFlags), mSize(size) {}

        Type mType;
        BindFlags mBindFlags;
        struct
        {
            bool isGlobal = true;
            State global = State::Undefined;
            std::vector<State> perSubresource;
        } mutable mState;

        void setSubresourceState(uint32_t arraySlice, uint32_t mipLevel, State newState) const;
        void setGlobalState(State newState) const;
        void apiSetName();

        ApiHandle mApiHandle;
        size_t mSize = 0;
        GpuAddress mGpuVaOffset = 0;
        std::string mName;

        mutable std::unordered_map<ResourceViewInfo, ShaderResourceView::SharedPtr, ViewInfoHashFunc> mSrvs;
        mutable std::unordered_map<ResourceViewInfo, RenderTargetView::SharedPtr, ViewInfoHashFunc> mRtvs;
        mutable std::unordered_map<ResourceViewInfo, DepthStencilView::SharedPtr, ViewInfoHashFunc> mDsvs;
        mutable std::unordered_map<ResourceViewInfo, UnorderedAccessView::SharedPtr, ViewInfoHashFunc> mUavs;
    };

    const std::string dlldecl to_string(Resource::Type);
    const std::string dlldecl to_string(Resource::State);
}
