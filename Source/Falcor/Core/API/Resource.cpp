/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "Resource.h"
#include "Texture.h"
#include "Device.h"
#include "Buffer.h"
#include "GFXAPI.h"
#include "NativeHandleTraits.h"
#include "Core/Error.h"
#include "Core/ObjectPython.h"
#include "Utils/Logger.h"
#include "Utils/StringUtils.h"
#include "Utils/Scripting/ScriptBindings.h"

namespace Falcor
{

Resource::Resource(ref<Device> pDevice, Type type, ResourceBindFlags bindFlags, uint64_t size)
    : mpDevice(pDevice), mType(type), mBindFlags(bindFlags), mSize(size)
{}

Resource::~Resource() = default;

const std::string to_string(Resource::Type type)
{
#define type_2_string(a)    \
    case Resource::Type::a: \
        return #a;
    switch (type)
    {
        type_2_string(Buffer);
        type_2_string(Texture1D);
        type_2_string(Texture2D);
        type_2_string(Texture3D);
        type_2_string(TextureCube);
        type_2_string(Texture2DMultisample);
    default:
        FALCOR_UNREACHABLE();
        return "";
    }
#undef type_2_string
}

const std::string to_string(Resource::State state)
{
    if (state == Resource::State::Common)
    {
        return "Common";
    }
    std::string s;
#define state_to_str(f_)              \
    if (state == Resource::State::f_) \
    {                                 \
        return #f_;                   \
    }

    state_to_str(Common);
    state_to_str(VertexBuffer);
    state_to_str(ConstantBuffer);
    state_to_str(IndexBuffer);
    state_to_str(RenderTarget);
    state_to_str(UnorderedAccess);
    state_to_str(DepthStencil);
    state_to_str(ShaderResource);
    state_to_str(StreamOut);
    state_to_str(IndirectArg);
    state_to_str(CopyDest);
    state_to_str(CopySource);
    state_to_str(ResolveDest);
    state_to_str(ResolveSource);
    state_to_str(Present);
    state_to_str(Predication);
    state_to_str(NonPixelShader);
    state_to_str(AccelerationStructure);
#undef state_to_str
    return s;
}

ref<Device> Resource::getDevice() const
{
    return mpDevice;
}

void Resource::invalidateViews() const
{
    auto invalidateAll = [](auto& vec)
    {
        for (const auto& item : vec)
            item.second->invalidate();
        vec.clear();
    };

    invalidateAll(mSrvs);
    invalidateAll(mUavs);
    invalidateAll(mRtvs);
    invalidateAll(mDsvs);
}

void Resource::setName(const std::string& name)
{
    mName = name;
    getGfxResource()->setDebugName(mName.c_str());
}

ref<Texture> Resource::asTexture()
{
    // In the past, Falcor relied on undefined behavior checking `this` for nullptr, returning nullptr if `this` was nullptr.
    FALCOR_ASSERT(this);
    return ref<Texture>(dynamic_cast<Texture*>(this));
}

ref<Buffer> Resource::asBuffer()
{
    // In the past, Falcor relied on undefined behavior checking `this` for nullptr, returning nullptr if `this` was nullptr.
    FALCOR_ASSERT(this);
    return ref<Buffer>(dynamic_cast<Buffer*>(this));
}

Resource::State Resource::getGlobalState() const
{
    if (mState.isGlobal == false)
    {
        logWarning(
            "Resource::getGlobalState() - the resource doesn't have a global state. The subresoruces are in a different state, use "
            "getSubResourceState() instead"
        );
        return State::Undefined;
    }
    return mState.global;
}

Resource::State Resource::getSubresourceState(uint32_t arraySlice, uint32_t mipLevel) const
{
    const Texture* pTexture = dynamic_cast<const Texture*>(this);
    if (pTexture)
    {
        uint32_t subResource = pTexture->getSubresourceIndex(arraySlice, mipLevel);
        return (mState.isGlobal) ? mState.global : mState.perSubresource[subResource];
    }
    else
    {
        logWarning(
            "Calling Resource::getSubresourceState() on an object that is not a texture. This call is invalid, use "
            "Resource::getGlobalState() instead"
        );
        FALCOR_ASSERT(mState.isGlobal);
        return mState.global;
    }
}

void Resource::setGlobalState(State newState) const
{
    mState.isGlobal = true;
    mState.global = newState;
}

void Resource::setSubresourceState(uint32_t arraySlice, uint32_t mipLevel, State newState) const
{
    const Texture* pTexture = dynamic_cast<const Texture*>(this);
    if (pTexture == nullptr)
    {
        logWarning("Calling Resource::setSubresourceState() on an object that is not a texture. This is invalid. Ignoring call");
        return;
    }

    // If we are transitioning from a global to local state, initialize the subresource array
    if (mState.isGlobal)
    {
        std::fill(mState.perSubresource.begin(), mState.perSubresource.end(), mState.global);
    }
    mState.isGlobal = false;
    mState.perSubresource[pTexture->getSubresourceIndex(arraySlice, mipLevel)] = newState;
}

SharedResourceApiHandle Resource::getSharedApiHandle() const
{
    gfx::InteropHandle handle = {};
    FALCOR_GFX_CALL(getGfxResource()->getSharedHandle(&handle));
    return (SharedResourceApiHandle)handle.handleValue;
}

NativeHandle Resource::getNativeHandle() const
{
    gfx::InteropHandle gfxNativeHandle = {};
    FALCOR_GFX_CALL(getGfxResource()->getNativeResourceHandle(&gfxNativeHandle));
#if FALCOR_HAS_D3D12
    if (mpDevice->getType() == Device::Type::D3D12)
        return NativeHandle(reinterpret_cast<ID3D12Resource*>(gfxNativeHandle.handleValue));
#endif
#if FALCOR_HAS_VULKAN
    if (mpDevice->getType() == Device::Type::Vulkan)
    {
        if (mType == Type::Buffer)
            return NativeHandle(reinterpret_cast<VkBuffer>(gfxNativeHandle.handleValue));
        else
            return NativeHandle(reinterpret_cast<VkImage>(gfxNativeHandle.handleValue));
    }
#endif
    return {};
}

void Resource::breakStrongReferenceToDevice()
{
    mpDevice.breakStrongReference();
}

FALCOR_SCRIPT_BINDING(Resource)
{
    pybind11::class_<Resource, ref<Resource>>(m, "Resource");
}
} // namespace Falcor
