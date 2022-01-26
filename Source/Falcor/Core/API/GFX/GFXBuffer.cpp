/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include "Core/API/Buffer.h"
#include "Core/API/Resource.h"
#include "GFXResource.h"

#define GFX_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT ( 256 )
#define GFX_TEXTURE_DATA_PLACEMENT_ALIGNMENT ( 512 )

namespace Falcor
{
    Slang::ComPtr<gfx::IBufferResource> createBuffer(Buffer::State initState, size_t size, Buffer::BindFlags bindFlags, Buffer::CpuAccess cpuAccess)
    {
        FALCOR_ASSERT(gpDevice);
        Slang::ComPtr<gfx::IDevice> pDevice = gpDevice->getApiHandle();

        // Create the buffer
        gfx::IBufferResource::Desc bufDesc = {};
        // Pipe through element size and format?
        bufDesc.sizeInBytes = size;
        switch (cpuAccess)
        {
        case Buffer::CpuAccess::None:
            bufDesc.memoryType = gfx::MemoryType::DeviceLocal;
            break;
        case Buffer::CpuAccess::Read:
            bufDesc.memoryType = gfx::MemoryType::ReadBack;
            break;
        case Buffer::CpuAccess::Write:
            bufDesc.memoryType = gfx::MemoryType::Upload;
            break;
        default:
            FALCOR_UNREACHABLE();
            break;
        }

        getGFXResourceState(bindFlags, bufDesc.defaultState, bufDesc.allowedStates);

        Slang::ComPtr<gfx::IBufferResource> pApiHandle;
        pDevice->createBufferResource(bufDesc, nullptr, pApiHandle.writeRef());
        FALCOR_ASSERT(pApiHandle);

        return pApiHandle;
    }

    size_t getBufferDataAlignment(const Buffer* pBuffer)
    {
        // This in order of the alignment size
        const auto& bindFlags = pBuffer->getBindFlags();
        if (is_set(bindFlags, Buffer::BindFlags::Constant)) return GFX_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
        if (is_set(bindFlags, Buffer::BindFlags::Index)) return sizeof(uint32_t); // This actually depends on the size of the index, but we can handle losing 2 bytes

        return GFX_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
    }


    void Buffer::apiInit(bool hasInitData)
    {
        if (mCpuAccess != CpuAccess::None && is_set(mBindFlags, BindFlags::Shared))
        {
            throw std::exception("Can't create shared resource with CPU access other than 'None'.");
        }

        if (mBindFlags == BindFlags::Constant)
        {
            mSize = align_to((size_t)GFX_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, mSize);
        }

        if (mCpuAccess == CpuAccess::Write)
        {
            mState.global = Resource::State::GenericRead;
            if (hasInitData == false) // Else the allocation will happen when updating the data
            {
                FALCOR_ASSERT(gpDevice);
                mDynamicData = gpDevice->getUploadHeap()->allocate(mSize, getBufferDataAlignment(this));
                mApiHandle = mDynamicData.pResourceHandle;
                mGpuVaOffset = mDynamicData.offset;
            }
        }
        else if (mCpuAccess == CpuAccess::Read && mBindFlags == BindFlags::None)
        {
            mState.global = Resource::State::CopyDest;
            mApiHandle = createBuffer(mState.global, mSize, mBindFlags, mCpuAccess);
        }
        else
        {
            mState.global = Resource::State::Common;
            if (is_set(mBindFlags, BindFlags::AccelerationStructure)) mState.global = Resource::State::AccelerationStructure;
            mApiHandle = createBuffer(mState.global, mSize, mBindFlags, mCpuAccess);
        }
    }

    void* mapBufferApi(const Buffer::ApiHandle& apiHandle, size_t size)
    {
        void* pData = nullptr;
        static_cast<gfx::IBufferResource*>(apiHandle.get())->map(nullptr, &pData);
        return pData;
    }

    uint64_t Buffer::getGpuAddress() const
    {
        gfx::IBufferResource* bufHandle = static_cast<gfx::IBufferResource*>(mApiHandle.get());
        FALCOR_ASSERT(bufHandle);
        // slang-gfx backend does not includ the mGpuVaOffset.
        return mGpuVaOffset + bufHandle->getDeviceAddress();
    }

    void Buffer::unmap()
    {
        // Only unmap read buffers, write buffers are persistently mapped.
        if (mpStagingResource)
        {
            static_cast<gfx::IBufferResource*>(mpStagingResource->mApiHandle.get())->unmap(nullptr);
        }
        else if (mCpuAccess == CpuAccess::Read)
        {
            static_cast<gfx::IBufferResource*>(mApiHandle.get())->unmap(nullptr);
        }
    }

#if FALCOR_ENABLE_CUDA
    void* Buffer::getCUDADeviceAddress() const
    {
        throw RuntimeError("Texture::getCUDADeviceAddress() - unimplemented");
        return nullptr;
    }

    void* Buffer::getCUDADeviceAddress(ResourceViewInfo const& viewInfo) const
    {
        throw RuntimeError("Texture::getCUDADeviceAddress() - unimplemented");
        return nullptr;
    }
#endif
}
