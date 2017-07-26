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
#include "Framework.h"
#include "API/Buffer.h"

namespace Falcor
{
    uint32_t GetBindFlags(Buffer::BindFlags flags)
    {
        uint32_t d3dFlags = 0;
        if((flags & Buffer::BindFlags::Vertex)          != Buffer::BindFlags::None) d3dFlags |= D3D11_BIND_VERTEX_BUFFER;
        if((flags & Buffer::BindFlags::Index)           != Buffer::BindFlags::None) d3dFlags |= D3D11_BIND_INDEX_BUFFER;
        if((flags & Buffer::BindFlags::Constant)        != Buffer::BindFlags::None) d3dFlags |= D3D11_BIND_CONSTANT_BUFFER;
        if((flags & Buffer::BindFlags::ShaderResource)  != Buffer::BindFlags::None) d3dFlags |= D3D11_BIND_SHADER_RESOURCE;
        if((flags & Buffer::BindFlags::StreamOutput)    != Buffer::BindFlags::None) d3dFlags |= D3D11_BIND_STREAM_OUTPUT;
        if((flags & Buffer::BindFlags::RenderTarget)    != Buffer::BindFlags::None) d3dFlags |= D3D11_BIND_RENDER_TARGET;
        if((flags & Buffer::BindFlags::DepthStencil)    != Buffer::BindFlags::None) d3dFlags |= D3D11_BIND_DEPTH_STENCIL;
        if((flags & Buffer::BindFlags::UnorderedAccess) != Buffer::BindFlags::None) d3dFlags |= D3D11_BIND_UNORDERED_ACCESS;

        return d3dFlags;
    }

    uint32_t getCpuAccessFlags(Buffer::AccessFlags flags)
    {
        uint32_t d3dFlags = 0;
        if((flags & Buffer::AccessFlags::MapRead) != Buffer::AccessFlags::None) d3dFlags |= D3D11_CPU_ACCESS_READ;
        if((flags & Buffer::AccessFlags::MapWrite) != Buffer::AccessFlags::None) d3dFlags |= D3D11_CPU_ACCESS_WRITE;
        return d3dFlags;
    }

    D3D11_USAGE getUsageFlags(Buffer::AccessFlags flags)
    {
        // Order matters
        if((flags & Buffer::AccessFlags::MapRead) != Buffer::AccessFlags::None) return D3D11_USAGE_STAGING;
        if((flags & Buffer::AccessFlags::MapWrite) != Buffer::AccessFlags::None) return D3D11_USAGE_DYNAMIC;
        return D3D11_USAGE_DEFAULT;
    }

    Buffer::SharedPtr Buffer::create(size_t size, BindFlags usage, AccessFlags access, const void* pInitData)
    {
        access = access | AccessFlags::Dynamic; // DX resources are always dynamic
        SharedPtr pBuffer = SharedPtr(new Buffer(size, usage, access));

        D3D11_BUFFER_DESC desc;
        desc.BindFlags = GetBindFlags(usage);
        assert(size < UINT32_MAX);
        desc.ByteWidth = (uint32_t)size;
        desc.CPUAccessFlags = getCpuAccessFlags(access);
        desc.MiscFlags = 0;
        desc.StructureByteStride = 0;
        desc.Usage = getUsageFlags(access);

        D3D11_SUBRESOURCE_DATA subresource;
        D3D11_SUBRESOURCE_DATA* pSubresource = pInitData ? &subresource : nullptr;
        if(pInitData)
        {
            subresource.pSysMem = pInitData;
        }

        d3d_call(getD3D11Device()->CreateBuffer(&desc, pSubresource, &pBuffer->mApiHandle));

        return pBuffer;
    }

    Buffer::~Buffer() = default;

    void Buffer::copy(Buffer* pDst) const
    {
        if(mSize != pDst->mSize)
        {
            logError("Error in Buffer::Copy().\nSource buffer size is " + std::to_string(mSize) + ", Destination buffer size is " + std::to_string(pDst->mSize) + ".\nBuffers should have the same size.");
            return;
        }
        getD3D11ImmediateContext()->CopyResource(pDst->mApiHandle, mApiHandle);
    }

    void Buffer::copy(Buffer* pDst, size_t srcOffset, size_t dstOffset, size_t count) const
    {
        if(mSize < srcOffset + count || pDst->mSize < dstOffset + count)
        {
            logError("Error in Buffer::Copy().\nSource buffer size is " + std::to_string(mSize) + ", Destination buffer size is " + std::to_string(pDst->mSize) + ", Copy offsets are " + std::to_string(srcOffset) + ", " + std::to_string(dstOffset) + ", Copy size is " + std::to_string(count) + ".\nBuffers are too small to perform copy.");
            return;
        }

        struct D3D11_BOX srcBox;
        srcBox.left = (UINT)dstOffset;
        srcBox.right = (UINT)(dstOffset + count);
        srcBox.top = 0;
        srcBox.bottom = 1;
        srcBox.front = 0;
        srcBox.back = 1;
        getD3D11ImmediateContext()->CopySubresourceRegion(pDst->mApiHandle, 0, (UINT)dstOffset, 0, 0, mApiHandle, 0, &srcBox);
    }

    void Buffer::updateData(const void* pData, size_t offset, size_t size, bool forceUpdate)
    {
        assert((mAccessFlags & Buffer::AccessFlags::Dynamic) != Buffer::AccessFlags::None);
        if((size + offset) > mSize)
        {
            std::string Error = "Buffer::updateData called with data larger then the buffer size. Buffer size = " + std::to_string(mSize) + " , Data size = " + std::to_string(size) + ".";
            logError(Error);
        }
        D3D11_BOX box;
        box.left = (uint32_t)offset;
        box.right = (uint32_t)(offset + size);
        box.top = 0;
        box.bottom = 1;
        box.front = 0;
        box.back = 1;

        getD3D11ImmediateContext()->UpdateSubresource(mApiHandle, 0, &box, pData, 0, 0);
    }

    void Buffer::readData(void* pData, size_t offset, size_t size) const
    {
        UNSUPPORTED_IN_D3D11("Buffer::ReadData(). If you really need this, create the resource with CPU read flag, and use Buffer::Map()");
    }

    uint64_t Buffer::getBindlessHandle(Buffer::GpuAccessFlags flags)
    {
        UNSUPPORTED_IN_D3D11("DX11 buffers don't have bindless handles.");
        return 0;
    }

    void* Buffer::map(MapType type)
    {
        if(mIsMapped)
        {
            logError("Buffer::Map() error. Buffer is already mapped");
            return nullptr;
        }

        D3D11_MAP dxFlag;
        switch(type)
        {
        case MapType::Read:
            dxFlag = D3D11_MAP_READ;
            break;
        case MapType::Write:
            dxFlag = D3D11_MAP_WRITE;
            break;
        case MapType::ReadWrite:
            dxFlag = D3D11_MAP_READ_WRITE;
            break;
        case MapType::WriteDiscard:
            dxFlag = D3D11_MAP_WRITE_DISCARD;
            break;
        case MapType::WriteNoOverwrite:
            dxFlag = D3D11_MAP_WRITE_NO_OVERWRITE;
        default:
            should_not_get_here();
        }

        mIsMapped = true;
        D3D11_MAPPED_SUBRESOURCE mapData;
        d3d_call(getD3D11ImmediateContext()->Map(mApiHandle, 0, dxFlag, 0, &mapData));
        return mapData.pData;
    }

    void Buffer::unmap()
    {
        if(mIsMapped == false)
        {
            logError("Buffer::Unmap() error. Buffer is not mapped");
            return;
        }
        mIsMapped = false;
        getD3D11ImmediateContext()->Unmap(mApiHandle, 0);
    }

    uint64_t Buffer::makeResident(Buffer::GpuAccessFlags flags/* = Buffer::GpuAccessFlags::ReadOnly*/) const
    {
        UNSUPPORTED_IN_D3D11("Buffer::makeResident()");
        return 0;
    }

    void Buffer::evict() const
    {
        UNSUPPORTED_IN_D3D11("Buffer::evict()");
    }
}
