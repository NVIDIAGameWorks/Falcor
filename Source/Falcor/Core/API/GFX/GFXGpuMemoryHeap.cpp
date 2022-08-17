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
#include "Core/API/GpuMemoryHeap.h"
#include "Core/API/Buffer.h"
#include "Core/API/GFX/GFXAPI.h"

namespace Falcor
{
    Slang::ComPtr<gfx::IBufferResource> createBuffer(Buffer::State initState, size_t size, Buffer::BindFlags bindFlags, Buffer::CpuAccess cpuAccess);

    namespace
    {
        Buffer::CpuAccess getCpuAccess(GpuMemoryHeap::Type t)
        {
            switch (t)
            {
            case GpuMemoryHeap::Type::Default:
                return Buffer::CpuAccess::None;
            case GpuMemoryHeap::Type::Upload:
                return Buffer::CpuAccess::Write;
            case GpuMemoryHeap::Type::Readback:
                return Buffer::CpuAccess::Read;
            default:
                FALCOR_UNREACHABLE();
                return Buffer::CpuAccess::None;
            }
        }

        Buffer::State getInitState(GpuMemoryHeap::Type t)
        {
            switch (t)
            {
            case GpuMemoryHeap::Type::Default:
                return Buffer::State::Common;
            case GpuMemoryHeap::Type::Upload:
                return Buffer::State::GenericRead;
            case GpuMemoryHeap::Type::Readback:
                return Buffer::State::CopyDest;
            default:
                FALCOR_UNREACHABLE();
                return Buffer::State::Undefined;
            }
        }
    }

    void GpuMemoryHeap::initBasePageData(BaseData& data, size_t size)
    {
        data.pResourceHandle = createBuffer(
            getInitState(mType),
            size,
            Buffer::BindFlags::Vertex | Buffer::BindFlags::Index | Buffer::BindFlags::Constant,
            getCpuAccess(mType));
        data.offset = 0;
        auto bufferResource = static_cast<gfx::IBufferResource*>(data.pResourceHandle.get());
        bufferResource->map(nullptr, (void**)&data.pData);
    }
}
