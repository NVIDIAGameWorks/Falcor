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
#include "Core/API/GpuMemoryHeap.h"
#include "D3D12Resource.h"

namespace Falcor
{
    ID3D12ResourcePtr createBuffer(Buffer::State initState, size_t size, const D3D12_HEAP_PROPERTIES& heapProps, Buffer::BindFlags bindFlags);

    namespace
    {
        D3D12_HEAP_PROPERTIES getHeapProps(GpuMemoryHeap::Type t)
        {
            switch (t)
            {
            case GpuMemoryHeap::Type::Default:
                return kDefaultHeapProps;
            case GpuMemoryHeap::Type::Upload:
                return kUploadHeapProps;
            case GpuMemoryHeap::Type::Readback:
                return kReadbackHeapProps;
            default:
                should_not_get_here();
                return D3D12_HEAP_PROPERTIES();
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
                should_not_get_here();
                return Buffer::State::Undefined;
            }
        }
    }

    void GpuMemoryHeap::initBasePageData(BaseData& data, size_t size)
    {
        data.pResourceHandle = createBuffer(getInitState(mType), size, getHeapProps(mType), Buffer::BindFlags::None);
        data.offset = 0;
        D3D12_RANGE readRange = {};
        d3d_call(data.pResourceHandle->Map(0, &readRange, (void**)&data.pData));
    }
}
