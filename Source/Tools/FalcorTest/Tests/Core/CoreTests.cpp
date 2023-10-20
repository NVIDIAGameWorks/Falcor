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
#include "Testing/UnitTest.h"

namespace Falcor
{
GPU_TEST(TransientHeapRecycling)
{
    ref<Device> pDevice = ctx.getDevice();
    RenderContext* pRenderContext = pDevice->getRenderContext();

    size_t M = 1024 * 1024 * 1024;
    std::vector<uint8_t> cpuBuf(M, 0);
    ref<Buffer> A = pDevice->createBuffer(M, ResourceBindFlags::None, MemoryType::DeviceLocal, cpuBuf.data());
    ref<Buffer> B = pDevice->createBuffer(4, ResourceBindFlags::None, MemoryType::DeviceLocal);

    // Progress through N frames (and transient heaps), ending up using the
    // same transient heap as is used for uploading the data to buffer A.
    // Before the fix, this leads to a validation error as the buffer for
    // uplading to buffer A is still in flight.
    for (uint32_t i = 0; i < Device::kInFlightFrameCount; ++i)
        pDevice->endFrame();

    // The following commands will trigger a TDR even if the validation error
    // is missed.
    pRenderContext->copyBufferRegion(B.get(), 0, A.get(), 0, 4);
    pRenderContext->submit(true);
    // A->map(Buffer::MapType::Read);
    // A->unmap();
}
} // namespace Falcor
