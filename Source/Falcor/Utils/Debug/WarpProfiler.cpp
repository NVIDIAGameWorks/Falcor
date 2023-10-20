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
#include "WarpProfiler.h"
#include "Core/API/Device.h"
#include "Core/API/RenderContext.h"
#include "Core/Program/ShaderVar.h"
#include <fstream>

namespace Falcor
{

WarpProfiler::WarpProfiler(ref<Device> pDevice, const uint32_t binCount) : mBinCount(binCount)
{
    mpFence = pDevice->createFence();
    uint32_t elemCount = binCount * kWarpSize;
    mpHistogramBuffer = pDevice->createStructuredBuffer(
        4, elemCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false
    );
    mpHistogramStagingBuffer = pDevice->createStructuredBuffer(4, elemCount, ResourceBindFlags::None, MemoryType::ReadBack, nullptr, false);
}

void WarpProfiler::bindShaderData(const ShaderVar& var) const
{
    var["gWarpHistogram"] = mpHistogramBuffer;
}

void WarpProfiler::begin(RenderContext* pRenderContext)
{
    FALCOR_CHECK(!mActive, "WarpProfiler: begin() already called.");

    pRenderContext->clearUAV(mpHistogramBuffer->getUAV().get(), uint4(0));

    mActive = true;
    mDataWaiting = false;
}

void WarpProfiler::end(RenderContext* pRenderContext)
{
    FALCOR_CHECK(mActive, "WarpProfiler: end() called without preceding begin().");

    pRenderContext->copyResource(mpHistogramStagingBuffer.get(), mpHistogramBuffer.get());

    // Submit command list and insert signal.
    pRenderContext->submit(false);
    pRenderContext->signal(mpFence.get());

    mActive = false;
    mDataWaiting = true;
}

std::vector<uint32_t> WarpProfiler::getWarpHistogram(const uint32_t binIndex, const uint32_t binCount)
{
    readBackData();

    FALCOR_CHECK(binIndex + binCount <= mBinCount, "WarpProfiler: Bin index out of range.");
    FALCOR_CHECK(!mHistograms.empty(), "WarpProfiler: No available data. Did you call begin()/end()?");

    std::vector<uint32_t> histogram(kWarpSize, 0);
    for (size_t i = binIndex; i < binIndex + binCount; i++)
    {
        for (size_t j = 0; j < kWarpSize; j++)
        {
            histogram[j] += mHistograms[i * kWarpSize + j];
        }
    }

    return histogram;
}

bool WarpProfiler::saveWarpHistogramsAsCSV(const std::filesystem::path& path)
{
    readBackData();

    std::ofstream ofs(path);
    if (!ofs.good())
        return false;

    size_t k = 0;
    for (size_t i = 0; i < mBinCount; i++)
    {
        for (size_t j = 0; j < kWarpSize; j++)
        {
            ofs << mHistograms[k++];
            if (j < kWarpSize - 1)
                ofs << ";";
        }
        ofs << std::endl;
    }

    return true;
}

void WarpProfiler::readBackData()
{
    if (!mDataWaiting)
        return;

    FALCOR_CHECK(!mActive, "WarpProfiler: readBackData() called without preceding before()/end() calls.");
    mpFence->wait();
    mHistograms.resize(mBinCount * kWarpSize);

    const uint32_t* data = reinterpret_cast<const uint32_t*>(mpHistogramStagingBuffer->map(Buffer::MapType::Read));
    std::memcpy(mHistograms.data(), data, mHistograms.size() * sizeof(uint32_t));
    mpHistogramStagingBuffer->unmap();

    mDataWaiting = false;
}

} // namespace Falcor
