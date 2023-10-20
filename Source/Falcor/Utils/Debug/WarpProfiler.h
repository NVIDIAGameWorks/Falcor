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
#pragma once

#include "Core/Macros.h"
#include "Core/API/Buffer.h"
#include "Core/API/Fence.h"

#include <filesystem>
#include <vector>

namespace Falcor
{

class RenderContext;
struct ShaderVar;

/**
 * @brief Utility class for warp-level profiling.
 */
class FALCOR_API WarpProfiler
{
public:
    static constexpr size_t kWarpSize = 32; // Do not change

    /**
     * @brief Construct new warp profiler object.
     * @param[in] pDevice GPU device.
     * @param[in] binCount Number of profiling bins.
     */
    WarpProfiler(ref<Device> pDevice, const uint32_t binCount);

    /**
     * @brief Binds the profiler data to shader vars.
     * This function must be called before the profiler can be used.
     * @param[in] var Shader vars of the program to set data into.
     */
    void bindShaderData(const ShaderVar& var) const;

    /**
     * @brief Begin profiling.
     * @param[in] pRenderContext The context.
     */
    void begin(RenderContext* pRenderContext);

    /**
     * @brief End profiling.
     * @param[in] pRenderContext The context.
     */
    void end(RenderContext* pRenderContext);

    /**
     * @brief Compute warp histogram over a range of profiling bins.
     * @param[in] binIndex Index of first profiling bin for histogram.
     * @param[in] binCount Number of profiling bins to include in histogram.
     * @return Histogram with `kWarpSize` buckets. The first bucket number of warps with 1 counted element, the last
     * bucket represents number of warps with `kWarpSize` counted elements.
     */
    std::vector<uint32_t> getWarpHistogram(const uint32_t binIndex, const uint32_t binCount = 1);

    /**
     * Save warp histograms for all profiling bins to file in CSV format.
     * @param[in] path File path.
     * @return True if successful, false otherwise.
     */
    bool saveWarpHistogramsAsCSV(const std::filesystem::path& path);

private:
    void readBackData();

    ref<Fence> mpFence;
    ref<Buffer> mpHistogramBuffer;
    ref<Buffer> mpHistogramStagingBuffer;

    const uint32_t mBinCount;          ///< Number of profiling bins.
    std::vector<uint32_t> mHistograms; ///< Histograms for all profiling bins.

    bool mActive = false;      ///< True while inside a begin()/end() section.
    bool mDataWaiting = false; ///< True when data is waiting for readback in the staging buffer.
};
} // namespace Falcor
