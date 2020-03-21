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
 **************************************************************************/
#include "stdafx.h"
#include "PixelStats.h"
#include <sstream>
#include <iomanip>

namespace Falcor
{
    PixelStats::SharedPtr PixelStats::create()
    {
        return SharedPtr(new PixelStats());
    }

    void PixelStats::beginFrame(RenderContext* pRenderContext, const glm::uvec2& frameDim)
    {
        // Prepare state.
        assert(!mRunning);
        mRunning = true;
        mWaitingForData = false;
        mFrameDim = frameDim;

        // Mark previously stored data as invalid. The config may have changed, so this is the safe bet.
        mStats = Stats();
        mStatsValid = false;
        mStatsBuffersValid = false;

        if (mStatsEnabled)
        {
            // Create parallel reduction helper.
            if (!mpParallelReduction)
            {
                mpParallelReduction = ComputeParallelReduction::create();
                mpReductionResult = Buffer::create(32, ResourceBindFlags::None, Buffer::CpuAccess::Read);
            }

            // Prepare stats buffer.
            if (!mpStatsRayCount || mpStatsRayCount->getWidth() != frameDim.x || mpStatsRayCount->getHeight() != frameDim.y)
            {
                mpStatsRayCount = Texture::create2D(frameDim.x, frameDim.y, ResourceFormat::R32Uint, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
                mpStatsPathLength = Texture::create2D(frameDim.x, frameDim.y, ResourceFormat::R32Uint, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
            }

            assert(mpStatsRayCount && mpStatsPathLength);
            pRenderContext->clearUAV(mpStatsRayCount->getUAV().get(), uvec4(0, 0, 0, 0));
            pRenderContext->clearUAV(mpStatsPathLength->getUAV().get(), uvec4(0, 0, 0, 0));
        }
    }

    void PixelStats::endFrame(RenderContext* pRenderContext)
    {
        assert(mRunning);
        mRunning = false;

        if (mStatsEnabled)
        {
            // Create fence first time we need it.
            if (!mpFence) mpFence = GpuFence::create();

            // Sum of the per-pixel counters. The results are copied to a GPU buffer.
            mpParallelReduction->execute<uint4>(pRenderContext, mpStatsRayCount, ComputeParallelReduction::Type::Sum, nullptr, mpReductionResult, 0);
            mpParallelReduction->execute<uint4>(pRenderContext, mpStatsPathLength, ComputeParallelReduction::Type::Sum, nullptr, mpReductionResult, 16);

            // Submit command list and insert signal.
            pRenderContext->flush(false);
            mpFence->gpuSignal(pRenderContext->getLowLevelData()->getCommandQueue());

            mStatsBuffersValid = true;
            mWaitingForData = true;
        }
    }

    void PixelStats::prepareProgram(const Program::SharedPtr& pProgram, const ShaderVar& var)
    {
        assert(mRunning);

        pProgram->addDefine("_PIXEL_STATS_ENABLED", mStatsEnabled ? "1" : "0");

        if (mStatsEnabled)
        {
            var["gStatsRayCount"] = mpStatsRayCount;
            var["gStatsPathLength"] = mpStatsPathLength;
        }
    }

    void PixelStats::renderUI(Gui::Widgets& widget)
    {
        // Configuration.
        widget.checkbox("Pixel stats", mStatsEnabled);
        widget.tooltip("Collects ray tracing traversal stats on the GPU.\nNote that this option slows down the performance.");

        // Fetch data and show stats if available.
        copyStatsToCPU();
        if (mStatsValid)
        {
            std::ostringstream oss;
            oss << "Path length (avg): " << std::fixed << std::setprecision(3) << mStats.avgPathLength << "\n";
            oss << "Traced rays (avg): " << std::fixed << std::setprecision(3) << mStats.avgRaysPerPixel << "\n";
            oss << "Traced rays (total): " << mStats.totalRays << "\n";
            widget.text(oss.str().c_str());
        }
    }

    bool PixelStats::getStats(PixelStats::Stats& stats)
    {
        copyStatsToCPU();
        if (!mStatsValid)
        {
            logWarning("PixelStats::getStats() - Stats are not valid. Ignoring.");
            return false;
        }
        stats = mStats;
        return true;
    }

    const Texture::SharedPtr PixelStats::getRayCountBuffer() const
    {
        assert(!mRunning);
        return mStatsBuffersValid ? mpStatsRayCount : nullptr;
    }

    void PixelStats::copyStatsToCPU()
    {
        assert(!mRunning);
        if (mWaitingForData)
        {
            // Wait for signal.
            mpFence->syncCpu();
            mWaitingForData = false;

            if (mStatsEnabled)
            {
                // Map the stats buffer.
                const uint4* data = static_cast<const uint4*>(mpReductionResult->map(Buffer::MapType::Read));
                assert(data);
                const uint32_t totalRayCount = data[0].x;
                const uint32_t totalPathLength = data[1].x;
                mpReductionResult->unmap();

                // Store stats locally.
                const uint32_t numPixels = mFrameDim.x * mFrameDim.y;
                assert(numPixels > 0);

                mStats.totalRays = totalRayCount;
                mStats.avgRaysPerPixel = (float)totalRayCount / numPixels;
                mStats.avgPathLength = (float)totalPathLength / numPixels;

                mStatsValid = true;
            }
        }
    }

}
