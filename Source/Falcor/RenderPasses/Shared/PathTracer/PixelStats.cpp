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
#include "PixelStats.h"
#include <sstream>
#include <iomanip>

namespace Falcor
{
    namespace
    {
        const char kComputeRayCountFilename[] = "RenderPasses/Shared/PathTracer/PixelStats.cs.slang";
    }

    PixelStats::SharedPtr PixelStats::create()
    {
        return SharedPtr(new PixelStats());
    }

    PixelStats::PixelStats()
    {
        mpComputeRayCount = ComputePass::create(kComputeRayCountFilename, "main");
    }

    void PixelStats::beginFrame(RenderContext* pRenderContext, const uint2& frameDim)
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
        mRayCountTextureValid = false;

        if (mStatsEnabled)
        {
            // Create parallel reduction helper.
            if (!mpParallelReduction)
            {
                mpParallelReduction = ComputeParallelReduction::create();
                mpReductionResult = Buffer::create((kRayTypeCount + 1) * sizeof(uint4), ResourceBindFlags::None, Buffer::CpuAccess::Read);
            }

            // Prepare stats buffers.
            if (!mpStatsPathLength || mpStatsPathLength->getWidth() != frameDim.x || mpStatsPathLength->getHeight() != frameDim.y)
            {
                for (uint32_t i = 0; i < kRayTypeCount; i++)
                {
                    mpStatsRayCount[i] = Texture::create2D(frameDim.x, frameDim.y, ResourceFormat::R32Uint, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
                }
                mpStatsPathLength = Texture::create2D(frameDim.x, frameDim.y, ResourceFormat::R32Uint, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
            }

            for (uint32_t i = 0; i < kRayTypeCount; i++)
            {
                pRenderContext->clearUAV(mpStatsRayCount[i]->getUAV().get(), uint4(0, 0, 0, 0));
            }
            pRenderContext->clearUAV(mpStatsPathLength->getUAV().get(), uint4(0, 0, 0, 0));
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
            for (uint32_t i = 0; i < kRayTypeCount; i++)
            {
                mpParallelReduction->execute<uint4>(pRenderContext, mpStatsRayCount[i], ComputeParallelReduction::Type::Sum, nullptr, mpReductionResult, i * sizeof(uint4));
            }
            mpParallelReduction->execute<uint4>(pRenderContext, mpStatsPathLength, ComputeParallelReduction::Type::Sum, nullptr, mpReductionResult, kRayTypeCount * sizeof(uint4));

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

        if (mStatsEnabled)
        {
            pProgram->addDefine("_PIXEL_STATS_ENABLED");
            for (uint32_t i = 0; i < kRayTypeCount; i++)
            {
                var["gStatsRayCount"][i] = mpStatsRayCount[i];
            }
            var["gStatsPathLength"] = mpStatsPathLength;
        }
        else
        {
            pProgram->removeDefine("_PIXEL_STATS_ENABLED");
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
            oss << "Path length (avg): " << std::fixed << std::setprecision(3) << mStats.avgPathLength << "\n"
                << "Total rays (avg): " << std::fixed << std::setprecision(3) << mStats.avgTotalRaysPerPixel << "\n"
                << "Shadow rays (avg): " << std::fixed << std::setprecision(3) << mStats.avgShadowRaysPerPixel << "\n"
                << "ClosestHit rays (avg): " << std::fixed << std::setprecision(3) << mStats.avgClosestHitRaysPerPixel << "\n"
                << "Total rays: " << mStats.totalRays << "\n"
                << "Shadow rays: " << mStats.shadowRays << "\n"
                << "ClosestHit rays: " << mStats.closestHitRays << "\n";
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

    const Texture::SharedPtr PixelStats::getRayCountTexture(RenderContext* pRenderContext)
    {
        assert(!mRunning);
        if (!mStatsBuffersValid) return nullptr;

        if (!mRayCountTextureValid)
        {
            computeRayCountTexture(pRenderContext);
        }

        assert(mRayCountTextureValid);
        return mpStatsRayCountTotal;
    }

    void PixelStats::computeRayCountTexture(RenderContext* pRenderContext)
    {
        assert(mStatsBuffersValid);
        if (!mpStatsRayCountTotal || mpStatsRayCountTotal->getWidth() != mFrameDim.x || mpStatsRayCountTotal->getHeight() != mFrameDim.y)
        {
            mpStatsRayCountTotal = Texture::create2D(mFrameDim.x, mFrameDim.y, ResourceFormat::R32Uint, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
        }

        auto var = mpComputeRayCount->getRootVar();
        for (uint32_t i = 0; i < kRayTypeCount; i++)
        {
            var["gStatsRayCount"][i] = mpStatsRayCount[i];
        }
        var["gStatsRayCountTotal"] = mpStatsRayCountTotal;
        var["CB"]["gFrameDim"] = mFrameDim;

        mpComputeRayCount->execute(pRenderContext, mFrameDim.x, mFrameDim.y);
        mRayCountTextureValid = true;
    }

    const Texture::SharedPtr PixelStats::getPathLengthTexture() const
    {
        assert(!mRunning);
        return mStatsBuffersValid ? mpStatsPathLength : nullptr;
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
                const uint4* result = static_cast<const uint4*>(mpReductionResult->map(Buffer::MapType::Read));
                assert(result);

                const uint32_t totalPathLength = result[kRayTypeCount].x;
                const uint32_t numPixels = mFrameDim.x * mFrameDim.y;
                assert(numPixels > 0);

                mStats.shadowRays = result[(uint32_t)PixelStatsRayType::Shadow].x;
                mStats.closestHitRays = result[(uint32_t)PixelStatsRayType::ClosestHit].x;
                mStats.totalRays = mStats.shadowRays + mStats.closestHitRays;
                mStats.avgShadowRaysPerPixel = (float)mStats.shadowRays / numPixels;
                mStats.avgClosestHitRaysPerPixel = (float)mStats.closestHitRays / numPixels;
                mStats.avgTotalRaysPerPixel = (float)mStats.totalRays / numPixels;
                mStats.avgPathLength = (float)totalPathLength / numPixels;

                mpReductionResult->unmap();
                mStatsValid = true;
            }
        }
    }
}
