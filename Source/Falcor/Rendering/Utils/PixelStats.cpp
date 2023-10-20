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
#include "PixelStats.h"
#include "Core/API/RenderContext.h"
#include "Utils/Logger.h"
#include "Utils/Scripting/ScriptBindings.h"
#include <sstream>
#include <iomanip>

namespace Falcor
{
    namespace
    {
        const char kComputeRayCountFilename[] = "Rendering/Utils/PixelStats.cs.slang";

        pybind11::dict toPython(const PixelStats::Stats& stats)
        {
            pybind11::dict d;
            d["visibilityRays"] = stats.visibilityRays;
            d["closestHitRays"] = stats.closestHitRays;
            d["totalRays"] = stats.totalRays;
            d["pathVertices"] = stats.pathVertices;
            d["volumeLookups"] = stats.volumeLookups;
            d["avgVisibilityRays"] = stats.avgVisibilityRays;
            d["avgClosestHitRays"] = stats.avgClosestHitRays;
            d["avgTotalRays"] = stats.avgTotalRays;
            d["avgPathLength"] = stats.avgPathLength;
            d["avgPathVertices"] = stats.avgPathVertices;
            d["avgVolumeLookups"] = stats.avgVolumeLookups;
            return d;
        }
    }

    PixelStats::PixelStats(ref<Device> pDevice)
        : mpDevice(pDevice)
    {
        mpComputeRayCount = ComputePass::create(mpDevice, kComputeRayCountFilename, "main");
    }

    void PixelStats::beginFrame(RenderContext* pRenderContext, const uint2& frameDim)
    {
        // Prepare state.
        FALCOR_ASSERT(!mRunning);
        mRunning = true;
        mWaitingForData = false;
        mFrameDim = frameDim;

        // Mark previously stored data as invalid. The config may have changed, so this is the safe bet.
        mStats = Stats();
        mStatsValid = false;
        mStatsBuffersValid = false;
        mRayCountTextureValid = false;

        if (mEnabled)
        {
            // Create parallel reduction helper.
            if (!mpParallelReduction)
            {
                mpParallelReduction = std::make_unique<ParallelReduction>(mpDevice);
                mpReductionResult = mpDevice->createBuffer((kRayTypeCount + 3) * sizeof(uint4), ResourceBindFlags::None, MemoryType::ReadBack);
            }

            // Prepare stats buffers.
            if (!mpStatsPathLength || mpStatsPathLength->getWidth() != frameDim.x || mpStatsPathLength->getHeight() != frameDim.y)
            {
                for (uint32_t i = 0; i < kRayTypeCount; i++)
                {
                    mpStatsRayCount[i] = mpDevice->createTexture2D(frameDim.x, frameDim.y, ResourceFormat::R32Uint, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
                }
                mpStatsPathLength = mpDevice->createTexture2D(frameDim.x, frameDim.y, ResourceFormat::R32Uint, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
                mpStatsPathVertexCount = mpDevice->createTexture2D(frameDim.x, frameDim.y, ResourceFormat::R32Uint, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
                mpStatsVolumeLookupCount = mpDevice->createTexture2D(frameDim.x, frameDim.y, ResourceFormat::R32Uint, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
            }

            for (uint32_t i = 0; i < kRayTypeCount; i++)
            {
                pRenderContext->clearUAV(mpStatsRayCount[i]->getUAV().get(), uint4(0, 0, 0, 0));
            }
            pRenderContext->clearUAV(mpStatsPathLength->getUAV().get(), uint4(0, 0, 0, 0));
            pRenderContext->clearUAV(mpStatsPathVertexCount->getUAV().get(), uint4(0, 0, 0, 0));
            pRenderContext->clearUAV(mpStatsVolumeLookupCount->getUAV().get(), uint4(0, 0, 0, 0));
        }
    }

    void PixelStats::endFrame(RenderContext* pRenderContext)
    {
        FALCOR_ASSERT(mRunning);
        mRunning = false;

        if (mEnabled)
        {
            // Create fence first time we need it.
            if (!mpFence) mpFence = mpDevice->createFence();

            // Sum of the per-pixel counters. The results are copied to a GPU buffer.
            for (uint32_t i = 0; i < kRayTypeCount; i++)
            {
                mpParallelReduction->execute<uint4>(pRenderContext, mpStatsRayCount[i], ParallelReduction::Type::Sum, nullptr, mpReductionResult, i * sizeof(uint4));
            }
            mpParallelReduction->execute<uint4>(pRenderContext, mpStatsPathLength, ParallelReduction::Type::Sum, nullptr, mpReductionResult, kRayTypeCount * sizeof(uint4));
            mpParallelReduction->execute<uint4>(pRenderContext, mpStatsPathVertexCount, ParallelReduction::Type::Sum, nullptr, mpReductionResult, (kRayTypeCount + 1) * sizeof(uint4));
            mpParallelReduction->execute<uint4>(pRenderContext, mpStatsVolumeLookupCount, ParallelReduction::Type::Sum, nullptr, mpReductionResult, (kRayTypeCount + 2) * sizeof(uint4));

            // Submit command list and insert signal.
            pRenderContext->submit(false);
            pRenderContext->signal(mpFence.get());

            mStatsBuffersValid = true;
            mWaitingForData = true;
        }
    }

    void PixelStats::prepareProgram(const ref<Program>& pProgram, const ShaderVar& var)
    {
        FALCOR_ASSERT(mRunning);

        if (mEnabled)
        {
            pProgram->addDefine("_PIXEL_STATS_ENABLED");
            for (uint32_t i = 0; i < kRayTypeCount; i++)
            {
                var["gStatsRayCount"][i] = mpStatsRayCount[i];
            }
            var["gStatsPathLength"] = mpStatsPathLength;
            var["gStatsPathVertexCount"] = mpStatsPathVertexCount;
            var["gStatsVolumeLookupCount"] = mpStatsVolumeLookupCount;
        }
        else
        {
            pProgram->removeDefine("_PIXEL_STATS_ENABLED");
        }
    }

    void PixelStats::renderUI(Gui::Widgets& widget)
    {
        // Configuration.
        widget.checkbox("Ray stats", mEnabled);
        widget.tooltip("Collects ray tracing traversal stats on the GPU.\nNote that this option slows down the performance.");

        // Fetch data and show stats if available.
        copyStatsToCPU();
        if (mStatsValid)
        {
            widget.text("Stats:");
            widget.tooltip("All averages are per pixel on screen.\n"
                "\n"
                "The path vertex count includes:\n"
                " - Primary hits\n"
                " - Secondary hits on geometry\n"
                " - Secondary misses on envmap\n"
                "\n"
                "Note that the camera/sensor is not included, nor misses when there is no envmap (no-op miss shader).");

            std::ostringstream oss;
            oss << "Path length (avg): " << std::fixed << std::setprecision(3) << mStats.avgPathLength << "\n"
                << "Path vertices (avg): " << std::fixed << std::setprecision(3) << mStats.avgPathVertices << "\n"
                << "Total rays (avg): " << std::fixed << std::setprecision(3) << mStats.avgTotalRays << "\n"
                << "Visibility rays (avg): " << std::fixed << std::setprecision(3) << mStats.avgVisibilityRays << "\n"
                << "ClosestHit rays (avg): " << std::fixed << std::setprecision(3) << mStats.avgClosestHitRays << "\n"
                << "Path vertices: " << mStats.pathVertices << "\n"
                << "Total rays: " << mStats.totalRays << "\n"
                << "Visibility rays: " << mStats.visibilityRays << "\n"
                << "ClosestHit rays: " << mStats.closestHitRays << "\n"
                << "Volume lookups: " << mStats.volumeLookups << "\n"
                << "Volume lookups (avg): " << mStats.avgVolumeLookups << "\n";

            widget.checkbox("Enable logging", mEnableLogging);
            widget.text(oss.str());

            if (mEnableLogging) logInfo("\n" + oss.str());
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

    const ref<Texture> PixelStats::getRayCountTexture(RenderContext* pRenderContext)
    {
        FALCOR_ASSERT(!mRunning);
        if (!mStatsBuffersValid) return nullptr;

        if (!mRayCountTextureValid)
        {
            computeRayCountTexture(pRenderContext);
        }

        FALCOR_ASSERT(mRayCountTextureValid);
        return mpStatsRayCountTotal;
    }

    void PixelStats::computeRayCountTexture(RenderContext* pRenderContext)
    {
        FALCOR_ASSERT(mStatsBuffersValid);
        if (!mpStatsRayCountTotal || mpStatsRayCountTotal->getWidth() != mFrameDim.x || mpStatsRayCountTotal->getHeight() != mFrameDim.y)
        {
            mpStatsRayCountTotal = mpDevice->createTexture2D(mFrameDim.x, mFrameDim.y, ResourceFormat::R32Uint, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
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

    const ref<Texture> PixelStats::getPathLengthTexture() const
    {
        FALCOR_ASSERT(!mRunning);
        return mStatsBuffersValid ? mpStatsPathLength : nullptr;
    }

    const ref<Texture> PixelStats::getPathVertexCountTexture() const
    {
        FALCOR_ASSERT(!mRunning);
        return mStatsBuffersValid ? mpStatsPathVertexCount : nullptr;
    }

    const ref<Texture> PixelStats::getVolumeLookupCountTexture() const
    {
        FALCOR_ASSERT(!mRunning);
        return mStatsBuffersValid ? mpStatsVolumeLookupCount : nullptr;
    }

    void PixelStats::copyStatsToCPU()
    {
        FALCOR_ASSERT(!mRunning);
        if (mWaitingForData)
        {
            // Wait for signal.
            mpFence->wait();
            mWaitingForData = false;

            if (mEnabled)
            {
                // Map the stats buffer.
                const uint4* result = static_cast<const uint4*>(mpReductionResult->map(Buffer::MapType::Read));
                FALCOR_ASSERT(result);

                const uint32_t totalPathLength = result[kRayTypeCount].x;
                const uint32_t totalPathVertices = result[kRayTypeCount + 1].x;
                const uint32_t totalVolumeLookups = result[kRayTypeCount + 2].x;
                const uint32_t numPixels = mFrameDim.x * mFrameDim.y;
                FALCOR_ASSERT(numPixels > 0);

                mStats.visibilityRays = result[(uint32_t)PixelStatsRayType::Visibility].x;
                mStats.closestHitRays = result[(uint32_t)PixelStatsRayType::ClosestHit].x;
                mStats.totalRays = mStats.visibilityRays + mStats.closestHitRays;
                mStats.pathVertices = totalPathVertices;
                mStats.volumeLookups = totalVolumeLookups;
                mStats.avgVisibilityRays = (float)mStats.visibilityRays / numPixels;
                mStats.avgClosestHitRays = (float)mStats.closestHitRays / numPixels;
                mStats.avgTotalRays = (float)mStats.totalRays / numPixels;
                mStats.avgPathLength = (float)totalPathLength / numPixels;
                mStats.avgPathVertices = (float)totalPathVertices / numPixels;
                mStats.avgVolumeLookups = (float)totalVolumeLookups / numPixels;

                mpReductionResult->unmap();
                mStatsValid = true;
            }
        }
    }

    FALCOR_SCRIPT_BINDING(PixelStats)
    {
        pybind11::class_<PixelStats> pixelStats(m, "PixelStats");
        pixelStats.def_property("enabled", &PixelStats::isEnabled, &PixelStats::setEnabled);
        pixelStats.def_property_readonly("stats", [](PixelStats* pPixelStats) {
            PixelStats::Stats stats;
            pPixelStats->getStats(stats);
            return toPython(stats);
        });
    }
}
