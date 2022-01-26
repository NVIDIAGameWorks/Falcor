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
#include "Animation.h"
#include "AnimatedVertexCache.h"

namespace Falcor
{
    namespace
    {
        const std::string kUpdateCurveVerticesFilename = "Scene/Animation/UpdateCurveVertices.slang";
        const std::string kUpdateCurveAABBsFilename = "Scene/Animation/UpdateCurveAABBs.slang";

        InterpolationInfo calculateInterpolation(double time, const std::vector<double>& timeSamples, Animation::Behavior preInfinityBehavior)
        {
            if (!std::isfinite(time))
            {
                return InterpolationInfo{ uint2(0), 0.f };
            }

            if (time < 0.0 || time > timeSamples.back())
            {
                time = clamp(time, 0.0, timeSamples.back());
            }

            uint2 keyframeIndices;
            float t;
            if (time <= timeSamples.front())
            {
                if (preInfinityBehavior == Animation::Behavior::Constant)
                {
                    keyframeIndices = uint2(0);
                    t = 0.f;
                }
                else if (preInfinityBehavior == Animation::Behavior::Cycle)
                {
                    keyframeIndices.x = (uint32_t)timeSamples.size() - 1;
                    keyframeIndices.y = 0;

                    // The first keyframe has timeCode >= 1 (see processCurve() in ImporterContext.cpp).
                    FALCOR_ASSERT(timeSamples.front() >= 1.0);
                    t = (float)(time / timeSamples.front());
                }
            }
            else
            {
                keyframeIndices.y = uint32_t(std::lower_bound(timeSamples.begin(), timeSamples.end(), time) - timeSamples.begin());
                keyframeIndices.x = keyframeIndices.y - 1;
                FALCOR_ASSERT(timeSamples[keyframeIndices.y] > timeSamples[keyframeIndices.x]);
                t = (float)((time - timeSamples[keyframeIndices.x]) / (timeSamples[keyframeIndices.y] - timeSamples[keyframeIndices.x]));
            }

            return InterpolationInfo{ keyframeIndices, t };
        }
    }

    AnimatedVertexCache::AnimatedVertexCache(Scene* pScene, std::vector<CachedCurve>&& cachedCurves, std::vector<CachedMesh>&& cachedMeshes)
        : mpScene(pScene)
        , mCachedCurves(cachedCurves)
    {
        if (cachedCurves.empty()) return;

        if (!mCachedCurves.empty())
        {
            initCurveKeyframes();
            bindCurveBuffers();

            createCurveVertexUpdatePass();
            createCurveAABBUpdatePass();
        }
    }

    AnimatedVertexCache::UniquePtr AnimatedVertexCache::create(Scene* pScene, std::vector<CachedCurve>&& cachedCurves, std::vector<CachedMesh>&& cachedMeshes)
    {
        return UniquePtr(new AnimatedVertexCache(pScene, std::move(cachedCurves), std::move(cachedMeshes)));
    }

    bool AnimatedVertexCache::animate(RenderContext* pContext, double time)
    {
        if (!hasAnimations()) return false;

        if (!mCachedCurves.empty())
        {
            double curveTime = mLoopAnimations ? std::fmod(time, mGlobalCurveAnimationLength) : time;
            executeCurveVertexUpdatePass(pContext, calculateInterpolation(curveTime, mCurveKeyframeTimes, mPreInfinityBehavior));
            executeCurveAABBUpdatePass(pContext);
        }

        return true;
    }

    void AnimatedVertexCache::copyToPrevVertices(RenderContext* pContext)
    {
        executeCurveVertexUpdatePass(pContext, InterpolationInfo{ uint2(0), 0.f }, true);
    }

    bool AnimatedVertexCache::hasAnimations() const
    {
        return hasCurveAnimations();
    }

    bool AnimatedVertexCache::hasCurveAnimations() const
    {
        return mCurveKeyframeTimes.size() > 1;
    }

    uint64_t AnimatedVertexCache::getMemoryUsageInBytes() const
    {
        uint64_t m = 0;
        for (size_t i = 0; i < mpCurveVertexBuffers.size(); i++) m += mpCurveVertexBuffers[i] ? mpCurveVertexBuffers[i]->getSize() : 0;
        m += mpPrevCurveVertexBuffer ? mpPrevCurveVertexBuffer->getSize() : 0;
        m += mpCurveIndexBuffer ? mpCurveIndexBuffer->getSize() : 0;
        return m;
    }

    // We create a merged list of all timestamps and generate new frames for curves where those timestamps are missing.
    // This can lead to fairly heavy overhead if we have cached curves with vastly different total length.
    // Currently, our assets have cached curves with the same list of timestamps.
    void AnimatedVertexCache::initCurveKeyframes()
    {
        // Align the time samples across vertex caches.
        mCurveKeyframeTimes.clear();
        for (size_t i = 0; i < mCachedCurves.size(); i++)
        {
            mCurveKeyframeTimes.insert(mCurveKeyframeTimes.end(), mCachedCurves[i].timeSamples.begin(), mCachedCurves[i].timeSamples.end());
        }
        std::sort(mCurveKeyframeTimes.begin(), mCurveKeyframeTimes.end());
        mCurveKeyframeTimes.erase(std::unique(mCurveKeyframeTimes.begin(), mCurveKeyframeTimes.end()), mCurveKeyframeTimes.end());

        mGlobalCurveAnimationLength = mCurveKeyframeTimes.empty() ? 0 : mCurveKeyframeTimes.back();
    }

    void AnimatedVertexCache::bindCurveBuffers()
    {
        // Compute curve vertex and index (segment) count.
        mCurveVertexCount = 0;
        mCurveIndexCount = 0;
        for (uint32_t i = 0; i < mpScene->getCurveCount(); i++)
        {
            FALCOR_ASSERT(mpScene->getCurve(i).vertexCount == (uint32_t)mCachedCurves[i].vertexData[0].size());
            FALCOR_ASSERT(mpScene->getCurve(i).indexCount == (uint32_t)mCachedCurves[i].indexData.size());
            mCurveVertexCount += mpScene->getCurve(i).vertexCount;
            mCurveIndexCount += mpScene->getCurve(i).indexCount;
        }

        // Create buffers for vertex positions in curve vertex caches.
        ResourceBindFlags vbBindFlags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
        mpCurveVertexBuffers.resize(mCurveKeyframeTimes.size());
        for (uint32_t i = 0; i < mCurveKeyframeTimes.size(); i++)
        {
            mpCurveVertexBuffers[i] = Buffer::createStructured(sizeof(DynamicCurveVertexData), mCurveVertexCount, vbBindFlags, Buffer::CpuAccess::None, nullptr, false);
            mpCurveVertexBuffers[i]->setName("AnimatedVertexCache::mpCurveVertexBuffers[" + std::to_string(i) + "]");
        }

        // Create buffers for previous vertex positions.
        mpPrevCurveVertexBuffer = Buffer::createStructured(sizeof(DynamicCurveVertexData), mCurveVertexCount, vbBindFlags, Buffer::CpuAccess::None, nullptr, false);
        mpPrevCurveVertexBuffer->setName("AnimatedVertexCache::mpPrevCurveVertexBuffer");

        // Initialize vertex buffers with cached positions.
        uint32_t offset = 0;
        for (size_t i = 0; i < mCachedCurves.size(); i++)
        {
            size_t vertexCount = mCachedCurves[i].vertexData[0].size();
            uint32_t bufSize = uint32_t(vertexCount * sizeof(DynamicCurveVertexData));
            uint32_t k = 0;
            const auto& timeSamples = mCachedCurves[i].timeSamples;

            for (uint32_t j = 0; j < mCurveKeyframeTimes.size(); j++)
            {
                while (k + 1 < timeSamples.size() && timeSamples[k] < mCurveKeyframeTimes[j]) k++;

                if (timeSamples[k] == mCurveKeyframeTimes[j])
                {
                    mpCurveVertexBuffers[j]->setBlob(mCachedCurves[i].vertexData[k].data(), offset, bufSize);
                }
                else
                {
                    // Linearly interpolate at the missing keyframe.
                    float t = float((mCurveKeyframeTimes[j] - timeSamples[k - 1]) / (timeSamples[k] - timeSamples[k - 1]));
                    std::vector<DynamicCurveVertexData> interpVertices(vertexCount);
                    for (size_t p = 0; p < vertexCount; p++)
                    {
                        interpVertices[p].position = (1.f - t) * mCachedCurves[i].vertexData[k - 1][p].position + t * mCachedCurves[i].vertexData[k][p].position;
                    }
                    mpCurveVertexBuffers[j]->setBlob(interpVertices.data(), offset, bufSize);
                }
            }

            // Initialize it with positions at the first keyframe.
            mpPrevCurveVertexBuffer->setBlob(mCachedCurves[i].vertexData[0].data(), offset, bufSize);

            offset += bufSize;
        }

        // Create curve index buffer.
        vbBindFlags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
        mpCurveIndexBuffer = Buffer::create(sizeof(uint32_t) * mCurveIndexCount, vbBindFlags);
        mpCurveIndexBuffer->setName("AnimatedVertexCache::mpCurveIndexBuffer");

        // Initialize index buffer.
        offset = 0;
        std::vector<uint32_t> indexData(mCurveIndexCount);
        for (uint32_t i = 0; i < (uint32_t)mCachedCurves.size(); i++)
        {
            for (size_t j = 0; j < mCachedCurves[i].indexData.size(); j++)
            {
                indexData[offset] = mpScene->getCurve(i).vbOffset;
                indexData[offset++] += mCachedCurves[i].indexData[j];
            }
        }
        mpCurveIndexBuffer->setBlob(indexData.data(), 0, mCurveIndexCount * sizeof(uint32_t));
    }

    void AnimatedVertexCache::createCurveVertexUpdatePass()
    {
        FALCOR_ASSERT(!mCachedCurves.empty());

        Program::DefineList defines;
        defines.add("CURVE_KEYFRAME_COUNT", std::to_string(mCurveKeyframeTimes.size()));
        mpCurveVertexUpdatePass = ComputePass::create(kUpdateCurveVerticesFilename, "main", defines);

        auto block = mpCurveVertexUpdatePass->getVars()["gCurveVertexUpdater"];
        auto var = block["curvePerKeyframe"];

        // Bind curve vertex data.
        for (uint32_t i = 0; i < mCurveKeyframeTimes.size(); i++) var[i]["vertexData"] = mpCurveVertexBuffers[i];
    }

    void AnimatedVertexCache::createCurveAABBUpdatePass()
    {
        FALCOR_ASSERT(!mCachedCurves.empty());

        mpCurveAABBUpdatePass = ComputePass::create(kUpdateCurveAABBsFilename);

        auto block = mpCurveAABBUpdatePass->getVars()["gCurveAABBUpdater"];
        block["curveIndexData"] = mpCurveIndexBuffer;
    }

    void AnimatedVertexCache::executeCurveVertexUpdatePass(RenderContext* pContext, const InterpolationInfo& info, bool copyPrev)
    {
        if (!mpCurveVertexUpdatePass) return;

        FALCOR_PROFILE("update curve vertices");

        auto block = mpCurveVertexUpdatePass->getVars()["gCurveVertexUpdater"];
        block["keyframeIndices"] = info.keyframeIndices;
        block["t"] = info.t;
        block["copyPrev"] = copyPrev;
        block["curveVertices"] = mpScene->mpCurveVao->getVertexBuffer(0);
        block["prevCurveVertices"] = mpPrevCurveVertexBuffer;

        uint32_t dimX = (1 << 16);
        uint32_t dimY = (uint32_t)std::ceil((float)mCurveVertexCount / dimX);
        block["dimX"] = dimX;
        block["vertexCount"] = mCurveVertexCount;

        mpCurveVertexUpdatePass->execute(pContext, dimX, dimY, 1);
    }

    void AnimatedVertexCache::executeCurveAABBUpdatePass(RenderContext* pContext)
    {
        if (!mpCurveAABBUpdatePass || !mpScene->mpRtAABBBuffer) return;

        FALCOR_PROFILE("update curve AABBs");

        auto block = mpCurveAABBUpdatePass->getVars()["gCurveAABBUpdater"];
        block["curveVertices"] = mpScene->mpCurveVao->getVertexBuffer(0);
        block["curveAABBs"].setUav(mpScene->mpRtAABBBuffer->getUAV(0, mCurveIndexCount));

        uint32_t dimX = (1 << 16);
        uint32_t dimY = (uint32_t)std::ceil((float)mCurveIndexCount / dimX);
        block["dimX"] = dimX;
        block["AABBCount"] = mCurveIndexCount;

        mpCurveAABBUpdatePass->execute(pContext, dimX, dimY, 1);
    }
}
