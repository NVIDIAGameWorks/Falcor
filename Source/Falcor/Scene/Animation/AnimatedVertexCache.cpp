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
    AnimatedVertexCache::AnimatedVertexCache(Scene* pScene, std::vector<CachedCurve>& cachedCurves, std::vector<CachedMesh>& cachedMeshes)
        : mpScene(pScene)
        , mCachedCurves(cachedCurves.size())
    {
        if (cachedCurves.empty()) return;

        for (size_t i = 0; i < cachedCurves.size(); i++)
        {
            auto& srcCache = cachedCurves[i];
            auto& dstCache = mCachedCurves[i];
            dstCache.timeSamples = std::move(srcCache.timeSamples);
            dstCache.indexData = std::move(srcCache.indexData);
            dstCache.vertexData.resize(srcCache.vertexData.size());
            for (size_t j = 0; j < srcCache.vertexData.size(); j++) dstCache.vertexData[j] = std::move(srcCache.vertexData[j]);
        }

        initKeyframes();
        bindCurveBuffers();

        createVertexUpdatePass();
        createAABBUpdatePass();
    }

    AnimatedVertexCache::UniquePtr AnimatedVertexCache::create(Scene* pScene, std::vector<CachedCurve>& cachedCurves, std::vector<CachedMesh>& cachedMeshes)
    {
        return UniquePtr(new AnimatedVertexCache(pScene, cachedCurves, cachedMeshes));
    }

    bool AnimatedVertexCache::animate(RenderContext* pContext, double time)
    {
        if (!hasAnimations()) return false;

        if (time < mKeyframeTimes.front() || time > mKeyframeTimes.back())
        {
            time = clamp(time, mKeyframeTimes.front(), mKeyframeTimes.back());
        }

        uint2 keyframeIndices;
        float t;
        if (time <= mKeyframeTimes.front())
        {
            keyframeIndices = uint2(0);
            t = 0.f;
        }
        else
        {
            keyframeIndices.y = uint32_t(std::lower_bound(mKeyframeTimes.begin(), mKeyframeTimes.end(), time) - mKeyframeTimes.begin());
            keyframeIndices.x = keyframeIndices.y - 1;
            t = (float)((time - mKeyframeTimes[keyframeIndices.x]) / (mKeyframeTimes[keyframeIndices.y] - mKeyframeTimes[keyframeIndices.x]));
        }

        executeVertexUpdatePass(pContext, keyframeIndices, t);
        executeAABBUpdatePass(pContext);

        return true;
    }

    void AnimatedVertexCache::copyToPrevVertices(RenderContext* pContext)
    {
        executeVertexUpdatePass(pContext, uint2(0), 0.f, true);
    }

    bool AnimatedVertexCache::hasAnimations()
    {
        return mGlobalAnimationLength > 0 && mKeyframeCount > 1;
    }

    uint64_t AnimatedVertexCache::getMemoryUsageInBytes() const
    {
        uint64_t m = 0;
        for (size_t i = 0; i < mpCurveVertexBuffers.size(); i++) m += mpCurveVertexBuffers[i] ? mpCurveVertexBuffers[i]->getSize() : 0;
        m += mpPrevCurveVertexBuffer ? mpPrevCurveVertexBuffer->getSize() : 0;
        m += mpCurveIndexBuffer ? mpCurveIndexBuffer->getSize() : 0;
        return m;
    }

    void AnimatedVertexCache::initKeyframes()
    {
        // Align the time samples across vertex caches.
        mKeyframeTimes.clear();
        for (size_t i = 0; i < mCachedCurves.size(); i++)
        {
            mKeyframeTimes.insert(mKeyframeTimes.end(), mCachedCurves[i].timeSamples.begin(), mCachedCurves[i].timeSamples.end());
        }
        std::sort(mKeyframeTimes.begin(), mKeyframeTimes.end());
        mKeyframeTimes.erase(std::unique(mKeyframeTimes.begin(), mKeyframeTimes.end()), mKeyframeTimes.end());

        mKeyframeCount = (uint32_t)mKeyframeTimes.size();
        mGlobalAnimationLength = mKeyframeTimes.back();
    }

    void AnimatedVertexCache::bindCurveBuffers()
    {
        // Compute curve vertex and index (segment) count.
        mCurveVertexCount = 0;
        mCurveIndexCount = 0;
        for (uint32_t i = 0; i < mpScene->getCurveCount(); i++)
        {
            assert(mpScene->getCurve(i).vertexCount == (uint32_t)mCachedCurves[i].vertexData[0].size());
            assert(mpScene->getCurve(i).indexCount == (uint32_t)mCachedCurves[i].indexData.size());
            mCurveVertexCount += mpScene->getCurve(i).vertexCount;
            mCurveIndexCount += mpScene->getCurve(i).indexCount;
        }

        // Create buffers for vertex positions in curve vertex caches.
        ResourceBindFlags vbBindFlags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
        mpCurveVertexBuffers.resize(mKeyframeCount);
        for (uint32_t i = 0; i < mKeyframeCount; i++)
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

            for (uint32_t j = 0; j < mKeyframeCount; j++)
            {
                while (k + 1 < timeSamples.size() && timeSamples[k] < mKeyframeTimes[j]) k++;

                if (timeSamples[k] == mKeyframeTimes[j])
                {
                    mpCurveVertexBuffers[j]->setBlob(mCachedCurves[i].vertexData[k].data(), offset, bufSize);
                }
                else
                {
                    // Linearly interpolate at the missing keyframe.
                    float t = float((mKeyframeTimes[j] - timeSamples[k - 1]) / (timeSamples[k] - timeSamples[k - 1]));
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

        // AABBs of curves are after AABBs of custom primitives.
        mCurveAABBOffset = mpScene->getCustomPrimitiveCount();
    }

    void AnimatedVertexCache::createVertexUpdatePass()
    {
        if (!hasAnimations()) return;

        Program::DefineList defines;
        defines.add("CURVE_KEYFRAME_COUNT", std::to_string(mKeyframeCount));
        mpVertexUpdatePass = ComputePass::create("Scene/Animation/UpdateVertices.slang", "main", defines);

        auto block = mpVertexUpdatePass->getVars()["gCurveVertexUpdater"];
        auto var = block["curvePerKeyframe"];

        // Bind curve vertex data.
        for (uint32_t i = 0; i < mKeyframeCount; i++) var[i]["vertexData"] = mpCurveVertexBuffers[i];
        block["prevCurveVertices"] = mpPrevCurveVertexBuffer;
    }

    void AnimatedVertexCache::createAABBUpdatePass()
    {
        if (!hasAnimations()) return;

        mpAABBUpdatePass = ComputePass::create("Scene/Animation/UpdateAABBs.slang");
        auto block = mpAABBUpdatePass->getVars()["gCurveAABBUpdater"];
        block["curveIndexData"] = mpCurveIndexBuffer;
    }

    void AnimatedVertexCache::executeVertexUpdatePass(RenderContext* pContext, uint2 keyframeIndices, float t, bool copyPrev)
    {
        PROFILE("update vertices");
        if (!mpVertexUpdatePass) return;

        auto block = mpVertexUpdatePass->getVars()["gCurveVertexUpdater"];
        block["keyframeIndices"] = keyframeIndices;
        block["t"] = t;
        block["copyPrev"] = copyPrev;
        block["curveVertices"] = mpScene->mpCurveVao->getVertexBuffer(0);

        uint32_t dimX = (1 << 16);
        uint32_t dimY = (uint32_t)std::ceil((float)mCurveVertexCount / dimX);
        block["dimX"] = dimX;
        block["vertexCount"] = mCurveVertexCount;

        mpVertexUpdatePass->execute(pContext, dimX, dimY, 1);
    }

    void AnimatedVertexCache::executeAABBUpdatePass(RenderContext* pContext)
    {
        PROFILE("update AABBs");
        if (!mpAABBUpdatePass || !mpScene->mpRtAABBBuffer) return;

        auto block = mpAABBUpdatePass->getVars()["gCurveAABBUpdater"];
        block["curveVertices"] = mpScene->mpCurveVao->getVertexBuffer(0);
        block["curveAABBs"].setUav(mpScene->mpRtAABBBuffer->getUAV(mCurveAABBOffset, mCurveIndexCount));

        uint32_t dimX = (1 << 16);
        uint32_t dimY = (uint32_t)std::ceil((float)mCurveIndexCount / dimX);
        block["dimX"] = dimX;
        block["AABBCount"] = mCurveIndexCount;

        mpAABBUpdatePass->execute(pContext, dimX, dimY, 1);
    }
}
