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
#include "AnimatedVertexCache.h"
#include "Animation.h"
#include "Core/API/RenderContext.h"
#include "Scene/Scene.h"
#include "Utils/Timing/Profiler.h"

namespace Falcor
{
    namespace
    {
        const std::string kUpdateCurveVerticesFilename = "Scene/Animation/UpdateCurveVertices.slang";
        const std::string kUpdateCurveAABBsFilename = "Scene/Animation/UpdateCurveAABBs.slang";
        const std::string kUpdateCurvePolyTubeVerticesFilename = "Scene/Animation/UpdateCurvePolyTubeVertices.slang";

        InterpolationInfo calculateInterpolation(double time, const std::vector<double>& timeSamples, Animation::Behavior preInfinityBehavior, Animation::Behavior postInfinityBehavior)
        {
            if (!std::isfinite(time))
            {
                return InterpolationInfo{ uint2(0), 0.f };
            }

            // Clamp to positive
            time = std::max(time, 0.0);

            // Post-Infinity Behavior
            if (time > timeSamples.back())
            {
                if (postInfinityBehavior == Animation::Behavior::Constant)
                {
                    time = timeSamples.back();
                }
                else if (postInfinityBehavior == Animation::Behavior::Cycle)
                {
                    time = std::fmod(time, timeSamples.back());
                }
            }

            uint2 keyframeIndices;
            float t = 0.0f;

            // Pre-Infinity Behavior
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
            // Regular Interpolation
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

    AnimatedVertexCache::AnimatedVertexCache(Scene* pScene, const Buffer::SharedPtr& pPrevVertexData, std::vector<CachedCurve>&& cachedCurves, std::vector<CachedMesh>&& cachedMeshes)
        : mpScene(pScene)
        , mCachedCurves(cachedCurves)
        , mCachedMeshes(cachedMeshes)
        , mpPrevVertexData(pPrevVertexData)
    {
        if (mCachedCurves.empty() && mCachedMeshes.empty()) return;

        if (!mCachedCurves.empty())
        {
            for (auto& cache : mCachedCurves)
            {
                if (cache.tessellationMode == CurveTessellationMode::LinearSweptSphere) mCurveLSSCount++;
                if (cache.tessellationMode == CurveTessellationMode::PolyTube) mCurvePolyTubeCount++;
            }

            initCurveKeyframes();

            if (mCurveLSSCount > 0)
            {
                bindCurveLSSBuffers();
                createCurveLSSVertexUpdatePass();
                createCurveLSSAABBUpdatePass();
            }

            if (mCurvePolyTubeCount > 0)
            {
                bindCurvePolyTubeBuffers();
                createCurvePolyTubeVertexUpdatePass();
            }


        }

        if (!mCachedMeshes.empty())
        {
            initMeshKeyframes();
            initMeshBuffers();

            createMeshVertexUpdatePass();
        }
    }

    AnimatedVertexCache::UniquePtr AnimatedVertexCache::create(Scene* pScene, const Buffer::SharedPtr& pPrevVertexData, std::vector<CachedCurve>&& cachedCurves, std::vector<CachedMesh>&& cachedMeshes)
    {
        return UniquePtr(new AnimatedVertexCache(pScene, pPrevVertexData, std::move(cachedCurves), std::move(cachedMeshes)));
    }

    bool AnimatedVertexCache::animate(RenderContext* pRenderContext, double time)
    {
        if (!hasAnimations()) return false;

        if (!mCachedCurves.empty())
        {
            double curveTime = mLoopAnimations ? std::fmod(time, mGlobalCurveAnimationLength) : time;
            InterpolationInfo interpolationInfo = calculateInterpolation(curveTime, mCurveKeyframeTimes, mPreInfinityBehavior, Animation::Behavior::Constant);

            if (mCurveLSSCount > 0)
            {
                executeCurveLSSVertexUpdatePass(pRenderContext, interpolationInfo);
                executeCurveLSSAABBUpdatePass(pRenderContext);
            }

            if (mCurvePolyTubeCount > 0)
            {
                executeCurvePolyTubeVertexUpdatePass(pRenderContext, interpolationInfo);
            }


        }

        if (!mCachedMeshes.empty())
        {
            executeMeshVertexUpdatePass(pRenderContext, time);
        }

        return true;
    }

    void AnimatedVertexCache::copyToPrevVertices(RenderContext* pRenderContext)
    {
        executeCurveLSSVertexUpdatePass(pRenderContext, InterpolationInfo{ uint2(0), 0.f }, true);
        executeCurvePolyTubeVertexUpdatePass(pRenderContext, InterpolationInfo{ uint2(0), 0.f }, true);

        executeMeshVertexUpdatePass(pRenderContext, 0.0f, true);
    }

    bool AnimatedVertexCache::hasAnimations() const
    {
        return hasMeshAnimations() || hasCurveAnimations();
    }

    uint64_t AnimatedVertexCache::getMemoryUsageInBytes() const
    {
        uint64_t m = 0;
        for (size_t i = 0; i < mpCurveVertexBuffers.size(); i++) m += mpCurveVertexBuffers[i] ? mpCurveVertexBuffers[i]->getSize() : 0;
        m += mpPrevCurveVertexBuffer ? mpPrevCurveVertexBuffer->getSize() : 0;
        m += mpCurveIndexBuffer ? mpCurveIndexBuffer->getSize() : 0;
        for (size_t i = 0; i < mpMeshVertexBuffers.size(); i++) m += mpMeshVertexBuffers[i] ? mpMeshVertexBuffers[i]->getSize() : 0;
        m += mpMeshInterpolationBuffer ? mpMeshInterpolationBuffer->getSize() : 0;
        m += mpMeshMetadataBuffer ? mpMeshMetadataBuffer->getSize() : 0;
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

    void AnimatedVertexCache::bindCurveLSSBuffers()
    {
        // Compute curve vertex and index (segment) count.
        mCurveVertexCount = 0;
        mCurveIndexCount = 0;
        for (uint32_t i = 0; i < mCachedCurves.size(); i++)
        {
            if (mCachedCurves[i].tessellationMode != CurveTessellationMode::LinearSweptSphere) continue;

            mCurveVertexCount += (uint32_t)mCachedCurves[i].vertexData[0].size();
            mCurveIndexCount += (uint32_t)mCachedCurves[i].indexData.size();
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
            if (mCachedCurves[i].tessellationMode != CurveTessellationMode::LinearSweptSphere) continue;

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
                        interpVertices[p].position = lerp(mCachedCurves[i].vertexData[k - 1][p].position, mCachedCurves[i].vertexData[k][p].position, t);
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
        for (CurveID curveID{ 0 }; curveID.get() < (uint32_t)mCachedCurves.size(); ++curveID)
        {
            if (mCachedCurves[curveID.get()].tessellationMode != CurveTessellationMode::LinearSweptSphere) continue;

            for (size_t j = 0; j < mCachedCurves[curveID.get()].indexData.size(); j++)
            {
                indexData[offset] = mpScene->getCurve(curveID).vbOffset;
                indexData[offset++] += mCachedCurves[curveID.get()].indexData[j];
            }
        }
        mpCurveIndexBuffer->setBlob(indexData.data(), 0, mCurveIndexCount * sizeof(uint32_t));
    }

    void AnimatedVertexCache::bindCurvePolyTubeBuffers()
    {
        std::vector<PerCurveMetadata> curveMetadata;
        curveMetadata.reserve(mCachedCurves.size());
        std::vector<PerMeshMetadata> meshMetadata;
        meshMetadata.reserve(mCachedCurves.size());

        // Compute curve vertex and index (segment) count.
        mCurvePolyTubeVertexCount = 0;
        mCurvePolyTubeIndexCount = 0;
        for (uint32_t i = 0; i < mCachedCurves.size(); i++)
        {
            const auto& cache = mCachedCurves[i];

            if (cache.tessellationMode != CurveTessellationMode::PolyTube) continue;

            PerCurveMetadata curveMeta;
            curveMeta.indexCount = (uint32_t)cache.indexData.size();
            curveMeta.indexOffset = mCurvePolyTubeIndexCount;
            curveMeta.vertexCount = (uint32_t)cache.vertexData[0].size();
            curveMeta.vertexOffset = mCurvePolyTubeVertexCount;
            curveMetadata.push_back(curveMeta);

            PerMeshMetadata meshMeta;
            meshMeta.vertexCount = mpScene->getMesh(MeshID{ cache.geometryID }).vertexCount;
            meshMeta.sceneVbOffset = mpScene->getMesh(MeshID{ cache.geometryID }).vbOffset;
            meshMeta.prevVbOffset = mpScene->getMesh(MeshID{ cache.geometryID }).prevVbOffset;
            meshMetadata.push_back(meshMeta);

            mMaxCurvePolyTubeVertexCount = std::max(curveMeta.vertexCount, mMaxCurvePolyTubeVertexCount);

            mCurvePolyTubeVertexCount += curveMeta.vertexCount;
            mCurvePolyTubeIndexCount += curveMeta.indexCount;
        }

        mpCurvePolyTubeCurveMetadataBuffer = Buffer::createStructured(sizeof(PerCurveMetadata), (uint32_t)curveMetadata.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, curveMetadata.data(), false);
        mpCurvePolyTubeCurveMetadataBuffer->setName("AnimatedVertexCache::mpCurvePolyTubeCurveMetadataBuffer");

        mpCurvePolyTubeMeshMetadataBuffer = Buffer::createStructured(sizeof(PerMeshMetadata), (uint32_t)meshMetadata.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, meshMetadata.data(), false);
        mpCurvePolyTubeMeshMetadataBuffer->setName("AnimatedVertexCache::mpCurvePolyTubeMeshMetadataBuffer");

        // Create buffers for vertex positions in curve vertex caches.
        ResourceBindFlags vbBindFlags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
        mpCurvePolyTubeVertexBuffers.resize(mCurveKeyframeTimes.size());
        for (uint32_t i = 0; i < mCurveKeyframeTimes.size(); i++)
        {
            mpCurvePolyTubeVertexBuffers[i] = Buffer::createStructured(sizeof(DynamicCurveVertexData), mCurvePolyTubeVertexCount, vbBindFlags, Buffer::CpuAccess::None, nullptr, false);
            mpCurvePolyTubeVertexBuffers[i]->setName("AnimatedVertexCache::mpCurvePolyTubeVertexBuffers[" + std::to_string(i) + "]");
        }

        // Initialize vertex buffers with cached positions.
        uint32_t offset = 0;
        for (size_t i = 0; i < mCachedCurves.size(); i++)
        {
            if (mCachedCurves[i].tessellationMode != CurveTessellationMode::PolyTube) continue;

            size_t vertexCount = mCachedCurves[i].vertexData[0].size();
            uint32_t bufSize = uint32_t(vertexCount * sizeof(DynamicCurveVertexData));
            uint32_t k = 0;
            const auto& timeSamples = mCachedCurves[i].timeSamples;

            for (uint32_t j = 0; j < mCurveKeyframeTimes.size(); j++)
            {
                while (k + 1 < timeSamples.size() && timeSamples[k] < mCurveKeyframeTimes[j]) k++;

                if (timeSamples[k] == mCurveKeyframeTimes[j])
                {
                    mpCurvePolyTubeVertexBuffers[j]->setBlob(mCachedCurves[i].vertexData[k].data(), offset, bufSize);
                }
                else
                {
                    // Linearly interpolate at the missing keyframe.
                    float t = float((mCurveKeyframeTimes[j] - timeSamples[k - 1]) / (timeSamples[k] - timeSamples[k - 1]));
                    std::vector<DynamicCurveVertexData> interpVertices(vertexCount);
                    for (size_t p = 0; p < vertexCount; p++)
                    {
                        interpVertices[p].position = lerp(mCachedCurves[i].vertexData[k - 1][p].position, mCachedCurves[i].vertexData[k][p].position, t);
                    }
                    mpCurvePolyTubeVertexBuffers[j]->setBlob(interpVertices.data(), offset, bufSize);
                }
            }

            offset += bufSize;
        }

        // Create curve strand index buffer.
        vbBindFlags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
        mpCurvePolyTubeStrandIndexBuffer = Buffer::create(sizeof(uint32_t) * mCurvePolyTubeVertexCount, vbBindFlags);
        mpCurvePolyTubeStrandIndexBuffer->setName("AnimatedVertexCache::mpCurvePolyTubeStrandIndexBuffer");

        // Initialize strand index buffer.
        offset = 0;
        const uint32_t strandLastVertexIndex = 0xffffffff;
        std::vector<uint32_t> strandIndexData(mCurvePolyTubeVertexCount);
        for (uint32_t i = 0; i < (uint32_t)mCachedCurves.size(); i++)
        {
            if (mCachedCurves[i].tessellationMode != CurveTessellationMode::PolyTube) continue;

            auto& cache = mCachedCurves[i];

            uint32_t strandIndex = 0;
            for (size_t j = 0; j < cache.indexData.size(); j++)
            {
                strandIndexData[offset++] = strandIndex++;

                if (j > 0 && j < cache.indexData.size() - 1)
                {
                    uint32_t curIndex = cache.indexData[j];
                    uint32_t nextIndex = cache.indexData[j + 1];

                    // This is how we detect a new strand in curve data - there is a jump in the index buffer.
                    if (nextIndex != curIndex + 1)
                    {
                        strandIndexData[offset++] = strandLastVertexIndex;
                    }
                }
            }

            strandIndexData[offset++] = strandLastVertexIndex;
        }

        FALCOR_ASSERT(offset == mCurvePolyTubeVertexCount)

        mpCurvePolyTubeStrandIndexBuffer->setBlob(strandIndexData.data(), 0, mCurvePolyTubeVertexCount * sizeof(uint32_t));
    }


    void AnimatedVertexCache::initMeshKeyframes()
    {
        for (const auto& cache : mCachedMeshes)
        {
            mGlobalMeshAnimationLength = std::max(mGlobalMeshAnimationLength, cache.timeSamples.back());
            mMeshKeyframeCount += (uint32_t)cache.timeSamples.size();
            mMaxMeshVertexCount = std::max((uint32_t)cache.vertexData.front().size(), mMaxMeshVertexCount);
        }
    }

    void AnimatedVertexCache::initMeshBuffers()
    {
        mpMeshVertexBuffers.resize(mMeshKeyframeCount);
        std::vector<PerMeshMetadata> meshMetadata;
        meshMetadata.reserve(mCachedMeshes.size());

        uint32_t keyframeOffset = 0;
        for (auto& cache : mCachedMeshes)
        {
            FALCOR_ASSERT(cache.vertexData.front().size() == mpScene->getMesh(cache.meshID).vertexCount);

            PerMeshMetadata meta;
            meta.keyframeBufferOffset = keyframeOffset;
            meta.vertexCount = (uint32_t)cache.vertexData.front().size();
            meta.sceneVbOffset = mpScene->getMesh(cache.meshID).vbOffset;
            meta.prevVbOffset = mpScene->getMesh(cache.meshID).prevVbOffset;
            meshMetadata.push_back(meta);

            // Create vertex buffer for each keyframe on this mesh
            for (size_t i = 0; i < cache.vertexData.size(); i++)
            {
                auto& data = cache.vertexData[i];
                size_t index = keyframeOffset + i;
                mpMeshVertexBuffers[index] = Buffer::createStructured(sizeof(PackedStaticVertexData), (uint32_t)data.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, data.data(), false);
                mpMeshVertexBuffers[index]->setName("AnimatedVertexCache::mpMeshVertexBuffers[" + std::to_string(index) + "]");
            }

            keyframeOffset += (uint32_t)cache.timeSamples.size();
        }

        mpMeshMetadataBuffer = Buffer::createStructured(sizeof(PerMeshMetadata), (uint32_t)meshMetadata.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, meshMetadata.data(), false);
        mpMeshMetadataBuffer->setName("AnimatedVertexCache::mpMeshMetadataBuffer");

        mMeshInterpolationInfo.resize(mCachedMeshes.size());
        mpMeshInterpolationBuffer = Buffer::createStructured(sizeof(InterpolationInfo), (uint32_t)mMeshInterpolationInfo.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
        mpMeshInterpolationBuffer->setName("AnimatedVertexCache::mpMeshInterpolationbuffer");
    }

    void AnimatedVertexCache::createMeshVertexUpdatePass()
    {
        FALCOR_ASSERT(!mCachedMeshes.empty());

        Program::DefineList defines;
        defines.add("MESH_KEYFRAME_COUNT", std::to_string(mMeshKeyframeCount));
        mpMeshVertexUpdatePass = ComputePass::create("Scene/Animation/UpdateMeshVertices.slang", "main", defines);

        // Bind data
        auto block = mpMeshVertexUpdatePass->getVars()["gMeshVertexUpdater"];
        auto keyframesVar = block["meshPerKeyframe"];
        for (size_t i = 0; i < mpMeshVertexBuffers.size(); i++) keyframesVar[i]["vertexData"] = mpMeshVertexBuffers[i];

        block["perMeshInterp"] = mpMeshInterpolationBuffer;
        block["perMeshData"] = mpMeshMetadataBuffer;
        block["prevVertexData"] = mpPrevVertexData;
    }

    void AnimatedVertexCache::createCurveLSSVertexUpdatePass()
    {
        FALCOR_ASSERT(mCurveLSSCount > 0);

        Program::DefineList defines;
        defines.add("CURVE_KEYFRAME_COUNT", std::to_string(mCurveKeyframeTimes.size()));
        mpCurveVertexUpdatePass = ComputePass::create(kUpdateCurveVerticesFilename, "main", defines);

        auto block = mpCurveVertexUpdatePass->getVars()["gCurveVertexUpdater"];
        auto var = block["curvePerKeyframe"];

        // Bind curve vertex data.
        for (uint32_t i = 0; i < mCurveKeyframeTimes.size(); i++) var[i]["vertexData"] = mpCurveVertexBuffers[i];
    }

    void AnimatedVertexCache::createCurveLSSAABBUpdatePass()
    {
        FALCOR_ASSERT(mCurveLSSCount > 0);

        mpCurveAABBUpdatePass = ComputePass::create(kUpdateCurveAABBsFilename);

        auto block = mpCurveAABBUpdatePass->getVars()["gCurveAABBUpdater"];
        block["curveIndexData"] = mpCurveIndexBuffer;
    }

    void AnimatedVertexCache::createCurvePolyTubeVertexUpdatePass()
    {
        FALCOR_ASSERT(mCurvePolyTubeCount > 0);

        Program::DefineList defines;
        defines.add("CURVE_KEYFRAME_COUNT", std::to_string(mCurveKeyframeTimes.size()));
        mpCurvePolyTubeVertexUpdatePass = ComputePass::create(kUpdateCurvePolyTubeVerticesFilename, "main", defines);

        auto block = mpCurvePolyTubeVertexUpdatePass->getVars()["gCurvePolyTubeVertexUpdater"];
        block["perCurveData"] = mpCurvePolyTubeCurveMetadataBuffer;
        block["curveStrandIndexData"] = mpCurvePolyTubeStrandIndexBuffer;

        auto var = block["curvePerKeyframe"];

        // Bind curve vertex data.
        for (uint32_t i = 0; i < mCurveKeyframeTimes.size(); i++) var[i]["vertexData"] = mpCurvePolyTubeVertexBuffers[i];
    }


    void AnimatedVertexCache::executeMeshVertexUpdatePass(RenderContext* pRenderContext, double t, bool copyPrev)
    {
        if (!mpMeshVertexUpdatePass) return;

        FALCOR_PROFILE("update mesh vertices");

        // Update interpolation
        for (size_t i = 0; i < mMeshInterpolationInfo.size(); i++)
        {
            auto postInfinityBehavior = mLoopAnimations ? Animation::Behavior::Cycle : Animation::Behavior::Constant;
            mMeshInterpolationInfo[i] = calculateInterpolation(t, mCachedMeshes[i].timeSamples, mPreInfinityBehavior, postInfinityBehavior);
        }

        mpMeshInterpolationBuffer->setBlob(mMeshInterpolationInfo.data(), 0, mpMeshInterpolationBuffer->getSize());

        auto block = mpMeshVertexUpdatePass->getVars()["gMeshVertexUpdater"];
        block["sceneVertexData"] = mpScene->getMeshVao()->getVertexBuffer(Scene::kStaticDataBufferIndex);
        block["copyPrev"] = copyPrev;

        mpMeshVertexUpdatePass->execute(pRenderContext, mMaxMeshVertexCount, (uint32_t)mCachedMeshes.size(), 1);
    }

    void AnimatedVertexCache::executeCurveLSSVertexUpdatePass(RenderContext* pRenderContext, const InterpolationInfo& info, bool copyPrev)
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

        mpCurveVertexUpdatePass->execute(pRenderContext, dimX, dimY, 1);
    }

    void AnimatedVertexCache::executeCurveLSSAABBUpdatePass(RenderContext* pRenderContext)
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

        mpCurveAABBUpdatePass->execute(pRenderContext, dimX, dimY, 1);
    }

    void AnimatedVertexCache::executeCurvePolyTubeVertexUpdatePass(RenderContext* pRenderContext, const InterpolationInfo& info, bool copyPrev)
    {
        if (!mpCurvePolyTubeVertexUpdatePass) return;

        FALCOR_PROFILE("Update curve poly-tube vertices");

        auto block = mpCurvePolyTubeVertexUpdatePass->getVars()["gCurvePolyTubeVertexUpdater"];
        block["keyframeIndices"] = info.keyframeIndices;
        block["t"] = info.t;
        block["copyPrev"] = copyPrev;

        block["perMeshData"] = mpCurvePolyTubeMeshMetadataBuffer;
        block["sceneVertexData"] = mpScene->getMeshVao()->getVertexBuffer(Scene::kStaticDataBufferIndex);
        block["prevVertexData"] = mpPrevVertexData;

        block["vertexCount"] = mCurvePolyTubeVertexCount;
        block["indexCount"] = mCurvePolyTubeIndexCount;

        mpCurvePolyTubeVertexUpdatePass->execute(pRenderContext, mMaxCurvePolyTubeVertexCount * 4, mCurvePolyTubeCount, 1);
    }


}
