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
#pragma once
#include "Animation.h"
#include "RenderGraph/BasePasses/ComputePass.h"
#include "Scene/SceneTypes.slang"

namespace Falcor
{
    class Scene;
    class Model;

    struct CachedCurve
    {
        std::vector<double> timeSamples;

        // Shared among all frames.
        // We assume the topology doesn't change during animation.
        std::vector<uint32_t> indexData;

        // vertexData[i][j] represents at the i-th keyframe, the cache data of the j-th vertex.
        std::vector<std::vector<DynamicCurveVertexData>> vertexData;
    };

    struct CachedMesh
    {
        uint32_t meshId = -1; ///< ID of the mesh this data is animating

        std::vector<double> timeSamples;

        // vertexData[i][j] represents at the i-th keyframe, the cache data of the j-th vertex.
        std::vector<std::vector<PackedStaticVertexData>> vertexData;
    };

    struct InterpolationInfo
    {
        uint2 keyframeIndices;
        float t;
    };

    class dlldecl AnimatedVertexCache
    {
    public:
        using UniquePtr = std::unique_ptr<AnimatedVertexCache>;
        using UniqueConstPtr = std::unique_ptr<const AnimatedVertexCache>;
        ~AnimatedVertexCache() = default;

        static UniquePtr create(Scene* pScene, std::vector<CachedCurve>& cachedCurves, std::vector<CachedMesh>& cachedMeshes);

        bool animate(RenderContext* pContext, double time);

        void copyToPrevVertices(RenderContext* pContext);

        bool hasAnimations();

        double getGlobalAnimationLength() const { return mGlobalAnimationLength; }

        uint64_t getMemoryUsageInBytes() const;

        Buffer::SharedPtr getPrevCurveVertexData() const { return mpPrevCurveVertexBuffer; }

    private:
        AnimatedVertexCache(Scene* pScene, std::vector<CachedCurve>& cachedCurves, std::vector<CachedMesh>& cachedMeshes);

        void initKeyframes();
        void bindCurveBuffers();

        void createVertexUpdatePass();
        void createAABBUpdatePass();

        // Interpolate vertex positions.
        void executeVertexUpdatePass(RenderContext* pContext, uint2 keyframeIndices, float t, bool copyPrev = false);

        // Update the AABBs of procedural primitives (such as curve segments).
        void executeAABBUpdatePass(RenderContext* pContext);

        double mGlobalAnimationLength = 0;
        Scene* mpScene = nullptr;

        ComputePass::SharedPtr mpVertexUpdatePass;
        ComputePass::SharedPtr mpAABBUpdatePass;

        // Manage keyframes.
        uint32_t mKeyframeCount = 0;
        std::vector<double> mKeyframeTimes;

        // Cached curve animation.
        std::vector<CachedCurve> mCachedCurves;
        uint32_t mCurveVertexCount = 0;
        uint32_t mCurveIndexCount = 0;
        uint32_t mCurveAABBOffset = 0;

        std::vector<Buffer::SharedPtr> mpCurveVertexBuffers;
        Buffer::SharedPtr mpPrevCurveVertexBuffer;
        Buffer::SharedPtr mpCurveIndexBuffer;

        // TODO: Add support for cached mesh vertex animation below.
        // ...
    };
}
