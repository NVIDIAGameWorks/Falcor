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
#pragma once
#include "Animation.h"
#include "SharedTypes.slang"
#include "Core/API/Buffer.h"
#include "Scene/Curves/CurveConfig.h"
#include "Scene/SceneTypes.slang"
#include "Scene/SceneIDs.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "RenderGraph/BasePasses/ComputePass.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

namespace Falcor
{
    class Scene;

    struct CachedCurve
    {
        static const uint32_t kInvalidID = std::numeric_limits<uint32_t>::max();

        CurveTessellationMode tessellationMode = CurveTessellationMode::LinearSweptSphere;  ///< Curve tessellation mode.
        CurveOrMeshID geometryID{ CurveOrMeshID::kInvalidID };                              ///< ID of the curve or mesh this data is animating.

        std::vector<double> timeSamples;

        // Shared among all frames.
        // We assume the topology doesn't change during animation.
        std::vector<uint32_t> indexData;

        // vertexData[i][j] represents at the i-th keyframe, the cache data of the j-th vertex.
        std::vector<std::vector<DynamicCurveVertexData>> vertexData;
    };

    struct CachedMesh
    {
        MeshID meshID{ MeshID::kInvalidID }; ///< ID of the mesh this data is animating.

        std::vector<double> timeSamples;

        // vertexData[i][j] represents at the i-th keyframe, the cache data of the j-th vertex.
        std::vector<std::vector<PackedStaticVertexData>> vertexData;
    };

    class FALCOR_API AnimatedVertexCache
    {
    public:
        using UniquePtr = std::unique_ptr<AnimatedVertexCache>;
        using UniqueConstPtr = std::unique_ptr<const AnimatedVertexCache>;
        ~AnimatedVertexCache() = default;

        static UniquePtr create(Scene* pScene, const Buffer::SharedPtr& pPrevVertexData, std::vector<CachedCurve>&& cachedCurves, std::vector<CachedMesh>&& cachedMeshes);

        void setIsLooped(bool looped) { mLoopAnimations = looped; }

        bool isLooped() const { return mLoopAnimations; };

        void setPreInfinityBehavior(Animation::Behavior behavior) { mPreInfinityBehavior = behavior; }

        bool hasAnimations() const;

        bool hasCurveAnimations() const { return !mCurveKeyframeTimes.empty(); }

        bool hasMeshAnimations() const { return !mCachedMeshes.empty(); }

        double getGlobalAnimationLength() const { return std::max(mGlobalCurveAnimationLength, mGlobalMeshAnimationLength); }

        bool animate(RenderContext* pContext, double time);

        void copyToPrevVertices(RenderContext* pContext);

        Buffer::SharedPtr getPrevCurveVertexData() const { return mpPrevCurveVertexBuffer; }

        uint64_t getMemoryUsageInBytes() const;

    private:
        AnimatedVertexCache(Scene* pScene, const Buffer::SharedPtr& pPrevVertexData, std::vector<CachedCurve>&& cachedCurves, std::vector<CachedMesh>&& cachedMeshes);

        void initCurveKeyframes();
        void bindCurveLSSBuffers();
        void bindCurvePolyTubeBuffers();

        void createCurveLSSVertexUpdatePass();
        void createCurveLSSAABBUpdatePass();
        void createCurvePolyTubeVertexUpdatePass();

        void initMeshKeyframes();
        void initMeshBuffers();

        void createMeshVertexUpdatePass();

        void executeMeshVertexUpdatePass(RenderContext* pContext, double t, bool copyPrev = false);

        // Interpolate vertex positions.
        // When copyPrev is set to true, interpolation info is ignored and we just copy the current vertex data to the previous data.
        void executeCurveLSSVertexUpdatePass(RenderContext* pContext, const InterpolationInfo& info, bool copyPrev = false);

        // Update the AABBs of procedural primitives (such as curve segments).
        void executeCurveLSSAABBUpdatePass(RenderContext* pContext);

        void executeCurvePolyTubeVertexUpdatePass(RenderContext* pContext, const InterpolationInfo& info, bool copyPrev = false);


        bool mLoopAnimations = true;
        double mGlobalCurveAnimationLength = 0;
        double mGlobalMeshAnimationLength = 0;
        Scene* mpScene = nullptr;
        Buffer::SharedPtr mpPrevVertexData; ///< Owned by AnimationController
        Animation::Behavior mPreInfinityBehavior = Animation::Behavior::Constant; // How the animation behaves before the first keyframe.

        std::vector<CachedCurve> mCachedCurves;
        uint32_t mCurveLSSCount = 0;
        uint32_t mCurvePolyTubeCount = 0;
        std::vector<double> mCurveKeyframeTimes;

        // Cached curve (LSS) animation.
        ComputePass::SharedPtr mpCurveVertexUpdatePass;
        ComputePass::SharedPtr mpCurveAABBUpdatePass;

        uint32_t mCurveVertexCount = 0;
        uint32_t mCurveIndexCount = 0;
        uint32_t mCurveAABBOffset = 0;

        std::vector<Buffer::SharedPtr> mpCurveVertexBuffers;
        Buffer::SharedPtr mpPrevCurveVertexBuffer;
        Buffer::SharedPtr mpCurveIndexBuffer;

        // Cached curve (poly-tube mesh) animation.
        ComputePass::SharedPtr mpCurvePolyTubeVertexUpdatePass;

        uint32_t mCurvePolyTubeVertexCount = 0;
        uint32_t mCurvePolyTubeIndexCount = 0;
        uint32_t mMaxCurvePolyTubeVertexCount = 0; ///< Greatest vertex count a curve has

        std::vector<Buffer::SharedPtr> mpCurvePolyTubeVertexBuffers;
        Buffer::SharedPtr mpCurvePolyTubeStrandIndexBuffer;
        Buffer::SharedPtr mpCurvePolyTubeCurveMetadataBuffer;
        Buffer::SharedPtr mpCurvePolyTubeMeshMetadataBuffer;


        // Cached mesh animations
        ComputePass::SharedPtr mpMeshVertexUpdatePass;

        std::vector<CachedMesh> mCachedMeshes;
        std::vector<InterpolationInfo> mMeshInterpolationInfo;
        uint32_t mMeshKeyframeCount = 0; ///< Total count of all keyframes for all meshes
        uint32_t mMaxMeshVertexCount = 0; ///< Greatest vertex count a mesh has

        std::vector<Buffer::SharedPtr> mpMeshVertexBuffers;
        Buffer::SharedPtr mpMeshInterpolationBuffer;
        Buffer::SharedPtr mpMeshMetadataBuffer;
    };
}
