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
#include "AnimatedVertexCache.h"
#include "Core/Macros.h"
#include "Core/API/Buffer.h"
#include "Utils/Math/Matrix.h"
#include "RenderGraph/BasePasses/ComputePass.h"
#include "Scene/SceneTypes.slang"
#include <memory>
#include <vector>

namespace Falcor
{
    class Scene;

    class FALCOR_API AnimationController
    {
    public:
        using UniquePtr = std::unique_ptr<AnimationController>;
        using UniqueConstPtr = std::unique_ptr<const AnimationController>;
        static const uint32_t kInvalidBoneID = -1;
        ~AnimationController() = default;

        using StaticVertexVector = std::vector<PackedStaticVertexData>;
        using SkinningVertexVector = std::vector<SkinningVertexData>;

        /** Create a new object.
            \return A new object, or throws an exception if creation failed.
        */
        static UniquePtr create(Scene* pScene, const StaticVertexVector& staticVertexData, const SkinningVertexVector& skinningVertexData, uint32_t prevVertexCount, const std::vector<Animation::SharedPtr>& animations);

        /** Add animated vertex caches (curves and meshes) to the controller.
        */
        void addAnimatedVertexCaches(std::vector<CachedCurve>&& cachedCurves, std::vector<CachedMesh>&& cachedMeshes, const StaticVertexVector& staticVertexData);

        /** Returns true if controller contains animations.
        */
        bool hasAnimations() const { return mAnimations.size() > 0 || hasAnimatedVertexCaches(); }

        /** Returns true if controller is handling any skinned meshes.
        */
        bool hasSkinnedMeshes() const { return mpSkinningPass != nullptr; }

        /** Returns true if controller contains any animated vertex caches.
        */
        bool hasAnimatedVertexCaches() const { return hasAnimatedCurveCaches() || hasAnimatedMeshCaches(); }

        /** Returns true if controller contains animated curve caches.
        */
        bool hasAnimatedCurveCaches() const { return mpVertexCache && mpVertexCache->hasCurveAnimations(); }

        /** Returns true if controller contains animated curve caches.
        */
        bool hasAnimatedMeshCaches() const { return mpVertexCache && mpVertexCache->hasMeshAnimations(); }

        /** Returns a list of all animations.
        */
        std::vector<Animation::SharedPtr>& getAnimations() { return mAnimations; }

        /** Enable/disable animations.
        */
        void setEnabled(bool enabled);

        /** Returns true if animations are enabled.
        */
        bool isEnabled() const { return mEnabled; };

        /** Enable/disable globally looping animations.
        */
        void setIsLooped(bool looped);

        /** Returns true if animations are currently globally looped.
        */
        bool isLooped() { return mLoopAnimations; }

        /** Mark a scene node as being edited externally.
            Ensures that all global matrices depending on this scene node are updated.
        */
        void setNodeEdited(size_t nodeID) { mNodesEdited[nodeID] = true; }

        /** Run the animation system.
            \return true if a change occurred, otherwise false.
        */
        bool animate(RenderContext* pContext, double currentTime);

        /** Check if a matrix changed since last frame.
        */
        bool isMatrixChanged(NodeID matrixID) const { return mMatricesChanged[matrixID.get()]; }

        /** Get the local matrices.
            These represent the current local transform for each scene graph node.
        */
        const std::vector<float4x4>& getLocalMatrices() const { return mLocalMatrices; }

        /** Get the global matrices.
            These represent the current object-to-world space transform for each scene graph node.
        */
        const std::vector<float4x4>& getGlobalMatrices() const { return mGlobalMatrices; }

        /** Get the transposed inverse global matrices.
        */
        const std::vector<float4x4>& getInvTransposeGlobalMatrices() const { return mInvTransposeGlobalMatrices; }

        /** Render the UI.
        */
        void renderUI(Gui::Widgets& widget);

        /** Get the previous vertex data buffer for dynamic meshes.
            \return Buffer containing the previous vertex data, or nullptr if no dynamic meshes exist.
        */
        Buffer::SharedPtr getPrevVertexData() const { return mpPrevVertexData; }

        /** Get the previous curve vertex data buffer for dynamic curves.
            \return Buffer containing the previous curve vertex data, or nullptr if no dynamic curves exist.
        */
        Buffer::SharedPtr getPrevCurveVertexData() const { return mpVertexCache ? mpVertexCache->getPrevCurveVertexData() : nullptr; }

        /** Get the total GPU memory usage in bytes.
        */
        uint64_t getMemoryUsageInBytes() const;

    private:
        friend class SceneBuilder;
        AnimationController(Scene* pScene, const StaticVertexVector& staticVertexData, const SkinningVertexVector& skinningVertexData, uint32_t prevVertexCount, const std::vector<Animation::SharedPtr>& animations);

        void initLocalMatrices();
        void updateLocalMatrices(double time);
        void updateWorldMatrices(bool updateAll = false);
        void uploadWorldMatrices(bool uploadAll = false);

        void bindBuffers();

        void createSkinningPass(const std::vector<PackedStaticVertexData>& staticVertexData, const SkinningVertexVector& skinningVertexData);
        void executeSkinningPass(RenderContext* pContext, bool initPrev = false);

        // Animation
        std::vector<Animation::SharedPtr> mAnimations;
        std::vector<bool> mNodesEdited;
        std::vector<float4x4> mLocalMatrices;
        std::vector<float4x4> mGlobalMatrices;
        std::vector<float4x4> mInvTransposeGlobalMatrices;
        std::vector<bool> mMatricesChanged;         ///< Flag per matrix, true if matrix changed since last frame.

        bool mFirstUpdate = true;       ///< True if this is the first update.
        bool mEnabled = true;           ///< True if animations are enabled.
        bool mPrevEnabled = false;      ///< True if animations were enabled in previous frame.
        double mTime = 0.0;             ///< Global time of current frame.
        double mPrevTime = 0.0;         ///< Global time of previous frame.

        bool mLoopAnimations = true;
        double mGlobalAnimationLength = 0;
        Scene* mpScene = nullptr;

        Buffer::SharedPtr mpWorldMatricesBuffer;
        Buffer::SharedPtr mpPrevWorldMatricesBuffer;
        Buffer::SharedPtr mpInvTransposeWorldMatricesBuffer;
        Buffer::SharedPtr mpPrevInvTransposeWorldMatricesBuffer;

        // Skinning
        ComputePass::SharedPtr mpSkinningPass;
        std::vector<float4x4> mMeshBindMatrices; // Optimization TODO: These are only needed per mesh
        std::vector<float4x4> mSkinningMatrices;
        std::vector<float4x4> mInvTransposeSkinningMatrices;
        uint32_t mSkinningDispatchSize = 0;

        Buffer::SharedPtr mpMeshBindMatricesBuffer;
        Buffer::SharedPtr mpMeshInvBindMatricesBuffer;
        Buffer::SharedPtr mpSkinningMatricesBuffer;
        Buffer::SharedPtr mpInvTransposeSkinningMatricesBuffer;
        Buffer::SharedPtr mpStaticVertexData;
        Buffer::SharedPtr mpSkinningVertexData;
        Buffer::SharedPtr mpPrevVertexData;

        // Animated vertex caches
        AnimatedVertexCache::UniquePtr mpVertexCache;
    };
}
