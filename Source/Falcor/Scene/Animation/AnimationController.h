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
#pragma once
#include "Animation.h"
#include "RenderGraph/BasePasses/ComputePass.h"
#include "Scene/SceneTypes.slang"

namespace Falcor
{
    class Scene;

    struct Bone
    {
        uint32_t parentID;
        uint32_t boneID;
        std::string name;
        glm::mat4 offset;
        glm::mat4 localTransform;
        glm::mat4 originalLocalTransform;
        glm::mat4 globalTransform;
    };

    class Model;
    class AssimpModelImporter;

    class dlldecl AnimationController
    {
    public:
        using UniquePtr = std::unique_ptr<AnimationController>;
        using UniqueConstPtr = std::unique_ptr<const AnimationController>;
        static const uint32_t kInvalidBoneID = -1;
        ~AnimationController() = default;

        using StaticVertexVector = std::vector<PackedStaticVertexData>;
        using DynamicVertexVector = std::vector<DynamicVertexData>;

        /** Create a new object.
            \return A new object, or throws an exception if creation failed.
        */
        static UniquePtr create(Scene* pScene, const StaticVertexVector& staticVertexData, const DynamicVertexVector& dynamicVertexData, const std::vector<Animation::SharedPtr>& animations);

        /** Returns true if controller contains animations.
        */
        bool hasAnimations() const { return mAnimations.size() > 0; }

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
        void setIsLooped(bool looped) { mLoopAnimations = looped; }

        /** Returns true if animations are currently globally looped.
        */
        bool isLooped() { return mLoopAnimations; }

        /** Run the animation
            \return true if a change occurred, otherwise false
        */
        bool animate(RenderContext* pContext, double currentTime);

        /** Check if a matrix is animated.
        */
        bool isMatrixAnimated(size_t matrixID) const { return mMatricesAnimated[matrixID]; }

        /** Check if a matrix changed since last frame.
        */
        bool isMatrixChanged(size_t matrixID) const { return mMatricesChanged[matrixID]; }

        /** Get the global matrices.
        */
        const std::vector<glm::mat4>& getGlobalMatrices() const { return mGlobalMatrices; }

        /** Render the UI.
        */
        void renderUI(Gui::Widgets& widget);


        /** Get the previous vertex data buffer for dynamic meshes.
            \return Buffer containing the previous vertex data, or nullptr if no dynamic meshes exist.
        */
        Buffer::SharedPtr getPrevVertexData() const { return mpPrevVertexData; }

        /** Get the total GPU memory usage in bytes.
        */
        uint64_t getMemoryUsageInBytes() const;

    private:
        friend class SceneBuilder;
        AnimationController(Scene* pScene, const StaticVertexVector& staticVertexData, const DynamicVertexVector& dynamicVertexData, const std::vector<Animation::SharedPtr>& animations);

        void initFlags();
        void bindBuffers();
        void updateMatrices();

        void createSkinningPass(const std::vector<PackedStaticVertexData>& staticVertexData, const std::vector<DynamicVertexData>& dynamicVertexData);
        void executeSkinningPass(RenderContext* pContext);
        void initLocalMatrices();

        // Animation
        std::vector<Animation::SharedPtr> mAnimations;
        std::vector<glm::mat4> mLocalMatrices;
        std::vector<glm::mat4> mGlobalMatrices;
        std::vector<glm::mat4> mInvTransposeGlobalMatrices;
        std::vector<bool> mMatricesAnimated;        ///< Flag per matrix, true if matrix is affected by animations.
        std::vector<bool> mMatricesChanged;         ///< Flag per matrix, true if matrix changed since last frame.

        bool mEnabled = true;
        bool mAnimationChanged = true;
        double mLastAnimationTime = 0;
        bool mLoopAnimations = true;
        double mGlobalAnimationLength = 0;
        Scene* mpScene = nullptr;

        Buffer::SharedPtr mpWorldMatricesBuffer;
        Buffer::SharedPtr mpPrevWorldMatricesBuffer;
        Buffer::SharedPtr mpInvTransposeWorldMatricesBuffer;

        // Skinning
        ComputePass::SharedPtr mpSkinningPass;
        std::vector<glm::mat4> mSkinningMatrices;
        std::vector<glm::mat4> mInvTransposeSkinningMatrices;
        uint32_t mSkinningDispatchSize = 0;

        Buffer::SharedPtr mpSkinningMatricesBuffer;
        Buffer::SharedPtr mpInvTransposeSkinningMatricesBuffer;
        Buffer::SharedPtr mpSkinningStaticVertexData;
        Buffer::SharedPtr mpSkinningDynamicVertexData;
        Buffer::SharedPtr mpPrevVertexData;
    };
}
