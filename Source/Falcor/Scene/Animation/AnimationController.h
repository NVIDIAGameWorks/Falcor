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
        static const uint32_t kBindPoseAnimationId = -1;
        static const uint32_t kInvalidBoneID = -1;
        ~AnimationController() = default;

        using StaticVertexVector = std::vector<PackedStaticVertexData>;
        using DynamicVertexVector = std::vector<DynamicVertexData>;

        /** Create a new object
        */
        static UniquePtr create(Scene* pScene, const StaticVertexVector& staticVertexData, const DynamicVertexVector& dynamicVertexData);
        
        /** Add an animation for a mesh
        */
        void addAnimation(uint32_t meshID, Animation::ConstSharedPtrRef pAnimation);

        /** Toggle all mesh animations on or off.
        */
        void toggleAnimations(bool animate);

        /** Run the animation
            \return true if a change occurred, otherwise false
        */
        bool animate(RenderContext* pContext, double currentTime);

        /** Get number of animations for a mesh
        */
        uint32_t getMeshAnimationCount(uint32_t meshID) const;

        /** Get the name of a mesh's animation
            If the animation doesn't exist it will return an empty string.
            Don't use this function to detect if an animation exists or not. Use `getMeshAnimationCount()` instead
        */
        const std::string& getAnimationName(uint32_t meshID, uint32_t animID) const;

        /** Set a mesh active animation. Use kBindPoseAnimationId to disable animations for the mesh
            \If the mesh/animation exists, will return true. Otherwise returns false
        */
        bool setActiveAnimation(uint32_t meshID, uint32_t animID);

        /** Get a mesh's active animation.
            \return Active animation ID, or kBindPoseAnimationId if no animations exist for the mesh.
        */
        uint32_t getActiveAnimation(uint32_t meshID) const;

        /** Whether the controller is handling any animations.
        */
        bool hasAnimations() const { return mHasAnimations; }

        /** Render the UI
        */
        void renderUI(Gui::Widgets& widget);

        /** Get the global matrices
        */
        const std::vector<glm::mat4>& getGlobalMatrices() const { return mGlobalMatrices; }

        /** Check if a matrix changed
        */
        bool didMatrixChanged(size_t matrixID) const { return mMatricesChanged[matrixID]; }

    private:
        friend class SceneBuilder;
        AnimationController(Scene* pScene, const StaticVertexVector& staticVertexData, const DynamicVertexVector& dynamicVertexData);

        void allocatePrevWorldMatrixBuffer();
        void bindBuffers();
        void updateMatrices();
        bool validateIndices(uint32_t meshID, uint32_t animID, const std::string& warningPrefix) const;

        struct MeshAnimation
        {
            std::vector<Animation::SharedPtr> pAnimations;
            uint32_t activeAnimation = kBindPoseAnimationId;
        };

        std::map<uint32_t, MeshAnimation> mMeshes;
        std::vector<glm::mat4> mLocalMatrices;
        std::vector<glm::mat4> mGlobalMatrices;
        std::vector<glm::mat4> mInvTransposeGlobalMatrices;
        std::vector<bool> mMatricesChanged;

        bool mHasAnimations = false;
        bool mAnimationChanged = true;
        uint32_t mActiveAnimationCount = 0;
        double mLastAnimationTime = 0;
        Scene* mpScene = nullptr;

        Buffer::SharedPtr mpWorldMatricesBuffer;
        Buffer::SharedPtr mpPrevWorldMatricesBuffer;
        Buffer::SharedPtr mpInvTransposeWorldMatricesBuffer;

        // Skinning
        ComputePass::SharedPtr mpSkinningPass;
        std::vector<glm::mat4> mSkinningMatrices;
        std::vector<glm::mat4> mInvTransposeSkinningMatrices;
        uint32_t mSkinningDispatchSize = 0;
        void createSkinningPass(const std::vector<PackedStaticVertexData>& staticVertexData, const std::vector<DynamicVertexData>& dynamicVertexData);
        void executeSkinningPass(RenderContext* pContext);

        Buffer::SharedPtr mpSkinningMatricesBuffer;
        Buffer::SharedPtr mpInvTransposeSkinningMatricesBuffer;
        void initLocalMatrices();
    };
}
