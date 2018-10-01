/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
***************************************************************************/
#pragma once
#include <vector>
#include <map>
#include "glm/mat4x4.hpp"
#include "glm/vec3.hpp"
#include "Graphics/Model/Mesh.h"
#include "Graphics/Model/ObjectInstance.h"
#include "API/Sampler.h"
#include "Graphics/Model/AnimationController.h"
#include "Graphics/Model/SkinningCache.h"

namespace Falcor
{
    class AssimpModelImporter;
    class BinaryModelImporter;
    class SimpleModelImporter;
    class BinaryModelExporter;
    class Buffer;
    class Camera;

    /** Class representing a complete model object, including meshes, animations and materials
    */

    class Model : public std::enable_shared_from_this<Model>
    {
    public:
        using SharedPtr = std::shared_ptr<Model>;
        using SharedConstPtr = std::shared_ptr<const Model>;

        using MeshInstance = ObjectInstance<Mesh>;
        using MeshInstanceList = std::vector<MeshInstance::SharedPtr>;

        enum class LoadFlags
        {
            None,
            DontGenerateTangentSpace    = 0x1,    ///< Do not attempt to generate tangents if they are missing
            FindDegeneratePrimitives    = 0x2,    ///< Replace degenerate triangles/lines with lines/points. This can create a meshes with topology that wasn't present in the original model.
            AssumeLinearSpaceTextures   = 0x4,    ///< By default, textures representing colors (diffuse/specular) are interpreted as sRGB data. Use this flag to force linear space for color textures.
            DontMergeMeshes             = 0x8,    ///< Preserve the original list of meshes in the scene, don't merge meshes with the same material
            BuffersAsShaderResource     = 0x10,   ///< Generate the VBs and IB with the shader-resource-view bind flag
            RemoveInstancing            = 0x20,   ///< Flatten mesh instances
            UseSpecGlossMaterials       = 0x40,   ///< Set materials to use Spec-Gloss shading model. Otherwise default is Metal-Rough.
        };

        /** Create a new model from file
        */
        static SharedPtr createFromFile(const char* filename, LoadFlags flags = LoadFlags::None);

        static SharedPtr create();

        static const char* kSupportedFileFormatsStr;

        virtual ~Model();

        /** Export the model to a binary file
        */
        void exportToBinaryFile(const std::string& filename);

        /** Get the model radius, calculated based on bounding box size.
        */
        float getRadius() const { return mRadius; }

        /** Get the model center.
        */
        const glm::vec3& getCenter() const { return mBoundingBox.center; }

        /** Get the model's AABB.
        */
        const BoundingBox& getBoundingBox() const { return mBoundingBox; }

        /** Get the number of vertices in the model.
        */
        uint32_t getVertexCount() const { return mVertexCount; }

        /** Get the number of indices in the model.
        */
        uint32_t getIndexCount() const { return mIndexCount; }

        /** Get the number of primitives in the model.
        */
        uint32_t getPrimitiveCount() const { return mPrimitiveCount; }

        /** Get the number of meshes in the model.
        */
        uint32_t getMeshCount() const { return uint32_t(mMeshes.size()); }

        /** Get the total number of mesh instances in the model.
        */
        uint32_t getInstanceCount() const { return mMeshInstanceCount; }

        /** Get the number of unique textures in the model.
        */
        uint32_t getTextureCount() const { return mTextureCount; }

        /** Get the number of unique materials in the model.
        */
        uint32_t getMaterialCount() const { return mMaterialCount; }

        /** Get the number of unique buffers in the model.
        */
        uint32_t getBufferCount() const { return mBufferCount; }

        /** Gets a mesh instance.
            \param[in] meshID ID of the mesh
            \param[in] instanceID ID of the instance
            \return Mesh instance
        */
        const MeshInstance::SharedPtr& getMeshInstance(uint32_t meshID, uint32_t instanceID) const { return mMeshes[meshID][instanceID]; }

        /** Gets a mesh.
            \param[in] meshID ID of the mesh
            \return Mesh object
        */
        const Mesh::SharedPtr& getMesh(uint32_t meshID) const { return mMeshes[meshID][0]->getObject(); };

        /** Gets how many instances exist of a mesh.
            \param[in] meshID ID of the mesh
            \return Number of instances
        */
        uint32_t getMeshInstanceCount(uint32_t meshID) const { return meshID >= mMeshes.size() ? 0 : (uint32_t)(mMeshes[meshID].size()); }

        /** Adds a new mesh instance.
            \param[in] pMesh Mesh geometry
            \param[in] baseTransform Base transform for the instance
        */
        void addMeshInstance(const Mesh::SharedPtr& pMesh, const glm::mat4& baseTransform);

        /** Check if the model contains animations.
        */
        bool hasAnimations() const;

        /** Get the number of animations in the model.
        */
        uint32_t getAnimationsCount() const;

        /** Animate the active animation. Use setActiveAnimation() to switch between different animations.
            \param[in] currentTime The current global time
            \return true if model has changed
        */
        bool animate(double currentTime);

        /** Get the animation name from animation ID.
        */
        const std::string& getAnimationName(uint32_t animationID) const;

        /** Turn animations off and use bind pose for rendering.
        */
        void setBindPose();
        
        /** Turn animation on and select active animation. Changing the active animation will cause the new animation to play from the beginning.
        */
        void setActiveAnimation(uint32_t animationID);

        /** Get the active animation.
        */
        uint32_t getActiveAnimation() const;

        /** Set the animation controller for the model.
        */
        void setAnimationController(AnimationController::UniquePtr pAnimController);

        /** Attach a skinning cache to the model, or nullptr to detach.
            When a cache is attached, the model will use compute shader based skinning with caching of the resulting skinned vertex buffers.
        */
        void attachSkinningCache(SkinningCache::SharedPtr pSkinningCache);

        /** Get the skinning cache for the model, or nullptr if none.
        */
        SkinningCache::SharedPtr getSkinningCache() const;

        /** Returns a vertex array object with skinned vertex buffers for skinned models, or the original vertex buffers otherwise.
            This function requires a skinning cache to be attached to skinned models.
        */
        Vao::SharedPtr getMeshVao(const Mesh* pMesh) const;

        /** Check if the model has bones.
        */
        bool hasBones() const;

        /** Get the number of bone matrices.
        */
        uint32_t getBoneCount() const;

        /** Get array of bone matrices.
            \return If model has bones, return pointer to matrices in the current state of the animation. Otherwise nullptr.
        */
        const mat4* getBoneMatrices() const;

        /** Get array of bones' inverse transpose matrices.
            \return If model has bones, return pointer to matrices in the current state of the animation. Otherwise nullptr.
        */
        const mat4* getBoneInvTransposeMatrices() const;

        /** Force all texture maps in all materials to use a specific texture sampler with one of their maps
            \param[in] Type The map Type to bind the sampler with
        */
        void bindSamplerToMaterials(const Sampler::SharedPtr& pSampler);

        /** Delete meshes from the model culled by the camera's frustum.
            The function will also delete buffers, textures and materials not in use anymore.
        */
        void deleteCulledMeshes(const Camera* pCamera);

        /** Name the model
        */
        void setName(const std::string& Name) { mName = Name; }

        /** Get the model's name
        */
        const std::string& getName() const { return mName; }

        /** Set the model's filename
        */
        void setFilename(const std::string& filename) { mFilename = filename; }

        /** Get the model's filename
        */
        const std::string& getFilename() const { return mFilename; }

        /** Get global ID of the model
        */
        const uint32_t getId() const { return mId; }
        
        /** Reset all global id counter of model, mesh and material
        */
        static void resetGlobalIdCounter();


    protected:
        friend class SimpleModelImporter;

        Model();
        Model(const Model& other);
        void sortMeshes();
        void deleteCulledMeshInstances(MeshInstanceList& meshInstances, const Camera *pCamera);
        virtual bool update();

        BoundingBox mBoundingBox;
        float mRadius;

        uint32_t mVertexCount;
        uint32_t mIndexCount;
        uint32_t mPrimitiveCount;
        uint32_t mMeshInstanceCount;
        uint32_t mBufferCount;
        uint32_t mMaterialCount;
        uint32_t mTextureCount;

        uint32_t mId;

        std::vector<MeshInstanceList> mMeshes; // [Mesh][Instance]

        AnimationController::UniquePtr mpAnimationController;
        SkinningCache::SharedPtr mpSkinningCache;

        std::string mName;
        std::string mFilename;

        static uint32_t sModelCounter;

        void calculateModelProperties();
    };

    enum_class_operators(Model::LoadFlags);

#define flag_str(a) case Model::LoadFlags::a: return #a
    inline std::string to_string(Model::LoadFlags f)
    {
        switch (f)
        {
            flag_str(None);
            flag_str(DontGenerateTangentSpace);
            flag_str(FindDegeneratePrimitives);
            flag_str(AssumeLinearSpaceTextures);
            flag_str(DontMergeMeshes);
            flag_str(BuffersAsShaderResource);
            flag_str(RemoveInstancing);            
            flag_str(UseSpecGlossMaterials);
        default:
            should_not_get_here();
            return "";
        }
    }
#undef flag_str
}
