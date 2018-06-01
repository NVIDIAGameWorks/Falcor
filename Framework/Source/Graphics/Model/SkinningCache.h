/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
#include <map>
#include "API/RenderContext.h"

namespace Falcor
{
    class Model;
    class Mesh;

    /** Cache for skinned vertex buffers for one or more models.

        Compute shader based skinning that caches the generated vertex buffers.
        This allows skinning to be done asynchronously before rendering.
        It also allows updating skinning at a lower frequency than the frame rate,
        and it simplifies the scene renderer as it does not have to deal with skinning.

        TODOs:

        1)  The class handles skinning of positions, normals, and bitangents.
            We might want to generalize it and allow the user to override the shader
            to output additional skinned vertex buffers.

        2)  Currently, a single set of skinned vertex buffers is stored per mesh.
            We could extend that to hold multiple buffers to cache entire animations.

        3)  We could also extend it to hold skinned buffers per mesh instance, to enable
            mesh instances to be animated separately.

        4)  Provide metric on amount of change to guide choice of BVH rebuild/refit for ray tracing purposes.

    */
    class SkinningCache : public std::enable_shared_from_this<SkinningCache>
    {
    public:
        using SharedPtr = std::shared_ptr<SkinningCache>;
        using SharedConstPtr = std::shared_ptr<const SkinningCache>;
        virtual ~SkinningCache() = default;

        static SharedPtr create();

        /** Create/update skinned vertex buffers for model.
        */
        bool update(const Model* pModel);

        /** Returns the vertex array object for pMesh containing skinned vertex buffers if it exists.
        */
        Vao::SharedPtr getVao(const Mesh* pMesh) const;

    protected:
        SkinningCache() = default;

        bool init();
        void initVariableOffsets(const ParameterBlockReflection* pBlock);
        void initMeshBufferLocations(const ParameterBlockReflection* pBlock);
        void createVertexBuffers(const Mesh* pMesh);
        void setPerModelData(const Model* pModel);
        void setPerMeshData(const Mesh* pMesh);

        struct VertexBuffers
        {
            Vao::SharedPtr pVao;
            bool valid = false;
        };

        struct VariableOffsets
        {
            size_t bonesOffset = ConstantBuffer::kInvalidOffset;
            size_t bonesInvTransposeOffset = ConstantBuffer::kInvalidOffset;
        };

        struct MeshBufferLocations
        {
            // Input
            ParameterBlockReflection::BindLocation position;
            ParameterBlockReflection::BindLocation normal;
            ParameterBlockReflection::BindLocation bitangent;
            ParameterBlockReflection::BindLocation boneWeights;
            ParameterBlockReflection::BindLocation boneIds;
            // Output
            ParameterBlockReflection::BindLocation positionOut;
            ParameterBlockReflection::BindLocation prevPositionOut;
            ParameterBlockReflection::BindLocation normalOut;
            ParameterBlockReflection::BindLocation bitangentOut;
        };

        VariableOffsets mVariableOffsets;
        MeshBufferLocations mMeshBufferLocations;

        std::map<const Mesh*, VertexBuffers> mSkinnedBuffers;

        struct
        {
            ComputeState::SharedPtr pState;
            ComputeProgram::SharedPtr pProgram;
            ComputeVars::SharedPtr pVars;
        } mSkinningPass;
    };
}