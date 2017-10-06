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
#include <string>
//#include "Utils/BinaryFileStream.h"
#include "BinaryModelSpec.h"
#include "../Model.h"
#include "glm/vec3.hpp"
#include "Data/VertexAttrib.h"
#include "Graphics/Model/Loaders/ModelImporter.h"

namespace Falcor
{
    class Texture;

    /** 
        DEPRECATED
    */

    // Prior Falcor classes load models from disk via some format.  Chris wanted some way to create
    //     models on the fly from memory resources.  This class is a start at doing this.
    //     Note 1:  this is pretty simplistic and may not correctly integrate with fancy materials, etc.
    //     Note 2:  this may not correctly setup model to interact with DirectX shaders.  In particular,
    //              vertex attribute names are required, and these may not all be set correctly!
    class SimpleModelImporter : public ModelImporter
    {
    public:
        
        // Where can we attach vertex attributes in our shader?
        enum class AttribType
        {
            Position = VERTEX_POSITION_LOC,
            Normal = VERTEX_NORMAL_LOC,
            Bitangent = VERTEX_BITANGENT_LOC,
            BoneWeight = VERTEX_BONE_WEIGHT_LOC,
            BoneID = VERTEX_BONE_ID_LOC,
            Color = VERTEX_DIFFUSE_COLOR_LOC,
            TexCoord = VERTEX_TEXCOORD_LOC,
            User0 = VERTEX_USER0_LOC,
            User1 = VERTEX_USER0_LOC + 1,
            User2 = VERTEX_USER0_LOC + 2,
            User3 = VERTEX_USER0_LOC + 3,
        };

        // What format are the attributes do we have for this model?
        struct VertexAttrib
        {
            SimpleModelImporter::AttribType  attribType;
            uint32_t                         numElements;
            AttribFormat                     attribFormat;
        };

        // What format are our vertices stored in memory?
        struct VertexFormat
        {
            std::vector<VertexAttrib> attribs;
        };

        // Create a model made up of a number of triangles, layed out (in the index buffer) as GL_TRIANGLES
        static Model::SharedPtr create( VertexFormat vertLayout, uint32_t vboSz, const void *vboData, 
                                        uint32_t idxBufSz, const uint32_t *idxData, 
                                        Texture::SharedPtr diffuseTexture = nullptr,
                                        Vao::Topology geomTopology = Vao::Topology::TriangleList );

    private:
        static ResourceFormat    getResourceFormat( AttribFormat format, uint32_t components );
        static int32_t           getFormatByteSize( AttribFormat format );
        static const std::string getSemanticName( AttribType type );
    };
}
