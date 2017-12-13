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
#include "Framework.h"
#include "SimpleModelImporter.h"
#include "../Model.h"
#include "../Mesh.h"
#include "Utils/Platform/OS.h"
#include "API/VertexLayout.h"
#include "Data/VertexAttrib.h"
#include "API/Buffer.h"
#include "glm/common.hpp"
#include "API/Formats.h"
#include "API/Texture.h"
#include "Graphics/Material/BasicMaterial.h"
#include "glm/geometric.hpp"

namespace Falcor
{

    Model::SharedPtr SimpleModelImporter::create( VertexFormat vertLayout, uint32_t vboSz, const void *vboData,
                                                  uint32_t idxBufSz, const uint32_t *idxBufData, Texture::SharedPtr diffuseTexture,
                                                  Vao::Topology geomTopology )
    {
        // Since SimpleModelImporter is all static, create an instance here to help track materials
        SimpleModelImporter modelImporter;

        // Create our model container
        Model::SharedPtr pModel = Model::create();

        // Need to create our vertex layout
        VertexBufferLayout::SharedPtr pVertexLayout = VertexBufferLayout::create();
        uint32_t vertexStride = 0;
        uint32_t positionOffset = 0;
        for ( int i = 0; i < vertLayout.attribs.size(); i++ )
        {
            // Convert the vertex attrib structure into what we need internally in this loop
            uint32_t     type = uint32_t( vertLayout.attribs[i].attribType );
            AttribFormat format = vertLayout.attribs[i].attribFormat;
            int32_t      length = vertLayout.attribs[i].numElements;
            if ( length <= 0 || length > 4 )
                continue;

            // If this is a "position" attribute, remember the offset, since we'll use this later to
            //    compute a bounding box for the entire mesh
            if ( vertLayout.attribs[i].attribType == AttribType::Position )
                positionOffset = vertexStride;

            // Do some conversions to the format we need data in to set a Falcor vertex attribute entry
            ResourceFormat falcorFormat = getResourceFormat( format, length );
            const std::string falcorName = getSemanticName( vertLayout.attribs[i].attribType );

            // Add this vertex attribute to our format
            pVertexLayout->addElement( falcorName, vertexStride, falcorFormat, 1, type );

            // Add this attribute's size to our per-vertex size.
            uint32_t size = getFormatByteSize( AttribFormat( format ) ) * length;
            vertexStride += size;
        }

        // Create vertex buffer and add to the model
        VertexLayout::SharedPtr pLayout = VertexLayout::create();
        pLayout->addBufferLayout(0, pVertexLayout);
        Buffer::SharedPtr pBuffer = Buffer::create( vboSz, Buffer::BindFlags::Vertex, Buffer::CpuAccess::None, vboData );

        // Create index buffer and add to the model
        Buffer::SharedPtr pIB = Buffer::create( idxBufSz, Buffer::BindFlags::Index, Buffer::CpuAccess::None, idxBufData );

        // Compute more explicit / traditional counts needed internally
        uint32_t numVertices = vboSz / vertexStride;
        uint32_t numIndicies = idxBufSz / (sizeof( uint32_t ));

        // Create a really simple, dumb material for this mesh
        BasicMaterial basicMat;
        if ( diffuseTexture )
        {
            basicMat.pTextures[BasicMaterial::MapType::DiffuseMap] = diffuseTexture;
        }
        basicMat.diffuseColor = vec3( 1.0f );
        Material::SharedPtr pSimpleMaterial = basicMat.convertToMaterial();
        pSimpleMaterial = modelImporter.checkForExistingMaterial( pSimpleMaterial );

        // Calculate a bounding-box for this model
        glm::vec3 posMax, posMin;
        for ( uint32_t i = 0; i < numIndicies; i++ )
        {
            // Find a pointer to the floats containing our vertex position
            uint32_t vertexID = idxBufData[i];
            uint8_t* pVertex = ((uint8_t *) vboData) + ( vertexStride * vertexID );
            float* pPosition = (float*) (pVertex + positionOffset);

            glm::vec3 xyz( pPosition[0], pPosition[1], pPosition[2] );
            posMin = glm::min( posMin, xyz );
            posMax = glm::max( posMax, xyz );
        }
        BoundingBox box = BoundingBox::fromMinMax( posMin, posMax );

        // create a mesh containing this index & vertex data.
        Mesh::SharedPtr pMesh = Mesh::create({ pBuffer }, numVertices, pIB, numIndicies, pLayout, geomTopology, pSimpleMaterial, box, false);
        pModel->addMeshInstance(pMesh, glm::mat4()); // Add this mesh to the model

        // Do internal computations on model properties
        pModel->calculateModelProperties();

        // Done
        return pModel;
    }

    ResourceFormat SimpleModelImporter::getResourceFormat( AttribFormat format, uint32_t components )
    {    
        ResourceFormat byteFormats[4] = { ResourceFormat::R8Unorm, ResourceFormat::RG8Unorm, ResourceFormat::RGBA8Unorm, ResourceFormat::RGBA8Unorm };
        ResourceFormat intFormats[4] = { ResourceFormat::R32Int, ResourceFormat::RG32Int, ResourceFormat::RGB32Int, ResourceFormat::RGBA32Int };
        ResourceFormat floatFormats[4] = { ResourceFormat::R32Float, ResourceFormat::RG32Float, ResourceFormat::RGB32Float, ResourceFormat::RGBA32Float };

        if ( format == AttribFormat::AttribFormat_U8 )
            return byteFormats[clamp( int( components ) - 1, 0, 3 )];
        else if ( format == AttribFormat::AttribFormat_S32 )
            return intFormats[clamp( int( components ) - 1, 0, 3 )];
        else if ( format == AttribFormat::AttribFormat_F32 )
            return floatFormats[clamp( int( components ) - 1, 0, 3 )];

        should_not_get_here();
        return ResourceFormat::Unknown;
    }

    int32_t SimpleModelImporter::getFormatByteSize( AttribFormat format )
    {
        switch ( format )
        {
        case AttribFormat_U8:
            return 1;
        case AttribFormat_S32:
        case AttribFormat_F32:
            return 4;
        default:
            should_not_get_here();
            return 0;
        }
    }

    const std::string SimpleModelImporter::getSemanticName( AttribType type )
    {
        switch ( type )
        {
        case AttribType::Position:
            return VERTEX_POSITION_NAME;
        case AttribType::Normal:
            return VERTEX_NORMAL_NAME;
        case AttribType::Color:
            return VERTEX_DIFFUSE_COLOR_NAME;
        case AttribType::TexCoord:
            return VERTEX_TEXCOORD_NAME;
        case AttribType::Bitangent:
            return VERTEX_BITANGENT_NAME;
        case AttribType::BoneWeight:
            return VERTEX_BONE_WEIGHT_NAME;
        case AttribType::BoneID:
            return VERTEX_BONE_ID_NAME;
        case AttribType::User0:
            return "USER0";
        case AttribType::User1:
            return "USER1";
        case AttribType::User2:
            return "USER2";
        case AttribType::User3:
            return "USER3";
        default:
            should_not_get_here();
            return "";
        }
    }

} // end namespace Falcor