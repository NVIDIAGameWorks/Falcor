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
#include "Transform.h"
#include "Core/Macros.h"
#include "Utils/Math/Vector.h"
#include "Utils/Math/Matrix.h"
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace Falcor
{
    /** Simple indexed triangle mesh.
        Vertices have position, normal and texture coordinate attributes.
        This class is used as a utility to pass simple geometry to the SceneBuilder.
    */
    class FALCOR_API TriangleMesh
    {
    public:
        using SharedPtr = std::shared_ptr<TriangleMesh>;

        struct Vertex
        {
            float3 position;
            float3 normal;
            float2 texCoord;
        };

        using VertexList = std::vector<Vertex>;
        using IndexList = std::vector<uint32_t>;

        /** Creates a triangle mesh.
            \return Returns the triangle mesh.
        */
        static SharedPtr create();

        /** Creates a triangle mesh.
            \param[in] vertices Vertex list.
            \param[in] indices Index list.
            \param[in] frontFaceCW Triangle winding.
            \return Returns the triangle mesh.
        */
        static SharedPtr create(const VertexList& vertices, const IndexList& indices, bool frontFaceCW = false);

        /** Creates a dummy mesh (single degenerate triangle).
            \return Returns the triangle mesh.
        */
        static SharedPtr createDummy();

        /** Creates a quad mesh, centered at the origin with normal pointing in positive Y direction.
            \param[in] size Size of the quad in X and Z.
            \return Returns the triangle mesh.
        */
        static SharedPtr createQuad(float2 size = float2(1.f));

        /** Creates a disk mesh, centered at the origin with the normal pointing in positive Y direction.
            \param[in] radius Radius of the disk.
            \return Returns the triangle mesh.
        */
        static SharedPtr createDisk(float radius, uint32_t segments = 32);

        /** Creates a cube mesh, centered at the origin.
            \param[in] size Size of the cube in each dimension.
            \return Returns the triangle mesh.
        */
        static SharedPtr createCube(float3 size = float3(1.f));

        /** Creates a UV sphere mesh, centered at the origin with poles in positive/negative Y direction.
            \param[in] radius Radius of the sphere.
            \param[in] segmentsU Number of segments along parallels.
            \param[in] segmentsV Number of segments along meridians.
            \return Returns the triangle mesh.
        */
        static SharedPtr createSphere(float radius = 0.5f, uint32_t segmentsU = 32, uint32_t segmentsV = 16);

        /** Creates a triangle mesh from a file.
            This is using ASSIMP to support a wide variety of asset formats.
            All geometry found in the asset is pre-transformed and merged into the same triangle mesh.
            \param[in] path File path to load mesh from.
            \param[in] smoothNormals If no normals are defined in the model, generate smooth instead of facet normals.
            \return Returns the triangle mesh or nullptr if the mesh failed to load.
        */
        static SharedPtr createFromFile(const std::filesystem::path& path, bool smoothNormals = false);

        /** Get the name of the triangle mesh.
            \return Returns the name.
        */
        const std::string getName() const { return mName; }

        /** Set the name of the triangle mesh.
            \param[in] name Name to set.
        */
        void setName(const std::string& name) { mName = name; }

        /** Adds a vertex to the vertex list.
            \param[in] position Vertex position.
            \param[in] normal Vertex normal.
            \param[in] texCoord Vertex texture coordinate.
            \return Returns the vertex index.
        */
        uint32_t addVertex(float3 position, float3 normal, float2 texCoord);

        /** Adds a triangle to the index list.
            \param[in] i0 First index.
            \param[in] i1 Second index.
            \param[in] i2 Third index.
        */
        void addTriangle(uint32_t i0, uint32_t i1, uint32_t i2);

        /** Get the vertex list.
        */
        const VertexList& getVertices() const { return mVertices; }

        /** Set the vertex list.
        */
        void setVertices(const VertexList& vertices) { mVertices = vertices; }

        /** Get the index list.
        */
        const IndexList& getIndices() const { return mIndices; }

        /** Set the index list.
        */
        void setIndices(const IndexList& indices) { mIndices = indices; }

        /** Get the triangle winding.
        */
        bool getFrontFaceCW() const { return mFrontFaceCW; }

        /** Set the triangle winding.
        */
        void setFrontFaceCW(bool frontFaceCW) { mFrontFaceCW = frontFaceCW; }

        /** Applies a transform to the triangle mesh.
            \param[in] transform Transform to apply.
        */
        void applyTransform(const Transform& transform);

        /** Applies a transform to the triangle mesh.
            \param[in] transform Transform to apply.
        */
        void applyTransform(const rmcv::mat4& transform);

    private:
        TriangleMesh();
        TriangleMesh(const VertexList& vertices, const IndexList& indices, bool frontFaceCW);

        std::string mName;
        std::vector<Vertex> mVertices;
        std::vector<uint32_t> mIndices;
        bool mFrontFaceCW = false;
    };
}
