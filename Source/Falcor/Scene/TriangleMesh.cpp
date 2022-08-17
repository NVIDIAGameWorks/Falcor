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
#include "TriangleMesh.h"
#include "Core/Assert.h"
#include "Core/Platform/OS.h"
#include "Utils/Logger.h"
#include "Utils/Scripting/ScriptBindings.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <cmath>

namespace Falcor
{
    TriangleMesh::SharedPtr TriangleMesh::create()
    {
        return SharedPtr(new TriangleMesh());
    }

    TriangleMesh::SharedPtr TriangleMesh::create(const VertexList& vertices, const IndexList& indices, bool frontFaceCW)
    {
        return SharedPtr(new TriangleMesh(vertices, indices, frontFaceCW));
    }

    TriangleMesh::SharedPtr TriangleMesh::createDummy()
    {
        VertexList vertices = {{{0.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f}}};
        IndexList indices = {0, 0, 0};
        return create(vertices, indices);
    }

    TriangleMesh::SharedPtr TriangleMesh::createQuad(float2 size)
    {
        float2 hsize = 0.5f * size;
        float3 normal{0.f, 1.f, 0.f};
        bool frontFaceCW = size.x * size.y < 0.f;

        VertexList vertices{
            {{ -hsize.x, 0.f, -hsize.y }, normal, { 0.f, 0.f }},
            {{  hsize.x, 0.f, -hsize.y }, normal, { 1.f, 0.f }},
            {{ -hsize.x, 0.f,  hsize.y }, normal, { 0.f, 1.f }},
            {{  hsize.x, 0.f,  hsize.y }, normal, { 1.f, 1.f }},
        };

        IndexList indices{
            2, 1, 0,
            1, 2, 3,
        };

        return create(vertices, indices, frontFaceCW);
    }

    TriangleMesh::SharedPtr TriangleMesh::createDisk(float radius, uint32_t segments)
    {
        std::vector<Vertex> vertices(segments + 1);
        std::vector<uint32_t> indices(segments * 3);

        float3 normal = { 0.f, 1.f, 0.f };
        vertices[0] = { { 0.f, 0.f, 0.f }, normal, { 0.5f, 0.5f } };

        for (uint32_t i = 0; i < segments; ++i)
        {
            float phi = float(i) / float(segments) * 2.f * (float)M_PI;
            float c = std::cos(phi);
            float s = -std::sin(phi);
            vertices[i + 1] = { { c * radius, 0.f, s * radius }, normal, { 0.5f + c * 0.5f, 0.5f + s * 0.5f } };

            indices[i * 3] = 0;
            indices[i * 3 + 1] = i + 1;
            indices[i * 3 + 2] = ((i + 1) % segments) + 1;
        }

        return create(vertices, indices, false);
    }

    TriangleMesh::SharedPtr TriangleMesh::createCube(float3 size)
    {
        const float3 positions[6][4] =
        {
            {{ -0.5f, -0.5f, -0.5f }, { -0.5f, -0.5f,  0.5f }, { 0.5f, -0.5f,  0.5f }, { 0.5f, -0.5f, -0.5f }},
            {{ -0.5f,  0.5f,  0.5f }, { -0.5f,  0.5f, -0.5f }, { 0.5f,  0.5f, -0.5f }, { 0.5f,  0.5f,  0.5f }},
            {{ -0.5f,  0.5f, -0.5f }, { -0.5f, -0.5f, -0.5f }, { 0.5f, -0.5f, -0.5f }, { 0.5f,  0.5f, -0.5f }},
            {{  0.5f,  0.5f,  0.5f }, {  0.5f, -0.5f,  0.5f }, {-0.5f, -0.5f,  0.5f }, {-0.5f,  0.5f,  0.5f }},
            {{ -0.5f,  0.5f,  0.5f }, { -0.5f, -0.5f,  0.5f }, {-0.5f, -0.5f, -0.5f }, {-0.5f,  0.5f, -0.5f }},
            {{  0.5f,  0.5f, -0.5f }, {  0.5f, -0.5f, -0.5f }, { 0.5f, -0.5f,  0.5f }, { 0.5f,  0.5f,  0.5f }},
        };

        const float3 normals[6] =
        {
            { 0.f, -1.f, 0.f },
            { 0.f, 1.f, 0.f },
            { 0.f, 0.f, -1.f },
            { 0.f, 0.f, 1.f },
            { -1.f, 0.f, 0.f },
            { 1.f, 0.f, 0.f },
        };

        const float2 texCoords[4] = {{ 0.f, 0.f }, { 1.f, 0.f }, { 1.f, 1.f }, { 0.f, 1.f }};

        VertexList vertices;
        IndexList indices;

        float3 sign = { size.x < 0.f ? -1.f : 1.f, size.y < 0.f ? -1.f : 1.f, size.z < 0.f ? -1.f : 1.f };
        bool frontFaceCW = size.x * size.y * size.z < 0.f;

        for (size_t i = 0; i < 6; ++i)
        {
            uint32_t idx = (uint32_t)vertices.size();
            indices.emplace_back(idx);
            indices.emplace_back(idx + 2);
            indices.emplace_back(idx + 1);
            indices.emplace_back(idx);
            indices.emplace_back(idx + 3);
            indices.emplace_back(idx + 2);

            for (size_t j = 0; j < 4; ++j)
            {
                vertices.emplace_back(Vertex{ positions[i][j] * size, normals[i] * sign, texCoords[j] });
            }
        }

        return create(vertices, indices, frontFaceCW);
    }

    TriangleMesh::SharedPtr TriangleMesh::createSphere(float radius, uint32_t segmentsU, uint32_t segmentsV)
    {
        VertexList vertices;
        IndexList indices;

        // Create vertices.
        for (uint32_t v = 0; v <= segmentsV; ++v)
        {
            for (uint32_t u = 0; u <= segmentsU; ++u)
            {
                float2 uv = float2(u / float(segmentsU), v / float(segmentsV));
                float theta = uv.x * 2.f * (float)M_PI;
                float phi = uv.y * (float)M_PI;
                float3 dir = float3(
                    std::cos(theta) * std::sin(phi),
                    std::cos(phi),
                    std::sin(theta) * std::sin(phi)
                );
                vertices.emplace_back(Vertex{ dir * radius, dir, uv });
            }
        }

        // Create indices.
        for (uint32_t v = 0; v < segmentsV; ++v)
        {
            for (uint32_t u = 0; u < segmentsU; ++u)
            {
                uint32_t i0 = v * (segmentsU + 1) + u;
                uint32_t i1 = v * (segmentsU + 1) + (u + 1) % (segmentsU + 1);
                uint32_t i2 = (v + 1) * (segmentsU + 1) + u;
                uint32_t i3 = (v + 1) * (segmentsU + 1) + (u + 1) % (segmentsU + 1);

                indices.emplace_back(i0);
                indices.emplace_back(i1);
                indices.emplace_back(i2);

                indices.emplace_back(i2);
                indices.emplace_back(i1);
                indices.emplace_back(i3);
            }
        }

        return create(vertices, indices);
    }

    TriangleMesh::SharedPtr TriangleMesh::createFromFile(const std::filesystem::path& path, bool smoothNormals)
    {
        std::filesystem::path fullPath;
        if (!findFileInDataDirectories(path, fullPath))
        {
            logWarning("Error when loading triangle mesh. Can't find mesh file '{}'.", path);
            return nullptr;
        }

        Assimp::Importer importer;

        unsigned int flags =
            aiProcess_FlipUVs |
            aiProcess_Triangulate |
            (smoothNormals ? aiProcess_GenSmoothNormals : aiProcess_GenNormals) |
            aiProcess_PreTransformVertices;

        const aiScene* scene = nullptr;

        if (hasExtension(fullPath, "gz"))
        {
            auto decompressed = decompressFile(fullPath);
            scene = importer.ReadFileFromMemory(decompressed.data(), decompressed.size(), flags);
        }
        else
        {
            scene = importer.ReadFile(fullPath.string().c_str(), flags);
        }

        if (!scene)
        {
            logWarning("Failed to load triangle mesh from '{}': {}", fullPath, importer.GetErrorString());
            return nullptr;
        }

        VertexList vertices;
        IndexList indices;

        size_t vertexCount = 0;
        size_t indexCount = 0;

        for (size_t meshIdx = 0; meshIdx < scene->mNumMeshes; ++meshIdx)
        {
            vertexCount += scene->mMeshes[meshIdx]->mNumVertices;
            indexCount += scene->mMeshes[meshIdx]->mNumFaces * 3;
        }

        vertices.reserve(vertexCount);
        indices.reserve(indexCount);

        for (size_t meshIdx = 0; meshIdx < scene->mNumMeshes; ++meshIdx)
        {
            size_t indexBase = vertices.size();
            auto mesh = scene->mMeshes[meshIdx];
            for (size_t vertexIdx = 0; vertexIdx < mesh->mNumVertices; ++vertexIdx)
            {
                const auto& vertex = mesh->mVertices[vertexIdx];
                const auto& normal = mesh->mNormals[vertexIdx];
                const auto& texCoord = mesh->mTextureCoords[0] ? mesh->mTextureCoords[0][vertexIdx] : aiVector3D(0.f);
                vertices.emplace_back(Vertex{
                    float3(vertex.x, vertex.y, vertex.z),
                    float3(normal.x, normal.y, normal.z),
                    float2(texCoord.x, texCoord.y)
                });
            }
            for (size_t faceIdx = 0; faceIdx < mesh->mNumFaces; ++faceIdx)
            {
                const auto& face = mesh->mFaces[faceIdx];
                for (size_t i = 0; i < 3; ++i) indices.emplace_back((uint32_t)(indexBase + face.mIndices[i]));
            }
        }

        return create(vertices, indices);
    }

    uint32_t TriangleMesh::addVertex(float3 position, float3 normal, float2 texCoord)
    {
        mVertices.emplace_back(Vertex{position, normal, texCoord});
        FALCOR_ASSERT(mVertices.size() < std::numeric_limits<uint32_t>::max());
        return (uint32_t)(mVertices.size() - 1);
    }

    void TriangleMesh::addTriangle(uint32_t i0, uint32_t i1, uint32_t i2)
    {
        mIndices.emplace_back(i0);
        mIndices.emplace_back(i1);
        mIndices.emplace_back(i2);
    }

    void TriangleMesh::applyTransform(const Transform& transform)
    {
        applyTransform(transform.getMatrix());
    }

    void TriangleMesh::applyTransform(const rmcv::mat4& transform)
    {
        auto invTranspose = (rmcv::mat3)rmcv::transpose(rmcv::inverse(transform));

        for (auto& vertex : mVertices)
        {
            vertex.position = (transform * float4(vertex.position, 1.f)).xyz;
            vertex.normal = glm::normalize(invTranspose * vertex.normal);
        }

        // Check if triangle winding has flipped and adjust winding order accordingly.
        bool flippedWinding = rmcv::determinant((rmcv::mat3)transform) < 0.f;
        if (flippedWinding) mFrontFaceCW = !mFrontFaceCW;
    }

    TriangleMesh::TriangleMesh()
    {}

    TriangleMesh::TriangleMesh(const VertexList& vertices, const IndexList& indices, bool frontFaceCW)
        : mVertices(vertices)
        , mIndices(indices)
        , mFrontFaceCW(frontFaceCW)
    {}

    FALCOR_SCRIPT_BINDING(TriangleMesh)
    {
        using namespace pybind11::literals;

        pybind11::class_<TriangleMesh, TriangleMesh::SharedPtr> triangleMesh(m, "TriangleMesh");
        triangleMesh.def_property("name", &TriangleMesh::getName, &TriangleMesh::setName);
        triangleMesh.def_property("frontFaceCW", &TriangleMesh::getFrontFaceCW, &TriangleMesh::setFrontFaceCW);
        triangleMesh.def_property_readonly("vertices", &TriangleMesh::getVertices);
        triangleMesh.def_property_readonly("indices", &TriangleMesh::getIndices);
        triangleMesh.def(pybind11::init(pybind11::overload_cast<>(&TriangleMesh::create)));
        triangleMesh.def("addVertex", &TriangleMesh::addVertex, "position"_a, "normal"_a, "texCoord"_a);
        triangleMesh.def("addTriangle", &TriangleMesh::addTriangle, "i0"_a, "i1"_a, "i2"_a);
        triangleMesh.def_static("createQuad", &TriangleMesh::createQuad, "size"_a = float2(1.f));
        triangleMesh.def_static("createDisk", &TriangleMesh::createDisk, "radius"_a = 1.f, "segments"_a = 32);
        triangleMesh.def_static("createCube", &TriangleMesh::createCube, "size"_a = float3(1.f));
        triangleMesh.def_static("createSphere", &TriangleMesh::createSphere, "radius"_a = 1.f, "segmentsU"_a = 32, "segmentsV"_a = 32);
        triangleMesh.def_static("createFromFile", &TriangleMesh::createFromFile, "path"_a, "smoothNormals"_a = false);

        pybind11::class_<TriangleMesh::Vertex> vertex(triangleMesh, "Vertex");
        vertex.def_readwrite("position", &TriangleMesh::Vertex::position);
        vertex.def_readwrite("normal", &TriangleMesh::Vertex::normal);
        vertex.def_readwrite("texCoord", &TriangleMesh::Vertex::texCoord);
    }
}
