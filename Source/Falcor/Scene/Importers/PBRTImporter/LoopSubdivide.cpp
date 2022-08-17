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

// This code is based on pbrt:
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include "LoopSubdivide.h"
#include "Core/Assert.h"
#include "Core/Errors.h"

#include <algorithm>
#include <map>
#include <memory>
#include <memory_resource>
#include <set>

#include <cmath>

namespace Falcor
{
    namespace pbrt
    {
        struct SDFace;
        struct SDVertex;

        #define NEXT(i) (((i) + 1) % 3)
        #define PREV(i) (((i) + 2) % 3)

        struct SDVertex
        {
            SDVertex(const float3& p = float3(0.f)) : p(p) {}

            int valence();
            void oneRing(float3 *p);

            float3 p;
            SDFace* startFace = nullptr;
            SDVertex* child = nullptr;
            bool regular = false;
            bool boundary = false;
        };

        struct SDFace
        {
            SDFace()
            {
                for (uint32_t i = 0; i < 3; ++i)
                {
                    v[i] = nullptr;
                    f[i] = nullptr;
                }
                for (uint32_t i = 0; i < 4; ++i)
                {
                    children[i] = nullptr;
                }
            }

            uint32_t vnum(SDVertex* vert) const
            {
                for (int i = 0; i < 3; ++i)
                {
                    if (v[i] == vert) return i;
                }
                throw RuntimeError("Basic logic error in SDFace::vnum().");
            }

            SDFace* nextFace(SDVertex* vert) const { return f[vnum(vert)]; }
            SDFace* prevFace(SDVertex* vert) const { return f[PREV(vnum(vert))]; }
            SDVertex* nextVert(SDVertex* vert) const { return v[NEXT(vnum(vert))]; }
            SDVertex* prevVert(SDVertex* vert) const { return v[PREV(vnum(vert))]; }
            SDVertex* otherVert(SDVertex* v0, SDVertex* v1)
            {
                for (uint32_t i = 0; i < 3; ++i)
                {
                    if (v[i] != v0 && v[i] != v1) return v[i];
                }
                throw RuntimeError("Basic logic error in SDFace::otherVert()");
            }

            SDVertex* v[3];
            SDFace* f[3];
            SDFace* children[4];
        };

        struct SDEdge
        {
            SDEdge(SDVertex* v0 = nullptr, SDVertex* v1 = nullptr)
            {
                v[0] = std::min(v0, v1);
                v[1] = std::max(v0, v1);
                f[0] = f[1] = nullptr;
                f0edgeNum = -1;
            }

            bool operator<(const SDEdge &e2) const
            {
                if (v[0] == e2.v[0]) return v[1] < e2.v[1];
                return v[0] < e2.v[0];
            }

            SDVertex* v[2];
            SDFace* f[2];
            int f0edgeNum;
        };

        static float3 weightOneRing(SDVertex* vert, float beta);
        static float3 weightBoundary(SDVertex* vert, float beta);

        inline int SDVertex::valence()
        {
            SDFace* f = startFace;
            if (!boundary)
            {
                // Compute valence of interior vertex.
                int nf = 1;
                while ((f = f->nextFace(this)) != startFace) ++nf;
                return nf;
            }
            else
            {
                // Compute valence of boundary vertex
                int nf = 1;
                while ((f = f->nextFace(this)) != nullptr) ++nf;
                f = startFace;
                while ((f = f->prevFace(this)) != nullptr) ++nf;
                return nf + 1;
            }
        }

        inline float beta(uint32_t valence)
        {
            if (valence == 3)
                return 3.f / 16.f;
            else
                return 3.f / (8.f * valence);
        }

        inline float loopGamma(uint32_t valence)
        {
            return 1.f / (valence + 3.f / (8.f * beta(valence)));
        }

        LoopSubdivideResult loopSubdivide(uint32_t levels, fstd::span<const float3> positions, fstd::span<const uint32_t> indices)
        {
            std::vector<SDVertex*> vertices;
            std::vector<SDFace*> faces;

            // Allocate vertices and faces.
            std::unique_ptr<SDVertex[]> verts = std::make_unique<SDVertex[]>(positions.size());
            for (size_t i = 0; i < positions.size(); ++i)
            {
                verts[i] = SDVertex(positions[i]);
                vertices.push_back(&verts[i]);
            }
            size_t faceCount = indices.size() / 3;
            std::unique_ptr<SDFace[]> fs = std::make_unique<SDFace[]>(faceCount);
            for (size_t i = 0; i < faceCount; ++i)
            {
                faces.push_back(&fs[i]);
            }

            // Set face to vertex pointers.
            const uint32_t* vp = indices.data();
            for (size_t i = 0; i < faceCount; ++i, vp += 3)
            {
                SDFace* f = faces[i];
                for (uint32_t j = 0; j < 3; ++j)
                {
                    SDVertex* v = vertices[vp[j]];
                    f->v[j] = v;
                    v->startFace = f;
                }
            }

            // Set neighbor pointers in faces.
            std::set<SDEdge> edges;
            for (size_t i = 0; i < faceCount; ++i)
            {
                SDFace* f = faces[i];
                for (uint32_t edgeNum = 0; edgeNum < 3; ++edgeNum)
                {
                    // Update neighbor pointer for edgeNum.
                    int v0 = edgeNum, v1 = NEXT(edgeNum);
                    SDEdge e(f->v[v0], f->v[v1]);
                    if (edges.find(e) == edges.end())
                    {
                        // Handle new edge.
                        e.f[0] = f;
                        e.f0edgeNum = edgeNum;
                        edges.insert(e);
                    }
                    else
                    {
                        // Handle previously seen edge.
                        e =* edges.find(e);
                        e.f[0]->f[e.f0edgeNum] = f;
                        f->f[edgeNum] = e.f[0];
                        edges.erase(e);
                    }
                }
            }

            // Finish vertex initialization.
            for (size_t i = 0; i < positions.size(); ++i)
            {
                SDVertex* v = vertices[i];
                SDFace* f = v->startFace;
                do
                {
                    f = f->nextFace(v);
                } while ((f != nullptr) && f != v->startFace);
                v->boundary = (f == nullptr);
                if (!v->boundary && v->valence() == 6) v->regular = true;
                else if (v->boundary && v->valence() == 4) v->regular = true;
                else v->regular = false;
            }

            // Refine LoopSubdiv into triangles.
            std::vector<SDFace*> f = faces;
            std::vector<SDVertex*> v = vertices;

            std::pmr::monotonic_buffer_resource buffer;
            std::pmr::polymorphic_allocator<SDVertex> vertexAllocator(&buffer);
            std::pmr::polymorphic_allocator<SDFace> faceAllocator(&buffer);

            for (size_t i = 0; i < levels; ++i)
            {
                // Update f and v for next level of subdivision.
                std::vector<SDFace*> newFaces;
                std::vector<SDVertex*> newVertices;

                // Allocate next level of children in mesh tree.
                for (SDVertex* vertex : v)
                {
                    vertex->child = vertexAllocator.allocate(1);
                    vertex->child->regular = vertex->regular;
                    vertex->child->boundary = vertex->boundary;
                    newVertices.push_back(vertex->child);
                }
                for (SDFace* face : f)
                {
                    for (uint32_t k = 0; k < 4; ++k)
                    {
                        face->children[k] = faceAllocator.allocate(1);
                        newFaces.push_back(face->children[k]);
                    }
                }

                // Update vertex positions and create new edge vertices.

                // Update vertex positions for even vertices.
                for (SDVertex* vertex : v)
                {
                    if (!vertex->boundary)
                    {
                        // Apply one-ring rule for even vertex.
                        if (vertex->regular) vertex->child->p = weightOneRing(vertex, 1.f / 16.f);
                        else vertex->child->p = weightOneRing(vertex, beta(vertex->valence()));
                    }
                    else
                    {
                        // Apply boundary rule for even vertex.
                        vertex->child->p = weightBoundary(vertex, 1.f / 8.f);
                    }
                }

                // Compute new odd edge vertices.
                std::map<SDEdge, SDVertex*> edgeVerts;
                for (SDFace* face : f)
                {
                    for (uint32_t k = 0; k < 3; ++k)
                    {
                        // Compute odd vertex on kth edge.
                        SDEdge edge(face->v[k], face->v[NEXT(k)]);
                        SDVertex* vert = edgeVerts[edge];
                        if (vert == nullptr)
                        {
                            // Create and initialize new odd vertex
                            vert = vertexAllocator.allocate(1);
                            newVertices.push_back(vert);
                            vert->regular = true;
                            vert->boundary = (face->f[k] == nullptr);
                            vert->startFace = face->children[3];

                            // Apply edge rules to compute new vertex position
                            if (vert->boundary)
                            {
                                vert->p = 0.5f * edge.v[0]->p;
                                vert->p += 0.5f * edge.v[1]->p;
                            }
                            else
                            {
                                vert->p = 3.f / 8.f * edge.v[0]->p;
                                vert->p += 3.f / 8.f * edge.v[1]->p;
                                vert->p += 1.f / 8.f * face->otherVert(edge.v[0], edge.v[1])->p;
                                vert->p += 1.f / 8.f * face->f[k]->otherVert(edge.v[0], edge.v[1])->p;
                            }
                            edgeVerts[edge] = vert;
                        }
                    }
                }

                // Update new mesh topology.

                // Update even vertex face pointers.
                for (SDVertex* vertex : v)
                {
                    int vertNum = vertex->startFace->vnum(vertex);
                    vertex->child->startFace = vertex->startFace->children[vertNum];
                }

                // Update face neighbor pointers.
                for (SDFace* face : f)
                {
                    for (uint32_t j = 0; j < 3; ++j)
                    {
                        // Update children f pointers for siblings.
                        face->children[3]->f[j] = face->children[NEXT(j)];
                        face->children[j]->f[NEXT(j)] = face->children[3];

                        // Update children f pointers for neighbor children.
                        SDFace* f2 = face->f[j];
                        face->children[j]->f[j] = f2 != nullptr ? f2->children[f2->vnum(face->v[j])] : nullptr;
                        f2 = face->f[PREV(j)];
                        face->children[j]->f[PREV(j)] = f2 != nullptr ? f2->children[f2->vnum(face->v[j])] : nullptr;
                    }
                }

                // Update face vertex pointers.
                for (SDFace* face : f)
                {
                    for (uint32_t j = 0; j < 3; ++j)
                    {
                        // Update child vertex pointer to new even vertex
                        face->children[j]->v[j] = face->v[j]->child;

                        // Update child vertex pointer to new odd vertex
                        SDVertex *vert = edgeVerts[SDEdge(face->v[j], face->v[NEXT(j)])];
                        face->children[j]->v[NEXT(j)] = vert;
                        face->children[NEXT(j)]->v[j] = vert;
                        face->children[3]->v[j] = vert;
                    }
                }

                // Prepare for next level of subdivision
                f = newFaces;
                v = newVertices;
            }

            // Push vertices to limit surface.
            std::vector<float3> pLimit(v.size());
            for (size_t i = 0; i < v.size(); ++i)
            {
                if (v[i]->boundary) pLimit[i] = weightBoundary(v[i], 1.f / 5.f);
                else pLimit[i] = weightOneRing(v[i], loopGamma(v[i]->valence()));
            }
            for (size_t i = 0; i < v.size(); ++i)
            {
                v[i]->p = pLimit[i];
            }

            // Compute vertex tangents on limit surface.
            std::vector<float3> Ns;
            Ns.reserve(v.size());
            std::vector<float3> pRing(16, float3());
            for (SDVertex* vertex : v)
            {
                float3 S(0.f);
                float3 T(0.f);
                uint32_t valence = vertex->valence();
                if (valence > pRing.size()) pRing.resize(valence);
                vertex->oneRing(&pRing[0]);
                if (!vertex->boundary)
                {
                    // Compute tangents of interior face
                    for (uint32_t j = 0; j < valence; ++j)
                    {
                        S += std::cos(2.f * float(M_PI) * j / valence) * float3(pRing[j]);
                        T += std::sin(2.f * float(M_PI) * j / valence) * float3(pRing[j]);
                    }
                }
                else
                {
                    // Compute tangents of boundary face
                    S = pRing[valence - 1] - pRing[0];
                    if (valence == 2)
                    {
                        T = float3(pRing[0] + pRing[1] - 2.f * vertex->p);
                    }
                    else if (valence == 3)
                    {
                        T = pRing[1] - vertex->p;
                    }
                    else if (valence == 4) // regular
                    {
                        T = float3(-1.f * pRing[0] + 2.f * pRing[1] + 2.f * pRing[2] + -1.f * pRing[3] + -2.f * vertex->p);
                    }
                    else
                    {
                        float theta = float(M_PI) / float(valence - 1);
                        T = float3(std::sin(theta) * (pRing[0] + pRing[valence - 1]));
                        for (uint32_t k = 1; k < valence - 1; ++k)
                        {
                            float wt = (2 * std::cos(theta) - 2) * std::sin((k)*theta);
                            T += float3(wt * pRing[k]);
                        }
                        T = -T;
                    }
                }
                Ns.push_back(cross(S, T));
            }

            // Create triangle mesh from subdivision mesh
            {
                size_t ntris = f.size();
                std::vector<uint32_t> verts(3 * ntris);
                uint32_t *vp = verts.data();
                uint32_t totVerts = (uint32_t)v.size();
                std::map<SDVertex*, uint32_t> usedVerts;
                for (uint32_t i = 0; i < totVerts; ++i)
                {
                    usedVerts[v[i]] = i;
                }
                for (size_t i = 0; i < ntris; ++i)
                {
                    for (uint32_t j = 0; j < 3; ++j)
                    {
                        *vp = usedVerts[f[i]->v[j]];
                        ++vp;
                    }
                }

                LoopSubdivideResult result;
                result.positions = std::move(pLimit);
                result.normals = std::move(Ns);
                result.indices = std::move(verts);
                return result;
            }
        }

        static float3 weightOneRing(SDVertex* vert, float beta)
        {
            // Put vert one-ring in pRing.
            uint32_t valence = vert->valence();
            FALCOR_ASSERT(valence < 16);
            float3 pRing[16];

            vert->oneRing(pRing);
            float3 p = (1 - valence * beta) * vert->p;
            for (uint32_t i = 0; i < valence; ++i)
            {
                p += beta * pRing[i];
            }
            return p;
        }

        void SDVertex::oneRing(float3* p)
        {
            if (!boundary)
            {
                // Get one-ring vertices for interior vertex.
                SDFace* face = startFace;
                do
                {
                    *p++ = face->nextVert(this)->p;
                    face = face->nextFace(this);
                } while (face != startFace);
            }
            else
            {
                // Get one-ring vertices for boundary vertex.
                SDFace* face = startFace;
                SDFace* f2;
                while ((f2 = face->nextFace(this)) != nullptr)
                {
                    face = f2;
                }
                *p++ = face->nextVert(this)->p;
                do
                {
                    *p++ = face->prevVert(this)->p;
                    face = face->prevFace(this);
                } while (face != nullptr);
            }
        }

        static float3 weightBoundary(SDVertex* vert, float beta)
        {
            // Put vert one-ring in pRing.
            uint32_t valence = vert->valence();
            FALCOR_ASSERT(valence < 16);
            float3 pRing[16];

            vert->oneRing(pRing);
            float3 p = (1 - 2 * beta) * vert->p;
            p += beta * pRing[0];
            p += beta * pRing[valence - 1];
            return p;
        }
    }
}
