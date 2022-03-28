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
#include "stdafx.h"
#include "CurveTessellation.h"
#include "Utils/Math/MathHelpers.h"
#define _USE_MATH_DEFINES
#include <math.h>

namespace Falcor
{
    namespace
    {
        // Curves tessellated to quad-tubes have the width somewhere between curveWidth and (curveWidth / sqrt(2)), depending on the viewing angle.
        // This scaling factor is trying to bring their width on average back to curveWidth.
        const float kMeshCompensationScale = 1.207f;

        float4 transformSphere(const glm::mat4& xform, const float4& sphere)
        {
            // Spheres are represented as (center.x, center.y, center.z, radius).
            // Assume the scaling is isotropic, i.e., the end points are still spheres after transformation.
            float3 q = sphere.xyz + float3(sphere.w, 0, 0);
            float4 xp = xform * float4(sphere.xyz, 1.f);
            float4 xq = xform * float4(q, 1.f);
            float xr = glm::length(xq.xyz - xp.xyz);
            return float4(xp.xyz, xr);
        }
    }

    CurveTessellation::SweptSphereResult CurveTessellation::convertToLinearSweptSphere(size_t strandCount, const int* vertexCountsPerStrand, const float3* controlPoints, const float* widths, const float2* UVs, uint32_t degree, uint32_t subdivPerSegment, uint32_t keepOneEveryXStrands, uint32_t keepOneEveryXVerticesPerStrand, float widthScale, const glm::mat4& xform)
    {
        SweptSphereResult result;

        // Only support linear tube segments now.
        // TODO: Add quadratic or cubic tube segments if necessary.
        FALCOR_ASSERT(degree == 1);
        result.degree = degree;

        uint32_t pointCounts = 0;
        uint32_t segCounts = 0;
        uint32_t maxVertexCountsPerStrand = 0;
        for (uint32_t i = 0; i < strandCount; i += keepOneEveryXStrands)
        {
            uint32_t tmpPointCount = div_round_up(subdivPerSegment * (vertexCountsPerStrand[i] - 1), keepOneEveryXVerticesPerStrand) + 1;
            pointCounts += tmpPointCount;
            segCounts += tmpPointCount - 1;
            maxVertexCountsPerStrand = std::max(maxVertexCountsPerStrand, static_cast<uint32_t>(vertexCountsPerStrand[i]));
        }
        result.indices.reserve(segCounts);
        result.points.reserve(pointCounts);
        result.radius.reserve(pointCounts);
        result.texCrds.reserve(pointCounts);

        uint32_t pointOffset = 0;

        std::vector<float3> strandControlPoints;
        std::vector<float> strandWidths;
        std::vector<float2> strandUVs;

        strandControlPoints.reserve(maxVertexCountsPerStrand);
        strandWidths.reserve(maxVertexCountsPerStrand);
        strandUVs.reserve(maxVertexCountsPerStrand);

        for (uint32_t i = 0; i < strandCount; i += keepOneEveryXStrands)
        {
            strandControlPoints.clear();
            strandWidths.clear();
            strandUVs.clear();

            // Optimize geometry by removing duplicates.
            for (uint32_t j = 0; j < (uint32_t)vertexCountsPerStrand[i] - 1; j++)
            {
                if (controlPoints[pointOffset + j] != controlPoints[pointOffset + j + 1])
                {
                    strandControlPoints.push_back(controlPoints[pointOffset + j]);
                    strandWidths.push_back(widths[pointOffset + j]);
                    if (UVs) strandUVs.push_back(UVs[pointOffset + j]);
                }
            }

            // Add the last control point.
            strandControlPoints.push_back(controlPoints[pointOffset + vertexCountsPerStrand[i] - 1]);
            strandWidths.push_back(widths[pointOffset + vertexCountsPerStrand[i] - 1]);
            if (UVs) strandUVs.push_back(UVs[pointOffset + vertexCountsPerStrand[i] - 1]);

            uint32_t optimizedVertexCount = static_cast<uint32_t>(strandControlPoints.size());

            CubicSpline splinePoints(strandControlPoints.data(), optimizedVertexCount);
            CubicSpline splineWidths(strandWidths.data(), optimizedVertexCount);

            uint32_t tmpCount = 0;
            for (uint32_t j = 0; j < optimizedVertexCount - 1; j++)
            {
                for (uint32_t k = 0; k < subdivPerSegment; k++)
                {
                    if (tmpCount % keepOneEveryXVerticesPerStrand == 0)
                    {
                        float t = (float)k / (float)subdivPerSegment;
                        result.indices.push_back((uint32_t)result.points.size());

                        // Pre-transform curve points.
                        float4 sph = transformSphere(xform, float4(splinePoints.interpolate(j, t), splineWidths.interpolate(j, t) * 0.5f * widthScale));

                        result.points.push_back(sph.xyz);
                        result.radius.push_back(sph.w);
                    }
                    tmpCount++;
                }
            }

            // Always keep the last vertex.
            float4 sph = transformSphere(xform, float4(splinePoints.interpolate(optimizedVertexCount - 2, 1.f), splineWidths.interpolate(optimizedVertexCount - 2, 1.f) * 0.5f * widthScale));
            result.points.push_back(sph.xyz);
            result.radius.push_back(sph.w);

            // Texture coordinates.
            if (UVs)
            {
                CubicSpline splineUVs(strandUVs.data(), optimizedVertexCount);
                tmpCount = 0;
                for (uint32_t j = 0; j < optimizedVertexCount - 1; j++)
                {
                    for (uint32_t k = 0; k < subdivPerSegment; k++)
                    {
                        if (tmpCount % keepOneEveryXVerticesPerStrand == 0)
                        {
                            float t = (float)k / (float)subdivPerSegment;
                            result.texCrds.push_back(splineUVs.interpolate(j, t));
                        }
                        tmpCount++;
                    }
                }

                // Always keep the last vertex.
                result.texCrds.push_back(splineUVs.interpolate(optimizedVertexCount - 2, 1.f));
            }

            for (uint32_t j = i; j < std::min((uint32_t)strandCount, i + keepOneEveryXStrands); j++) pointOffset += vertexCountsPerStrand[j];
        }

        return result;
    }

    CurveTessellation::MeshResult CurveTessellation::convertToMesh(size_t strandCount, const int* vertexCountsPerStrand, const float3* controlPoints, const float* widths, const float2* UVs, uint32_t subdivPerSegment, uint32_t keepOneEveryXStrands, uint32_t keepOneEveryXVerticesPerStrand, float widthScale, uint32_t pointCountPerCrossSection)
    {
        MeshResult result;
        uint32_t vertexCounts = 0;
        uint32_t faceCounts = 0;
        uint32_t maxVertexCountsPerStrand = 0;
        for (uint32_t i = 0; i < strandCount; i += keepOneEveryXStrands)
        {
            uint32_t tmpPointCount = div_round_up(subdivPerSegment * (vertexCountsPerStrand[i] - 1), keepOneEveryXVerticesPerStrand) + 1;
            vertexCounts += pointCountPerCrossSection * tmpPointCount;
            faceCounts += 2 * pointCountPerCrossSection * (tmpPointCount - 1);
            maxVertexCountsPerStrand = std::max(maxVertexCountsPerStrand, static_cast<uint32_t>(vertexCountsPerStrand[i]));
        }
        result.vertices.reserve(vertexCounts);
        result.normals.reserve(vertexCounts);
        result.tangents.reserve(vertexCounts);
        result.texCrds.reserve(vertexCounts);
        result.radii.reserve(vertexCounts);
        result.faceVertexCounts.reserve(faceCounts);
        result.faceVertexIndices.reserve(faceCounts * 3);

        uint32_t pointOffset = 0;
        uint32_t meshVertexOffset = 0;

        std::vector<float3> strandControlPoints;
        std::vector<float> strandWidths;
        std::vector<float2> strandUVs;

        strandControlPoints.reserve(maxVertexCountsPerStrand);
        strandWidths.reserve(maxVertexCountsPerStrand);
        strandUVs.reserve(maxVertexCountsPerStrand);

        for (uint32_t i = 0; i < strandCount; i += keepOneEveryXStrands)
        {
            strandControlPoints.clear();
            strandWidths.clear();
            strandUVs.clear();

            // Optimize geometry by removing duplicates.
            for (uint32_t j = 0; j < (uint32_t)vertexCountsPerStrand[i] - 1; j++)
            {
                if (controlPoints[pointOffset + j] != controlPoints[pointOffset + j + 1])
                {
                    strandControlPoints.push_back(controlPoints[pointOffset + j]);
                    strandWidths.push_back(widths[pointOffset + j]);
                    if (UVs) strandUVs.push_back(UVs[pointOffset + j]);
                }
            }

            // Add the last control point.
            strandControlPoints.push_back(controlPoints[pointOffset + vertexCountsPerStrand[i] - 1]);
            strandWidths.push_back(widths[pointOffset + vertexCountsPerStrand[i] - 1]);
            if (UVs) strandUVs.push_back(UVs[pointOffset + vertexCountsPerStrand[i] - 1]);

            uint32_t optimizedVertexCount = static_cast<uint32_t>(strandControlPoints.size());

            CubicSpline splinePoints(strandControlPoints.data(), optimizedVertexCount);
            CubicSpline splineWidths(strandWidths.data(), optimizedVertexCount);

            std::vector<float3> curvePoints;
            std::vector<float> curveWidths;

            uint32_t tmpCount = 0;
            for (uint32_t j = 0; j < optimizedVertexCount - 1; j++)
            {
                for (uint32_t k = 0; k < subdivPerSegment; k++)
                {
                    if (tmpCount % keepOneEveryXVerticesPerStrand == 0)
                    {
                        float t = (float)k / (float)subdivPerSegment;
                        curvePoints.push_back(splinePoints.interpolate(j, t));
                        curveWidths.push_back(kMeshCompensationScale * widthScale * splineWidths.interpolate(j, t));
                    }
                    tmpCount++;
                }
            }

            // Always keep the last vertex.
            curvePoints.push_back(splinePoints.interpolate(optimizedVertexCount - 2, 1.f));
            curveWidths.push_back(kMeshCompensationScale * widthScale * splineWidths.interpolate(optimizedVertexCount - 2, 1.f));

            std::vector<float2> curveUVs;

            // Texture coordinates.
            if (UVs)
            {
                CubicSpline splineUVs(strandUVs.data(), optimizedVertexCount);
                tmpCount = 0;
                for (uint32_t j = 0; j < optimizedVertexCount - 1; j++)
                {
                    for (uint32_t k = 0; k < subdivPerSegment; k++)
                    {
                        if (tmpCount % keepOneEveryXVerticesPerStrand == 0)
                        {
                            float t = (float)k / (float)subdivPerSegment;
                            curveUVs.push_back(splineUVs.interpolate(j, t));
                        }
                        tmpCount++;
                    }
                }

                // Always keep the last vertex.
                curveUVs.push_back(splineUVs.interpolate(optimizedVertexCount - 2, 1.f));
            }

            for (uint32_t j = i; j < std::min((uint32_t)strandCount, i + keepOneEveryXStrands); j++) pointOffset += vertexCountsPerStrand[j];

            // Build the initial frame.
            float3 prevFwd, s, t;
            prevFwd = normalize(curvePoints[1] - curvePoints[0]);
            buildFrame(prevFwd, s, t);

            // Create mesh.
            for (uint32_t j = 0; j < curvePoints.size(); j++)
            {
                float3 fwd;

                if (j == 0)
                {
                    fwd = prevFwd;
                }
                else if (j < curvePoints.size() - 1)
                {
                    fwd = normalize(curvePoints[j + 1] - curvePoints[j]);
                }
                else
                {
                    fwd = normalize(curvePoints[j] - curvePoints[j - 1]);
                }

                // Use quaternions to smoothly rotate the other vectors.
                glm::quat rotQuat = glm::rotation(prevFwd, fwd);
                s = glm::rotate(rotQuat, s);
                t = glm::rotate(rotQuat, t);

                // Mesh vertices, normals, tangents, and texCrds (if any).
                for (uint32_t k = 0; k < pointCountPerCrossSection; k++)
                {
                    float phi = (float)k / (float)pointCountPerCrossSection * (float)M_PI * 2.f;
                    float3 vNormal = std::cos(phi) * s + std::sin(phi) * t;

                    float curveRadius = 0.5f * curveWidths[j];
                    result.vertices.push_back(curvePoints[j] + curveRadius * vNormal);
                    result.normals.push_back(vNormal);
                    result.tangents.push_back(float4(fwd.x, fwd.y, fwd.z, 1));
                    result.radii.push_back(curveRadius);

                    if (UVs)
                    {
                        result.texCrds.push_back(curveUVs[j]);
                    }
                }

                // Mesh faces.
                if (j < curvePoints.size() - 1)
                {
                    for (uint32_t k = 0; k < pointCountPerCrossSection; k++)
                    {
                        result.faceVertexCounts.push_back(3);
                        result.faceVertexIndices.push_back(meshVertexOffset + j * pointCountPerCrossSection + k);
                        result.faceVertexIndices.push_back(meshVertexOffset + j * pointCountPerCrossSection + (k + 1) % pointCountPerCrossSection);
                        result.faceVertexIndices.push_back(meshVertexOffset + (j + 1) * pointCountPerCrossSection + (k + 1) % pointCountPerCrossSection);

                        result.faceVertexCounts.push_back(3);
                        result.faceVertexIndices.push_back(meshVertexOffset + j * pointCountPerCrossSection + k);
                        result.faceVertexIndices.push_back(meshVertexOffset + (j + 1) * pointCountPerCrossSection + (k + 1) % pointCountPerCrossSection);
                        result.faceVertexIndices.push_back(meshVertexOffset + (j + 1) * pointCountPerCrossSection + k);
                    }
                }

                prevFwd = fwd;
            }

            meshVertexOffset += pointCountPerCrossSection * (uint32_t)curvePoints.size();
        }
        return result;
    }
}
