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
#include "stdafx.h"
#include "CurveTessellation.h"
#include "Utils/Math/MathHelpers.h"
#define _USE_MATH_DEFINES
#include <math.h>

namespace Falcor
{
    namespace
    {
        float4 transformSphere(const glm::mat4& xform, const float4& sphere)
        {
            // Spheres are represented as (center.x, center.y, center.z, radius).
            // Assume the scaling is isotropic, i.e., the end points are still spheres after transformation.
            float3 q = sphere.xyz + float3(sphere.w, 0, 0);
            float4 xp = xform * float4(sphere.xyz, 1.f);
            float4 xq = xform * float4(q, 1.f);
            float xr = glm::length(xq - xp);
            return float4(xp.xyz, xr);
        }
    }

    CurveTessellation::SweptSphereResult CurveTessellation::convertToLinearSweptSphere(size_t strandCount, const int* vertexCountsPerStrand, const float3* controlPoints, const float* widths, const float2* UVs, uint32_t degree, uint32_t subdivPerSegment, const glm::mat4& xform)
    {
        SweptSphereResult result;

        // Only support linear tube segments now.
        // TODO: Add quadratic or cubic tube segments if necessary.
        assert(degree == 1);
        result.degree = degree;

        uint32_t pointCounts = 0;
        uint32_t segCounts = 0;
        for (uint32_t i = 0; i < strandCount; i++)
        {
            pointCounts += subdivPerSegment * (vertexCountsPerStrand[i] - 1) + 1;
            segCounts += pointCounts - 1;
        }
        result.indices.reserve(segCounts);
        result.points.reserve(pointCounts);
        result.radius.reserve(pointCounts);
        result.tangents.reserve(pointCounts);
        result.normals.reserve(pointCounts);
        result.texCrds.reserve(pointCounts);

        uint32_t pointOffset = 0;
        for (uint32_t i = 0; i < strandCount; i++)
        {
            CubicSpline strandPoints(controlPoints + pointOffset, vertexCountsPerStrand[i]);
            CubicSpline strandWidths(widths + pointOffset, vertexCountsPerStrand[i]);

            uint32_t resOffset = (uint32_t)result.points.size();
            for (uint32_t j = 0; j < (uint32_t)vertexCountsPerStrand[i] - 1; j++)
            {
                for (uint32_t k = 0; k < subdivPerSegment; k++)
                {
                    float t = (float)k / (float)subdivPerSegment;
                    result.indices.push_back((uint32_t)result.points.size());

                    // Pre-transform curve points.
                    float4 sph = transformSphere(xform, float4(strandPoints.interpolate(j, t), strandWidths.interpolate(j, t) * 0.5f));
                    result.points.push_back(sph.xyz);
                    result.radius.push_back(sph.w);
                }
            }

            float4 sph = transformSphere(xform, float4(strandPoints.interpolate(vertexCountsPerStrand[i] - 2, 1.f), strandWidths.interpolate(vertexCountsPerStrand[i] - 2, 1.f) * 0.5f));
            result.points.push_back(sph.xyz);
            result.radius.push_back(sph.w);

            // Compute tangents and normals.
            for (uint32_t j = resOffset; j < result.points.size(); j++)
            {
                float3 fwd, s, t;
                if (j < result.points.size() - 1)
                {
                    fwd = normalize(result.points[j + 1] - result.points[j]);
                }
                else
                {
                    fwd = normalize(result.points[j] - result.points[j - 1]);
                }
                buildFrame(fwd, s, t);

                result.tangents.push_back(fwd);
                result.normals.push_back(s);
            }

            // Texture coordinates.
            if (UVs)
            {
                CubicSpline strandUVs(UVs + pointOffset, vertexCountsPerStrand[i]);
                for (uint32_t j = 0; j < (uint32_t)vertexCountsPerStrand[i] - 1; j++)
                {
                    for (uint32_t k = 0; k < subdivPerSegment; k++)
                    {
                        float t = (float)k / (float)subdivPerSegment;
                        result.texCrds.push_back(strandUVs.interpolate(j, t));
                    }
                }
                result.texCrds.push_back(strandUVs.interpolate(vertexCountsPerStrand[i] - 2, 1.f));
            }

            pointOffset += vertexCountsPerStrand[i];
        }

        return result;
    }

    CurveTessellation::MeshResult CurveTessellation::convertToMesh(size_t strandCount, const int* vertexCountsPerStrand, const float3* controlPoints, const float* widths, const float2* UVs, uint32_t subdivPerSegment, uint32_t pointCountPerCrossSection)
    {
        MeshResult result;
        uint32_t vertexCounts = 0;
        uint32_t faceCounts = 0;
        for (uint32_t i = 0; i < strandCount; i++)
        {
            vertexCounts += pointCountPerCrossSection * subdivPerSegment * (vertexCountsPerStrand[i] - 1) + 1;
            faceCounts += 2 * pointCountPerCrossSection * subdivPerSegment * (vertexCountsPerStrand[i] - 1);
        }
        result.vertices.reserve(vertexCounts);
        result.normals.reserve(vertexCounts);
        result.tangents.reserve(vertexCounts);
        result.faceVertexCounts.reserve(faceCounts);
        result.faceVertexIndices.reserve(faceCounts * 3);
        result.texCrds.reserve(vertexCounts);

        uint32_t pointOffset = 0;
        uint32_t meshVertexOffset = 0;
        for (uint32_t i = 0; i < strandCount; i++)
        {
            CubicSpline strandPoints(controlPoints + pointOffset, vertexCountsPerStrand[i]);
            CubicSpline strandWidths(widths + pointOffset, vertexCountsPerStrand[i]);

            std::vector<float3> curvePoints;
            std::vector<float> curveRadius;
            std::vector<float2> curveUVs;

            curvePoints.push_back(strandPoints.interpolate(0, 0.f));
            curveRadius.push_back(strandWidths.interpolate(0, 0.f) * 0.5f);

            for (uint32_t j = 0; j < (uint32_t)vertexCountsPerStrand[i] - 1; j++)
            {
                for (uint32_t k = 1; k <= subdivPerSegment; k++)
                {
                    float t = (float)k / (float)subdivPerSegment;
                    curvePoints.push_back(strandPoints.interpolate(j, t));
                    curveRadius.push_back(strandWidths.interpolate(j, t) * 0.5f);
                }
            }

            // Texture coordinates.
            if (UVs)
            {
                CubicSpline strandUVs(UVs + pointOffset, vertexCountsPerStrand[i]);
                curveUVs.push_back(strandUVs.interpolate(0, 0.f));
                for (uint32_t j = 0; j < (uint32_t)vertexCountsPerStrand[i] - 1; j++)
                {
                    for (uint32_t k = 1; k <= subdivPerSegment; k++)
                    {
                        float t = (float)k / (float)subdivPerSegment;
                        curveUVs.push_back(strandUVs.interpolate(j, t));
                    }
                }
            }

            pointOffset += vertexCountsPerStrand[i];

            // Create mesh.
            for (uint32_t j = 0; j < curvePoints.size(); j++)
            {
                float3 fwd, s, t;
                if (j < curvePoints.size() - 1)
                {
                    fwd = normalize(curvePoints[j + 1] - curvePoints[j]);
                }
                else
                {
                    fwd = normalize(curvePoints[j] - curvePoints[j - 1]);
                }
                buildFrame(fwd, s, t);

                // Mesh vertices, normals, tangents, and texCrds (if any).
                for (uint32_t k = 0; k < pointCountPerCrossSection; k++)
                {
                    float phi = (float)k / (float)pointCountPerCrossSection * (float)M_PI * 2.f;
                    float3 vNormal = std::cos(phi) * s + std::sin(phi) * t;

                    result.vertices.push_back(curvePoints[j] + curveRadius[j] * vNormal);
                    result.normals.push_back(vNormal);
                    result.tangents.push_back(float4(fwd.x, fwd.y, fwd.z, 1));

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
            }

            meshVertexOffset += pointCountPerCrossSection * (uint32_t)curvePoints.size();
        }
        return result;
    }
}
