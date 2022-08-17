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
#include "CurveTessellation.h"
#include "Core/Assert.h"
#include "Utils/Math/Common.h"
#include "Utils/Math/MathHelpers.h"
#include "Utils/Math/CubicSpline.h"
#include "Utils/Math/Matrix/Matrix.h"
#include <glm/gtx/quaternion.hpp>
#include <cmath>

namespace Falcor
{
    struct StrandArrays {
        std::vector<float3> controlPoints;
        std::vector<float>  widths;
        std::vector<float2> UVs;
        uint32_t vertexCount { 0 };
    };

    struct CurveArrays {
        const float3* controlPoints;
        const float* widths;
        const float2* UVs;

        // Initializer
        CurveArrays(const float3* paramControlPoints, const float* paramWidths, const float2* paramUVs)
        {
            controlPoints = paramControlPoints;
            widths = paramWidths;
            UVs = paramUVs;
        }
    };

    struct CubicSplineCache
    {
        CubicSpline<float3> optSplinePoints;
        CubicSpline<float>  optSplineWidths;
        CubicSpline<float2> optSplineUVs;

        CubicSpline<float3> splinePoints;
        CubicSpline<float>  splineWidths;
        CubicSpline<float2> splineUVs;
    };

    namespace
    {
        // Curves tessellated to quad-tubes have the width somewhere between curveWidth and (curveWidth / sqrt(2)), depending on the viewing angle.
        // To achieve curveWidth on average, however, we need to scale the initial curveWidth by 1.11 (the number was deducted numerically).
        const float kMeshCompensationScale = 1.11f;

        float4 transformSphere(const rmcv::mat4& xform, const float4& sphere)
        {
            // Spheres are represented as (center.x, center.y, center.z, radius).
            // Assume the scaling is isotropic, i.e., the end points are still spheres after transformation.
#if 1
            float  scale = std::sqrt(xform[0][0] * xform[0][0] + xform[0][1] * xform[0][1] + xform[0][2] * xform[0][2]);
            float3 xyz = xform * float4(sphere.xyz, 1.f);
            return float4(xyz, sphere.w * scale);
#else
            float3 q = sphere.xyz + float3(sphere.w, 0, 0);
            float4 xp = xform * float4(sphere.xyz, 1.f);
            float4 xq = xform * float4(q, 1.f);
            float xr = glm::length(xq.xyz - xp.xyz);
            return float4(xp.xyz, xr);
#endif
        }

        void optimizeStrandGeometry(CubicSplineCache& splineCache, const CurveArrays& curveArrays, StrandArrays& strandArrays, StrandArrays& optimizedStrandArrays, uint32_t pointOffset, uint32_t subdivPerSegment, uint32_t keepOneEveryXVerticesPerStrand, float widthScale)
        {
            strandArrays.controlPoints.clear();
            strandArrays.UVs.clear();
            strandArrays.widths.clear();

            // Optimize geometry by removing duplicates.
            for (uint32_t j = 0; j < strandArrays.vertexCount - 1; j++)
            {
                if (curveArrays.controlPoints[pointOffset + j] != curveArrays.controlPoints[pointOffset + j + 1])
                {
                    strandArrays.controlPoints.push_back(curveArrays.controlPoints[pointOffset + j]);
                    strandArrays.widths.push_back(curveArrays.widths[pointOffset + j]);
                    if (curveArrays.UVs) strandArrays.UVs.push_back(curveArrays.UVs[pointOffset + j]);
                }
            }

            // Add the last control point.
            strandArrays.controlPoints.push_back(curveArrays.controlPoints[pointOffset + strandArrays.vertexCount - 1]);
            strandArrays.widths.push_back(curveArrays.widths[pointOffset + strandArrays.vertexCount - 1]);
            if (curveArrays.UVs) strandArrays.UVs.push_back(curveArrays.UVs[pointOffset + strandArrays.vertexCount - 1]);

            optimizedStrandArrays.vertexCount = static_cast<uint32_t>(strandArrays.controlPoints.size());

            const CubicSpline<float3>& splinePoints = splineCache.optSplinePoints.setup(strandArrays.controlPoints.data(), optimizedStrandArrays.vertexCount);
            const CubicSpline<float>& splineWidths = splineCache.optSplineWidths.setup(strandArrays.widths.data(), optimizedStrandArrays.vertexCount);

            uint32_t tmpCount = 0;
            for (uint32_t j = 0; j < optimizedStrandArrays.vertexCount - 1; j++)
            {
                for (uint32_t k = 0; k < subdivPerSegment; k++)
                {
                    if (tmpCount % keepOneEveryXVerticesPerStrand == 0)
                    {
                        float t = (float)k / (float)subdivPerSegment;
                        optimizedStrandArrays.controlPoints.push_back(splinePoints.interpolate(j, t));
                        optimizedStrandArrays.widths.push_back(kMeshCompensationScale * widthScale * splineWidths.interpolate(j, t));
                    }
                    tmpCount++;
                }
            }

            // Always keep the last vertex.
            optimizedStrandArrays.controlPoints.push_back(splinePoints.interpolate(optimizedStrandArrays.vertexCount - 2, 1.f));
            optimizedStrandArrays.widths.push_back(kMeshCompensationScale * widthScale * splineWidths.interpolate(optimizedStrandArrays.vertexCount - 2, 1.f));

            // Texture coordinates.
            if (curveArrays.UVs)
            {
                const CubicSpline<float2>& splineUVs = splineCache.optSplineUVs.setup(strandArrays.UVs.data(), optimizedStrandArrays.vertexCount);
                tmpCount = 0;
                for (uint32_t j = 0; j < optimizedStrandArrays.vertexCount - 1; j++)
                {
                    for (uint32_t k = 0; k < subdivPerSegment; k++)
                    {
                        if (tmpCount % keepOneEveryXVerticesPerStrand == 0)
                        {
                            float t = (float)k / (float)subdivPerSegment;
                            optimizedStrandArrays.UVs.push_back(splineUVs.interpolate(j, t));
                        }
                        tmpCount++;
                    }
                }

                // Always keep the last vertex.
                optimizedStrandArrays.UVs.push_back(splineUVs.interpolate(optimizedStrandArrays.vertexCount - 2, 1.f));
            }
        }

        void updateCurveFrame(const StrandArrays& strandArrays, float3& fwd, float3& s, float3& t, uint32_t j)
        {
            float3 prevFwd;

            if (j <= 0 || j >= strandArrays.controlPoints.size())
            {
                // The forward tangents should be the same, meaning s & t are also the same
                prevFwd = fwd;
            }
            else if (j == 1)
            {
                prevFwd = normalize(strandArrays.controlPoints[j] - strandArrays.controlPoints[j - 1]);
                fwd = normalize(strandArrays.controlPoints[j + 1] - strandArrays.controlPoints[j - 1]);
            }
            else if (j < strandArrays.controlPoints.size() - 2)
            {
                prevFwd = normalize(strandArrays.controlPoints[j] - strandArrays.controlPoints[j - 2]);
                fwd = normalize(strandArrays.controlPoints[j + 1] - strandArrays.controlPoints[j - 1]);
            }
            else if (j == strandArrays.controlPoints.size() - 1)
            {
                prevFwd = normalize(strandArrays.controlPoints[j] - strandArrays.controlPoints[j - 2]);
                fwd = normalize(strandArrays.controlPoints[j] - strandArrays.controlPoints[j - 1]);
            }

            // Use quaternions to smoothly rotate the other vectors and update s & t vectors.
            glm::quat rotQuat = glm::rotation(prevFwd, fwd);
            s = glm::rotate(rotQuat, s);
            t = glm::rotate(rotQuat, t);
        }

        void updateMeshResultBuffers(CurveTessellation::MeshResult& result, const CurveArrays& curveArrays, StrandArrays& optimizedStrandArrays, const float3& fwd, const float3& s, const float3& t, uint32_t pointCountPerCrossSection, const float& widthScale, uint32_t j)
        {
            // Mesh vertices, normals, tangents, and texCrds (if any).
            for (uint32_t k = 0; k < pointCountPerCrossSection; k++)
            {
                float phi = (float)k / (float)pointCountPerCrossSection * (float)M_PI * 2.f;
                float3 vNormal = std::cos(phi) * s + std::sin(phi) * t;

                float curveRadius = 0.5f * optimizedStrandArrays.widths[j];
                result.vertices.push_back(optimizedStrandArrays.controlPoints[j] + curveRadius * vNormal);
                result.normals.push_back(vNormal);
                result.tangents.push_back(float4(fwd.x, fwd.y, fwd.z, 1));
                result.radii.push_back(curveRadius);

                if (curveArrays.UVs)
                {
                    result.texCrds.push_back(optimizedStrandArrays.UVs[j]);
                }
            }
        }

        void connectFaceVertices(CurveTessellation::MeshResult& result, uint32_t meshVertexOffset, uint32_t pointCountPerCrossSection, uint32_t quadCountLimit, uint32_t nextCrossSectionVertexOffset, uint32_t multiplier, uint32_t j)
        {
            for (uint32_t k = 0; k < quadCountLimit; k++)
            {
                result.faceVertexCounts.push_back(3);
                result.faceVertexIndices.push_back(meshVertexOffset + multiplier * j * pointCountPerCrossSection + k);
                result.faceVertexIndices.push_back(meshVertexOffset + multiplier * j * pointCountPerCrossSection + (k + nextCrossSectionVertexOffset) % pointCountPerCrossSection);
                result.faceVertexIndices.push_back(meshVertexOffset + (multiplier * j + 1) * pointCountPerCrossSection + (k + nextCrossSectionVertexOffset) % pointCountPerCrossSection);

                result.faceVertexCounts.push_back(3);
                result.faceVertexIndices.push_back(meshVertexOffset + multiplier * j * pointCountPerCrossSection + k);
                result.faceVertexIndices.push_back(meshVertexOffset + (multiplier * j + 1) * pointCountPerCrossSection + (k + nextCrossSectionVertexOffset) % pointCountPerCrossSection);
                result.faceVertexIndices.push_back(meshVertexOffset + (multiplier * j + 1) * pointCountPerCrossSection + k);
            }
        }
    }

    CurveTessellation::SweptSphereResult CurveTessellation::convertToLinearSweptSphere(uint32_t strandCount, const uint32_t* vertexCountsPerStrand, const float3* controlPoints, const float* widths, const float2* UVs, uint32_t degree, uint32_t subdivPerSegment, uint32_t keepOneEveryXStrands, uint32_t keepOneEveryXVerticesPerStrand, float widthScale, const rmcv::mat4& xform)
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
            maxVertexCountsPerStrand = std::max(maxVertexCountsPerStrand, vertexCountsPerStrand[i]);
        }
        result.indices.reserve(segCounts);
        result.points.reserve(pointCounts);
        result.radius.reserve(pointCounts);
        result.texCrds.reserve(pointCounts);

        uint32_t pointOffset = 0;

        StrandArrays strandArrays;
        strandArrays.controlPoints.reserve(maxVertexCountsPerStrand);
        strandArrays.widths.reserve(maxVertexCountsPerStrand);
        strandArrays.UVs.reserve(maxVertexCountsPerStrand);
        CurveArrays curveArrays(controlPoints, widths, UVs);

        StrandArrays optimizedStrandArrays;
        CubicSplineCache splineCache;
        for (uint32_t i = 0; i < strandCount; i += keepOneEveryXStrands)
        {
            optimizedStrandArrays.controlPoints.clear();
            optimizedStrandArrays.UVs.clear();
            optimizedStrandArrays.widths.clear();
            optimizedStrandArrays.vertexCount = 0;
            strandArrays.vertexCount = vertexCountsPerStrand[i];

            optimizeStrandGeometry(splineCache, curveArrays, strandArrays, optimizedStrandArrays, pointOffset, subdivPerSegment, keepOneEveryXVerticesPerStrand, widthScale);

            const CubicSpline<float3>& splinePoints = splineCache.splinePoints.setup(strandArrays.controlPoints.data(), optimizedStrandArrays.vertexCount);
            const CubicSpline<float>& splineWidths = splineCache.splineWidths.setup(strandArrays.widths.data(), optimizedStrandArrays.vertexCount);

            uint32_t tmpCount = 0;
            for (uint32_t j = 0; j < optimizedStrandArrays.vertexCount - 1; j++)
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
            float4 sph = transformSphere(xform, float4(splinePoints.interpolate(optimizedStrandArrays.vertexCount - 2, 1.f), splineWidths.interpolate(optimizedStrandArrays.vertexCount - 2, 1.f) * 0.5f * widthScale));
            result.points.push_back(sph.xyz);
            result.radius.push_back(sph.w);

            // Texture coordinates.
            if (UVs)
            {
                const CubicSpline<float2>& splineUVs = splineCache.splineUVs.setup(strandArrays.UVs.data(), optimizedStrandArrays.vertexCount);
                tmpCount = 0;
                for (uint32_t j = 0; j < optimizedStrandArrays.vertexCount - 1; j++)
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
                result.texCrds.push_back(splineUVs.interpolate(optimizedStrandArrays.vertexCount - 2, 1.f));
            }

            for (uint32_t j = i; j < std::min(strandCount, i + keepOneEveryXStrands); j++) pointOffset += vertexCountsPerStrand[j];
        }

        return result;
    }

    CurveTessellation::MeshResult CurveTessellation::convertToPolytube(uint32_t strandCount, const uint32_t* vertexCountsPerStrand, const float3* controlPoints, const float* widths, const float2* UVs, uint32_t subdivPerSegment, uint32_t keepOneEveryXStrands, uint32_t keepOneEveryXVerticesPerStrand, float widthScale, uint32_t pointCountPerCrossSection)
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
            maxVertexCountsPerStrand = std::max(maxVertexCountsPerStrand, vertexCountsPerStrand[i]);
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

        StrandArrays strandArrays;
        strandArrays.controlPoints.reserve(maxVertexCountsPerStrand);
        strandArrays.widths.reserve(maxVertexCountsPerStrand);
        strandArrays.UVs.reserve(maxVertexCountsPerStrand);
        CurveArrays curveArrays(controlPoints, widths, UVs);

        StrandArrays optimizedStrandArrays;
        CubicSplineCache splineCache;
        for (uint32_t i = 0; i < strandCount; i += keepOneEveryXStrands)
        {
            optimizedStrandArrays.controlPoints.clear();
            optimizedStrandArrays.UVs.clear();
            optimizedStrandArrays.widths.clear();
            optimizedStrandArrays.vertexCount = 0;

            strandArrays.vertexCount = vertexCountsPerStrand[i];

            optimizeStrandGeometry(splineCache, curveArrays, strandArrays, optimizedStrandArrays, pointOffset, subdivPerSegment, keepOneEveryXVerticesPerStrand, widthScale);

            for (uint32_t j = i; j < std::min(strandCount, i + keepOneEveryXStrands); j++) pointOffset += vertexCountsPerStrand[j];

            // Build the initial frame.
            float3 fwd, s, t;
            fwd = normalize(optimizedStrandArrays.controlPoints[1] - optimizedStrandArrays.controlPoints[0]);
            buildFrame(fwd, s, t);

            // Create mesh.
            for (uint32_t j = 0; j < optimizedStrandArrays.controlPoints.size(); j++)
            {
                // Update the curve's frame vectors: [fwd, s, t]
                updateCurveFrame(optimizedStrandArrays, fwd, s, t, j);

                // Mesh vertices, normals, tangents, and texCrds (if any).
                updateMeshResultBuffers(result, curveArrays, optimizedStrandArrays, fwd, s, t, pointCountPerCrossSection, widthScale, j);

                // Mesh faces.
                if (j < optimizedStrandArrays.controlPoints.size() - 1)
                {
                    uint32_t quadCountLimit = pointCountPerCrossSection;
                    connectFaceVertices(result, meshVertexOffset, pointCountPerCrossSection, quadCountLimit, 1, 1, j);
                }
            }

            meshVertexOffset += pointCountPerCrossSection * (uint32_t)optimizedStrandArrays.controlPoints.size();
        }
        return result;
    }


}
