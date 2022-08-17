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
#include "Core/Macros.h"
#include "Utils/Math/Matrix.h"
#include "Utils/Math/Vector.h"
#include <vector>

namespace Falcor
{
    class FALCOR_API CurveTessellation
    {
    public:
        // Swept spheres

        struct SweptSphereResult
        {
            uint32_t degree;
            std::vector<uint32_t> indices;
            std::vector<float3> points;
            std::vector<float> radius;
            std::vector<float2> texCrds;
        };

        /** Convert cubic B-splines to a couple of linear swept sphere segments.
            \param[in] strandCount Number of curve strands.
            \param[in] vertexCountsPerStrand Number of control points per strand.
            \param[in] controlPoints Array of control points.
            \param[in] widths Array of curve widths, i.e., diameters of swept spheres.
            \param[in] UVs Array of texture coordinates.
            \param[in] degree Polynomial degree of strand (linear -- cubic).
            \param[in] subdivPerSegment Number of sub-segments within each cubic bspline segment (defined by 4 control points).
            \param[in] keepOneEveryXStrands Keep one of every X curve strands.
            \param[in] keepOneEveryXVerticesPerStrand Keep one of every X vertices in each curve strand.
            \param[in] widthScale Global scaling factor for curve width (normally set to 1.0).
            \param[in] xform Row-major 4x4 transformation matrix. We apply pre-transformation to curve geometry.
            \return Linear swept sphere segments.
        */
        static SweptSphereResult convertToLinearSweptSphere(uint32_t strandCount, const uint32_t* vertexCountsPerStrand, const float3* controlPoints, const float* widths, const float2* UVs, uint32_t degree, uint32_t subdivPerSegment, uint32_t keepOneEveryXStrands, uint32_t keepOneEveryXVerticesPerStrand, float widthScale, const rmcv::mat4& xform);

        // Tessellated mesh

        struct MeshResult
        {
            std::vector<float3> vertices;
            std::vector<float3> normals;
            std::vector<float4> tangents;
            std::vector<float2> texCrds;
            std::vector<float> radii;
            std::vector<uint32_t> faceVertexCounts;
            std::vector<uint32_t> faceVertexIndices;
        };

        /** Tessellate cubic B-splines to a triangular mesh.
            \param[in] strandCount Number of curve strands.
            \param[in] vertexCountsPerStrand Number of control points per strand.
            \param[in] controlPoints Array of control points.
            \param[in] widths Array of curve widths, i.e., diameters of swept spheres.
            \param[in] UVs Array of texture coordinates.
            \param[in] subdivPerSegment Number of sub-segments within each cubic bspline segment (defined by 4 control points).
            \param[in] keepOneEveryXStrands Keep one of every X curve strands.
            \param[in] keepOneEveryXVerticesPerStrand Keep one of every X vertices in each curve strand.
            \param[in] widthScale Global scaling factor for curve width (normally set to 1.0).
            \param[in] pointCountPerCrossSection Number of points sampled at each cross-section.
            \return Tessellated mesh.
        */
        static MeshResult convertToPolytube(uint32_t strandCount, const uint32_t* vertexCountsPerStrand, const float3* controlPoints, const float* widths, const float2* UVs, uint32_t subdivPerSegment, uint32_t keepOneEveryXStrands, uint32_t keepOneEveryXVerticesPerStrand, float widthScale, uint32_t pointCountPerCrossSection);


    private:
        CurveTessellation() = default;
        CurveTessellation(const CurveTessellation&) = delete;
        void operator=(const CurveTessellation&) = delete;
    };
}
