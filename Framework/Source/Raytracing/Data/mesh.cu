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
#include "declarations.cuh"
#include <optixu/optixu_aabb_namespace.h>

RT_PROGRAM void mesh_intersect(int primIdxOffset)
{
    const int3 prim_id = gInstIndices[primIdxOffset];

    const float3 p0 = gInstPositions[prim_id.x];
    const float3 p1 = gInstPositions[prim_id.y];
    const float3 p2 = gInstPositions[prim_id.z];

    // Intersect ray with triangle
    float3 n;
    float t, beta, gamma;
    if(intersect_triangle(ray, p0, p1, p2, n, t, beta, gamma))
    {
        bool reportIntesection = true;
        // Alpha test
        if(alphaTestEnabled(gInstMaterial))
        {
            ShadingAttribs shAttr;
            shAttr.UV = v2(fetchAndInterpolate(gInstTexcoords, prim_id, v2(beta, gamma)));
            if(!alphaTestPassed(gInstMaterial, shAttr))
                reportIntesection = false;
        }
        // Report intersection
        if(reportIntesection && rtPotentialIntersection(t)) 
        {
            gGeoNormal = normalize(n);
            gBarys = v2(beta, gamma);
            gPrimOffset = primIdxOffset;
            gPrimId = prim_id;
            rtReportIntersection(0);
        }
    }
}

RT_PROGRAM void mesh_bounds(int primIdxOffset, float result[6])
{
  const int3 prim_id = gInstIndices[primIdxOffset];

  const float3 v0   = gInstPositions[ prim_id.x ];
  const float3 v1   = gInstPositions[ prim_id.y ];
  const float3 v2   = gInstPositions[ prim_id.z ];
  const float  area = length(cross(v1-v0, v2-v0));

  optix::Aabb* aabb = (optix::Aabb*)result;
  
  if(area > 0.0f && !isinf(area)) {
    aabb->m_min = fminf( fminf( v0, v1), v2 );
    aabb->m_max = fmaxf( fmaxf( v0, v1), v2 );
  } else {
    aabb->invalidate();
  }
}

RT_PROGRAM void exception()
{
  const vec4 bad_color = v4(1e5f, 0.f, 1e5f, 1.f);
  const unsigned int code = rtGetExceptionCode();
  rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
  rtPrintExceptionDetails();
  fbStore<0, vec4>(bad_color);
}
