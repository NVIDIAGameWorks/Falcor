/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
RWStructuredBuffer<uint> result;

[numthreads(1, 1, 1)]
void testRayFlags()
{
    result[0] = RAY_FLAG_NONE;                              // RayFlags::None
    result[1] = RAY_FLAG_FORCE_OPAQUE;                      // RayFlags::ForceOpaque
    result[2] = RAY_FLAG_FORCE_NON_OPAQUE;                  // RayFlags::ForceNonOpaque
    result[3] = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;   // RayFlags::AcceptFirstHitAndEndSearch
    result[4] = RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;           // RayFlags::SkipClosestHitShader
    result[5] = RAY_FLAG_CULL_BACK_FACING_TRIANGLES;        // RayFlags::CullBackFacingTriangles
    result[6] = RAY_FLAG_CULL_FRONT_FACING_TRIANGLES;       // RayFlags::CullFrontFacingTriangles
    result[7] = RAY_FLAG_CULL_OPAQUE;                       // RayFlags::CullOpaque
    result[8] = RAY_FLAG_CULL_NON_OPAQUE;                   // RayFlags::CullNonOpaque
#ifdef DXR_1_1
    result[9] = RAY_FLAG_SKIP_TRIANGLES;                    // RayFlags::SkipTriangles
    result[10] = RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES;       // RayFlags::SkipProceduralPrimitives
#endif
}
