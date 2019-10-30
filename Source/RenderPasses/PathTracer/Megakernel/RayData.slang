/***************************************************************************
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

/** Declarations of the ray payload structs.

    We have utility functions to pack/unpack the payload to save registers
    and increase performance.
*/

import Utils.Math.MathHelpers;
import Utils.Math.FormatConversion;
__exported import Utils.Sampling.SampleGenerator;
__exported import RenderPasses.PathTracer.PathTracerHelpers;

/** Payload for shadow ray.
*/
struct ShadowRayData
{
    bool visible;
};

/** Unpacked payload for scatter ray.
*/
struct ScatterRayData
{
    bool terminated;        ///< Set to true when path is terminated.
    uint pathLength;        ///< Path length in number of path segments (0 at origin, 1 at first secondary hit, etc.).

    float3 Le;              ///< Emitted radiance at the last path vertex, weighted by path throughput and MIS weight.
    float3 Lr;              ///< Reflected radiance from the sampled light at the last path vertex, weighted by path throughput, BRDF, and MIS weight.

    float3 origin;          ///< Next path segment origin (unless path is terminated).
    float3 direction;       ///< Next path segment direction (unless path is terminated).

    float3 thp;             ///< Current path throughput. This is updated at each path vertex.

    float3 normal;          ///< Shading normal at the ray origin. Only used if MIS is enabled.
    float pdf;              ///< Probability density function with respect to solid angle at the last path vertex. Only used if MIS is enabled.

    float4 shadowRay;       ///< Shadow ray direction (xyz) and distance (w). Only used if Lr > 0.

    SampleGenerator sg;     ///< Per-ray state for the sample generator (up to 16B).

    /** Create ray payload with default parameters.
    */
    static ScatterRayData create(SampleGenerator sg)
    {
        ScatterRayData d;
        d.terminated = false;
        d.pathLength = 0;
        d.Le = float3(0, 0, 0);
        d.Lr = float3(0, 0, 0);
        d.origin = float3(0, 0, 0);
        d.direction = float3(0, 0, 0);
        d.thp = float3(1, 1, 1);
        d.normal = float3(0, 0, 0);
        d.pdf = 0.f;
        d.shadowRay = float4(0, 0, 0, 0);
        d.sg = sg;

        return d;
    }
};

/** Packed payload for scatter ray.

    For performance reasons it's recommended to keep the payload as small as possible.
    We store the payload in packed format, and have helper function to pack/unpack it.
    Note that we use a compressed format for the ray direction (16-bit precision).
*/
struct ScatterRayDataPacked
{
    float4 packedData[5];   ///< Packed payload (80B).
    SampleGenerator sg;     ///< Sample generator state (up to 16B).

    /** Pack payload.
        \param[in] data The unpacked payload data to be stored in the struct.
    */
    [mutating] void pack(ScatterRayData data)
    {
        // Pack direction as 3x 16-bit snorm and flags/length in remaining 16 bits.
        // TODO: We could convert the direction to spherical coordinates first and pack as 2x 16-bit values, but avoid if necessary (precision/perf loss).
        uint flags = ((data.pathLength & kMaxPathLength) << 1) | (data.terminated ? 1 : 0);
        uint packed0 = (flags << 16) | packSnorm16(data.direction.x);
        uint packed1 = packSnorm2x16(data.direction.yz);

        // Compress shading normal as 2x 16-bit snorms values in the octahedral mapping.
        // Note that there is some loss which we should adjust for where the normal is used.
        float2 octNormal = ndir_to_oct_snorm(data.normal);
        uint packed2 = packSnorm2x16(octNormal);

        packedData[0].xyz = data.Le;
        packedData[0].w = asfloat(packed0);
        packedData[1].xyz = data.Lr;
        packedData[1].w = asfloat(packed1);
        packedData[2].xyz = data.origin;
        packedData[2].w = data.pdf;
        packedData[3].xyz = data.thp;
        packedData[3].w = asfloat(packed2);
        packedData[4].xyzw = data.shadowRay;
        sg = data.sg;
    }

    /** Unpack payload.
        \return The unpacked payload data.
    */
    ScatterRayData unpack()
    {
        ScatterRayData rayData;

        uint packed0 = asuint(packedData[0].w);
        uint packed1 = asuint(packedData[1].w);
        uint flags = (packed0 >> 16);

        // Decompress shading normal.
        uint packed2 = asuint(packedData[3].w);
        float2 octNormal = unpackSnorm2x16(packed2);
        float3 normal = oct_to_ndir_snorm(octNormal);

        rayData.terminated = (flags & 1) != 0;
        rayData.pathLength = (flags >> 1) & kMaxPathLength;
        rayData.direction.x = unpackSnorm16(packed0);
        rayData.direction.yz = unpackSnorm2x16(packed1);
        rayData.direction = normalize(rayData.direction);
        rayData.Le = packedData[0].xyz;
        rayData.Lr = packedData[1].xyz;
        rayData.origin = packedData[2].xyz;
        rayData.pdf = packedData[2].w;
        rayData.thp = packedData[3].xyz;
        rayData.normal = normal;
        rayData.shadowRay = packedData[4].xyzw;
        rayData.sg = sg;

        return rayData;
    }
};
