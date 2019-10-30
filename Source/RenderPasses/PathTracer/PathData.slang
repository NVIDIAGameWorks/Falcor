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
__exported import Utils.Sampling.SampleGenerator;
__exported import RenderPasses.PathTracer.StaticParams;

static const uint kInvalidIndex = 0xffffffff;

/** Ray hit information. This uniquely identifies a hit point.
*/
struct HitInfo
{
    uint    meshInstanceID;     ///< Mesh instance ID for ray hit, or kInvalidIndex if miss.
    uint    primitiveIndex;     ///< Primitive index for ray hit, undefined otherwise.
    float2  barycentrics;       ///< Barycentric coordinates at the hit, undefined otherwise.

    [mutating] void clear()
    {
        meshInstanceID = kInvalidIndex;
        primitiveIndex = kInvalidIndex;
        barycentrics = float2(0, 0);
    }
};

struct ShadowRay
{
    float4 rayParams;           ///< Shadow ray normalized direction (xyz) and distance (w).
    float3 Lr;                  ///< Unoccluded contribution from the shadow ray (xyz).
    float _pad;
};

enum class PathFlags
{
    terminated      = 0x000001,     ///< Path is terminated. There are no more operations left to do.
    scatterRay      = 0x000002,     ///< Scatter ray is active.
    scatterHit      = 0x000004,     ///< Result of the scatter ray (0 = miss, 1 = hit).
    // Bit 3 reserved

    // Bits 4-13 shadow ray active
    // Bits 14-24 shadow test result
    shadowRay       = 0x000010,     ///< Shadow ray is active.
    shadowResult    = 0x004000,     ///< Result of the shadow test (0 = occluded, 1 = visible).

    anyShadowRays   = 0x003ff0,     ///< Bit mask to test if there are any active shadow rays.
    anyRays         = 0x003ff2,     ///< Bit mask to test if there are any active rays.
    shadowRayBits   = 0xfffff0,
};

/** Working data for the path tracer.

    Note that the shadow ray data is handled separately to avoid having a very
    large live state when there are multiple light samples per vertex.
*/
struct PathData
{
    uint    flags;              ///< Flags indicating the current status. This can be multiple PathFlags flags OR'ed together.
    uint    length;             ///< Path length (0 at origin, 1 at first secondary hit, etc.).

    // Scatter ray
    float3  origin;             ///< Origin of the shadow/scatter rays. Note: This field is only loaded if MIS is enabled.
    float3  dir;                ///< Scatter ray normalized direction.
    float3  thp;                ///< Path throughput.
    float   pdf;                ///< Pdf for generating the scatter ray. Note: This field is only loaded/stored if MIS is enabled.
    float3  normal;             ///< Shading normal at the scatter ray origin. Note: This field is only loaded/stored if MIS is enabled.
    HitInfo hit;                ///< Hit information for the scatter ray. This is populated by the tracing pass. Only valid if the flag 'scatterHit' is set.

    // Common data
    float3  Ldirect;            ///< Accumulated radiance due to direct illumination.
    float3  Lindirect;          ///< Accumulated radiance due to indirect illumination.

    SampleGenerator sg;         ///< Sample generator state. Note: This is only valid when path.length < kMaxBounces.


    // Utility functions

    bool isTerminated() { return (flags & uint(PathFlags::terminated)) != 0; }
    bool hasScatterRay() { return (flags & uint(PathFlags::scatterRay)) != 0; }
    bool hasShadowRay(uint i) { return (flags & (uint(PathFlags::shadowRay) << i)) != 0; }
    bool hasRays() { return (flags & uint(PathFlags::anyRays)) != 0; }

    [mutating] void clearScatterRay() { flags &= ~(uint(PathFlags::scatterRay) | uint(PathFlags::scatterHit)); }
    [mutating] void clearShadowRay(uint i) { flags &= ~((uint(PathFlags::shadowRay) | uint(PathFlags::shadowResult)) << i); }
    [mutating] void clearShadowRays() { flags &= ~(uint(PathFlags::shadowRayBits)); }

    /** Clear all members to their defaults.
    */
    [mutating] void clear()
    {
        flags = 0;
        length = 0;
        origin = float3(0);
        dir = float3(0);
        thp = float3(0);
        pdf = 0;
        normal = float3(0);
        hit.clear();
        Ldirect = float3(0);
        Lindirect = float3(0);
        sg = {};
    }
};
