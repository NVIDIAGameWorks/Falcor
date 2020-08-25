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
__exported import Scene.HitInfo;
__exported import Utils.Sampling.SampleGenerator;
__exported import RenderPasses.Shared.PathTracer.StaticParams;
__exported import RenderPasses.Shared.PathTracer.InteriorList;
__exported import RenderPasses.Shared.PathTracer.RayFootprintData;

struct ShadowRay
{
    float4 rayParams;           ///< Shadow ray normalized direction (xyz) and distance (w).
    float3 Lr;                  ///< Unoccluded contribution from the shadow ray (xyz).
    float _pad;
};

// TODO: Move enum to host/device shared header so that we can add validation.
/** The path flags are stored in `kMaxPathFlagsBits` bits.
    If all ray bits are zero the path is inactive/terminated.
*/
enum class PathFlags
{
    // Bits 0-7 path flags (8 bits)
    scatterRay      = 0x0001,   ///< Scatter ray is active.
    scatterHit      = 0x0002,   ///< Result of the scatter ray (0 = miss, 1 = hit).
    transmission    = 0x0004,   ///< Scatter ray went through a transmission event.
    specular        = 0x0008,   ///< Scatter ray went through a specular event.
    delta           = 0x0010,   ///< Scatter ray went through a delta event.
    insideVolume    = 0x0020,   ///< Path vertex is inside a volume.
    // Bits 6-7 reserved

    // Bits 8-15 shadow ray active (8 bits)
    shadowRay       = 0x0100,   ///< Shadow ray is active (0 = inactive/occluded, 1 = active/visible)

    anyShadowRays   = 0xff00,   ///< Bit mask to test if there are any active shadow rays.
    anyRays         = 0xff01,   ///< Bit mask to test if there are any active rays (scatter and/or shadow).
};

/** Working data for the path tracer.

    Note that the shadow ray data is handled separately to avoid having a very
    large live state when there are multiple light samples per vertex.
*/
struct PathData
{
    uint    flags;              ///< Flags indicating the current status. This can be multiple PathFlags flags OR'ed together.
    uint    length;             ///< Path length (0 at origin, 1 at first secondary hit, etc.).
    uint    nonSpecularBounces; ///< Number of non-specular bounces along the path.

    // Scatter ray
    float3  origin;             ///< Origin of the shadow/scatter rays. Note: This field is only loaded if MIS is enabled.
    float3  transmissionOrigin; ///< Origin of the scatter ray after a transmission event. Note: This is write only.
    float3  dir;                ///< Scatter ray normalized direction.
    float3  thp;                ///< Path throughput.
    float   pdf;                ///< Pdf for generating the scatter ray. Note: This field is only loaded/stored if MIS is enabled.
    float3  normal;             ///< Shading normal at the scatter ray origin. Note: This field is only loaded/stored if MIS is enabled.
    HitInfo hit;                ///< Hit information for the scatter ray. This is populated by the tracing pass. Only valid if the flag 'scatterHit' is set.

    // Common data
    float3  L;                  ///< Accumulated path contribution.

    InteriorList interiorList;  ///< Interior list. Keeping track of a stack of materials with medium properties.

    SampleGenerator sg;         ///< Sample generator state. Note: This is only valid when path.length < kMaxBounces.

    RayFootprint rayFootprint;  ///< Ray footprint for TexLOD. Actual data depends on the footprint mode.

    // Utility functions

    bool isTerminated() { return !hasFlag(PathFlags::anyRays); }
    bool isTransmission() { return hasFlag(PathFlags::transmission); }
    bool isSpecular() { return hasFlag(PathFlags::specular); }
    bool isDelta() { return hasFlag(PathFlags::delta); }
    bool isInsideVolume() { return hasFlag(PathFlags::insideVolume); }

    // Check if the scatter event is samplable by the light sampling technique.
    bool isLightSamplable() { return !(isDelta() || isTransmission()); }

    bool hasRays() { return hasFlag(PathFlags::anyRays); }
    bool hasScatterRay() { return hasFlag(PathFlags::scatterRay); }
    bool hasShadowRay(uint i) { return (flags & (uint(PathFlags::shadowRay) << i)) != 0; }
    bool isShadowRayOccluded(uint i) { return (flags & ((uint)PathFlags::shadowRay << i)) == 0; }

    [mutating] void clearScatterRay() { flags &= ~(uint(PathFlags::scatterRay) | uint(PathFlags::scatterHit)); }
    [mutating] void clearShadowRay(uint i) { flags &= ~(uint(PathFlags::shadowRay) << i); }
    [mutating] void clearShadowRays() { flags &= ~(uint(PathFlags::anyShadowRays)); }

    [mutating] void clearEventFlags() { flags &= ~(uint(PathFlags::transmission) | uint(PathFlags::specular) | uint(PathFlags::delta)); }
    [mutating] void setTransmission(bool value = true) { setFlag(PathFlags::transmission, value); }
    [mutating] void setSpecular(bool value = true) { setFlag(PathFlags::specular, value); }
    [mutating] void setDelta(bool value = true) { setFlag(PathFlags::delta, value); }
    [mutating] void setInsideVolume(bool value = true) { setFlag(PathFlags::insideVolume, value); }

    bool hasFlag(PathFlags flag) { return (flags & uint(flag)) != 0; }
    [mutating] void setFlag(PathFlags flag, bool value = true)
    {
        if (value) flags |= uint(flag);
        else flags &= ~uint(flag);
    }
};
