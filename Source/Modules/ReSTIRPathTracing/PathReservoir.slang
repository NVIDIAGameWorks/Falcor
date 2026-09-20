/***************************************************************************
 # Copyright (c) 2015-26, NVIDIA CORPORATION. All rights reserved.
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
import Utils.Sampling.SampleGenerator;
import Scene.HitInfo;
import Utils.Debug.PixelDebug;

import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;
#include "Utils/Math/MathConstants.slangh"
#include "Scene/SceneDefines.slangh"

/** Packed path flags carried by every reservoir.
    Bit layout (LSB to MSB), all named below as `kXxxShift` / `kXxxMask`:

        bits  0..7   : path length                   (8 bits, max 255)
        bits  8..15  : rc-vertex length              (8 bits, max 255)
        bit  16      : last-vertex was NEE
        bit  17      : "very direct light" flag (primary-hit emissive contribution)
        bits 18..19  : light type                    (2 bits)
        bits 20..25  : "before rc-vertex" sub-bits   (6 bits, see below)
        bits 26..30  : "after  rc-vertex" sub-bits   (5 bits, see below; no BSDF-indexing bit)

    The 6-bit "before-rc" packed payload (`kBouncePackedBeforeRcMask`) is laid out as:
        bit 0 (= 20) : isDelta
        bit 1 (= 21) : isTransmission
        bits 2..4    : BSDFComponent (3 bits)
        bit 5  (= 25): supportBSDFIndexing
    The 5-bit "after-rc" packed payload (`kBouncePackedAfterRcMask`) is laid out the same way
    *except* it does not carry the supportBSDFIndexing bit (so 5 bits, not 6).
*/
struct ReSTIRPathFlags
{
    // ---- Path-level fields ----
    static const uint kPathLengthMask              = 0xFF;
    static const uint kPathLengthShift             = 0;
    static const uint kRcVertexLengthMask          = 0xFF;
    static const uint kRcVertexLengthShift         = 8;
    static const uint kLastVertexNEEShift          = 16;
    static const uint kIsVeryDirectLightShift      = 17;
    static const uint kLightTypeMask               = 0x3;
    static const uint kLightTypeShift              = 18;

    // ---- Before-rc-vertex sub-bits (bits 20..25, 6 bits) ----
    static const uint kBouncePackedBeforeRcMask    = 0x3f;
    static const uint kBouncePackedBeforeRcShift   = 20;
    static const uint kIsDeltaBeforeRcShift        = 20;
    static const uint kIsTransmissionBeforeRcShift = 21;
    static const uint kBSDFComponentBeforeRcShift  = 22;
    static const uint kSupportBSDFIndexingShift    = 25;

    // ---- After-rc-vertex sub-bits (bits 26..30, 5 bits, no BSDF-indexing) ----
    static const uint kBouncePackedAfterRcMask     = 0x1f;
    static const uint kBouncePackedAfterRcShift    = 26;
    static const uint kIsDeltaAfterRcShift         = 26;
    static const uint kIsTransmissionAfterRcShift  = 27;
    static const uint kBSDFComponentAfterRcShift   = 28;

    static const uint kBSDFComponentMask           = 0x7; // 3 bits, used by both before/after.

    uint flags = 0;


    [mutating] void tryInsertBounceTypesBeforeRcVertex(uint data, bool doInsert)
    {
        flags &= doInsert ? ~(kBouncePackedBeforeRcMask << kBouncePackedBeforeRcShift) : 0xffffffff;
        flags |= doInsert ? (data << kBouncePackedBeforeRcShift) : 0;
    }

    [mutating] void tryInsertBounceTypesAfterRcVertex(uint data, bool doInsert)
    {
        flags &= doInsert ? ~(kBouncePackedAfterRcMask << kBouncePackedAfterRcShift) : 0xffffffff;
        flags |= doInsert ? (data << kBouncePackedAfterRcShift) : 0;
    }

    [mutating]
    void insertIsDeltaEvent(bool isDeltaEvent, bool beforeRcVertex)
    {
        const uint shift = beforeRcVertex ? kIsDeltaBeforeRcShift : kIsDeltaAfterRcShift;
        flags &= ~(1 << shift);
        flags |= uint(isDeltaEvent) << shift;
    }

    [mutating]
    void insertIsTransmissionEvent(bool isTransmissionEvent, bool beforeRcVertex)
    {
        const uint shift = beforeRcVertex ? kIsTransmissionBeforeRcShift : kIsTransmissionAfterRcShift;
        flags &= ~(1 << shift);
        flags |= uint(isTransmissionEvent) << shift;
    }

    // component uses 3 bits
    [mutating]
    void insertBSDFComponentType(uint component, bool beforeRcVertex)
    {
        const uint shift = beforeRcVertex ? kBSDFComponentBeforeRcShift : kBSDFComponentAfterRcShift;
        flags &= ~(kBSDFComponentMask << shift);
        flags |= component << shift;
    }

    [mutating]
    void insertSupportBSDFIndexingBeforeRcVertex(bool enabled)
    {
        flags &= ~(1 << kSupportBSDFIndexingShift);
        flags |= uint(enabled) << kSupportBSDFIndexingShift;
    }

    bool decodeIsDeltaEvent(bool beforeRcVertex)
    {
        return (flags >> (beforeRcVertex ? kIsDeltaBeforeRcShift : kIsDeltaAfterRcShift)) & 1;
    }

    bool decodeIsTransmissionEvent(bool beforeRcVertex)
    {
        return (flags >> (beforeRcVertex ? kIsTransmissionBeforeRcShift : kIsTransmissionAfterRcShift)) & 1;
    }

    uint decodeBSDFComponent(bool beforeRcVertex)
    {
        return (flags >> (beforeRcVertex ? kBSDFComponentBeforeRcShift : kBSDFComponentAfterRcShift)) & kBSDFComponentMask;
    }

    bool decodeSupportBSDFIndexingBeforeRcVertex()
    {
        return (flags >> kSupportBSDFIndexingShift) & 1;
    }

    // maximum length: 255
    [mutating]
    void insertPathLength(int pathLength)
    {
        flags &= ~kPathLengthMask;
        flags |= pathLength & kPathLengthMask;
    }

    // maximum length: 255
    [mutating]
    void insertRcVertexLength(int rcVertexLength)
    {
        flags &= ~(kRcVertexLengthMask << kRcVertexLengthShift);
        flags |= (rcVertexLength & kRcVertexLengthMask) << kRcVertexLengthShift;
    }

    [mutating]
    void resetRcVertexLength()
    {
        flags |= kRcVertexLengthMask << kRcVertexLengthShift;
    }

    int pathLength()
    {
        return flags & kPathLengthMask;
    }

    int rcVertexLength()
    {
        return (flags >> kRcVertexLengthShift) & kRcVertexLengthMask;
    }

    [mutating]
    void insertLastVertexNEE(bool isNEE)
    {
        flags &= ~(1 << kLastVertexNEEShift);
        flags |= (int(isNEE) & 1) << kLastVertexNEEShift;
    }

    bool lastVertexNEE()
    {
        return (flags >> kLastVertexNEEShift) & 1;
    }

    [mutating]
    void insertIsVeryDirectLight(bool isVeryDirectLight)
    {
        flags &= ~(1 << kIsVeryDirectLightShift);
        flags |= (int(isVeryDirectLight) & 1) << kIsVeryDirectLightShift;
    }

    bool isVeryDirectLight()
    {
        return (flags >> kIsVeryDirectLightShift) & 1;
    }

    [mutating]
    void insertLightType(uint lightType)
    {
        flags &= ~(kLightTypeMask << kLightTypeShift);
        flags |= ((int(lightType) & kLightTypeMask) << kLightTypeShift);
    }

    uint lightType()
    {
        return (flags >> kLightTypeShift) & kLightTypeMask;
    }
};

struct LastVertexState
{
    static const uint kIsDeltaShift              = 0;
    static const uint kIsTransmissionShift       = 1;
    static const uint kBSDFComponentShift        = 2;
    static const uint kBSDFComponentMask         = 0x7; // 3 bits
    static const uint kSupportBSDFIndexingShift  = 5;
    static const uint kIsRoughShift              = 6;
    static const uint kIsFarFromPrevShift        = 7;

    uint data = 0;

    [mutating] void Init( bool isLastVertexDelta, bool isLastVertexTransmission, uint lastBSDFComponent,
                    bool lastVertexSupportBSDFIndexing, bool isLastVertexRough, bool isCurrentVertexFarFromPrev )
    {
        data = 0;
        data |= (uint)isLastVertexDelta                   << kIsDeltaShift;
        data |= (uint)isLastVertexTransmission            << kIsTransmissionShift;
        data |= (uint)lastBSDFComponent                   << kBSDFComponentShift;
        data |= (uint)lastVertexSupportBSDFIndexing       << kSupportBSDFIndexingShift;
        data |= (uint)isLastVertexRough                   << kIsRoughShift;
        data |= (uint)isCurrentVertexFarFromPrev          << kIsFarFromPrevShift;
    }

    bool isLastVertexDelta()              { return (data >> kIsDeltaShift) & 1; }
    bool isLastVertexTransmission()       { return (data >> kIsTransmissionShift) & 1; }
    uint lastBSDFComponent()              { return (data >> kBSDFComponentShift) & kBSDFComponentMask; }
    bool lastVertexSupportBSDFIndexing()  { return (data >> kSupportBSDFIndexingShift) & 1; }
    bool isLastVertexRough()              { return (data >> kIsRoughShift) & 1; }
    bool isCurrentVertexFarFromPrev()     { return (data >> kIsFarFromPrevShift) & 1; }
}


// fall back to HitInfo if compression is not possible (ReSTIR PT doesn't support volume hit yet)
#if !USE_RESERVOIR_COMPRESSION || HIT_INFO_USE_COMPRESSION || 3 + HIT_INFO_INSTANCE_ID_BITS + HIT_INFO_PRIMITIVE_INDEX_BITS > 32 || SCENE_GEOMETRY_TYPES != (1 << GEOMETRY_TYPE_TRIANGLE_MESH)
typedef HitInfo HitInfoPacked;

HitInfo unpackReservoirHitInfo(HitInfo input)
{
    return input;
}

HitInfo packReservoirHitInfo(HitInfo input)
{
    return input;
}
#else
struct HitInfoPacked
{
    uint2 data;

    // assume traingle hit
    HitInfo unpack()
    {
        float2 barycentrics = unpackUnorm2x16(data[1]);
        HitInfo hit;
        hit.data = { data[0], 0, asuint(barycentrics[0]), asuint(barycentrics[1]) };
        return hit;
    }

    __init(const HitInfo hitInfo)
    {
        data[0] = hitInfo.data[0];
        data[1] = packUnorm2x16_unsafe(float2(asfloat(hitInfo.data[2]), asfloat(hitInfo.data[3])));
    }
}

HitInfo unpackReservoirHitInfo(HitInfoPacked input)
{
    return input.unpack();
}

HitInfoPacked packReservoirHitInfo(HitInfo input)
{
    return HitInfoPacked(input);
}
#endif

// We use PSS-solid angle mixed parameterization in the implementation
// Original PSS integrand in the GRIS paper (f_i and p_i are BSDF and the sampling PDF at each bounce) :
// (assuming the reconnection vertex is x_k, and omitting the MIS weight)
// f_1 / p_1 * f_2 / p_2 * ... * f_{ k - 1 } / p_{ k - 1 } * f_k / p_k * f_{ k + 1 } / p_{ k + 1 } * ...
// For reconnection , we will need to re-evaluate f_{ k - 1 } / p_{ k - 1 } and f_k / p_k , the Jacobian contains the new p_{ k - 1 } and p_k , besides geometry terms.
// The new mixed integrand : (switching to solid angle parameterization locally around the reconnection vertex)
// f_1 / p_1* f_2 / p_2*...* f_{ k - 1 }  * f_k   * (f_{ k + 1 } / p_{ k + 1 } * ...). The bracketed terms are cached as reconnectionIncidentRadiance.
// For reconnection, we only need to re-evaluate f_{ k - 1 } and f_k.The Jacobian only contains the geometry terms.
// This simplifies implementation, speeds up computation, and gives mathematically equivalent results.

struct PathReservoir
{
    float M;
    float weight;
    float3 integrand; // the integrand value f/p in PSS
    ReSTIRPathFlags pathFlags;
    uint initRandomSeed;
    uint rcNextVertexBounceSeed;
    float lightPdf; // NEE pdf of the sampled light in the path
#if USE_AREA_RESERVOIRS
    float4 filmLensSample;
#endif
#if !USE_RESERVOIR_COMPRESSION
    HitInfo reconnectionHit;
#else
    HitInfoPacked reconnectionHit;
#endif
    float reconnectionJacobian;
    float3 reconnectionIncidentRadiance;
#if !USE_RESERVOIR_COMPRESSION
    float3 reconnectionWi;   
#else
    uint packedRcWi;    

    property float3 reconnectionWi
    {
        get { return decodeNormal2x16(packedRcWi); }
        set { packedRcWi = encodeNormal2x16(newValue); }
    }
#endif

    property float2 filmSample
    {
#if USE_AREA_RESERVOIRS
        get { return filmLensSample.xy; }
#else
        get { return float2(0.f); }
#endif
    }

    property float2 lensSample
    {
#if USE_AREA_RESERVOIRS
        get { return filmLensSample.zw; }
#else
        get { return float2(0.f); }
#endif
    }

    [mutating]
    void setFilmSample(float2 xy)
    {
#if USE_AREA_RESERVOIRS
        filmLensSample.xy = xy;
#endif
    }

    [mutating]
    void setLensSample(float2 zw)
    {
#if USE_AREA_RESERVOIRS
        filmLensSample.zw = zw;
#endif
    }

    [mutating]
    void init()
    {
        this.M = 0;
        this.weight = 0.;
#if USE_AREA_RESERVOIRS
        this.filmLensSample = 0.f;
#endif
        this.reconnectionHit = packReservoirHitInfo(HitInfo());
    }

    [mutating]
    void init(const PathRISState state)
    {
        this.M = 1;
        this.weight = state.weight;
        this.integrand = state.integrand;
        this.lightPdf = state.lightPdf;
        this.rcNextVertexBounceSeed = state.rcNextVertexBounceSeed;
#if USE_AREA_RESERVOIRS
        this.filmLensSample = 0.f;
#endif
        this.reconnectionHit = packReservoirHitInfo(state.reconnectionHit);
        this.pathFlags = state.pathFlags;
        this.reconnectionJacobian = state.reconnectionJacobian;
        this.reconnectionIncidentRadiance = state.reconnectionIncidentRadiance;
        this.reconnectionWi = state.reconnectionWi;
    }

    static PathReservoir createFromRisState( const PathRISState state, uint initRandomSeed, float4 filmLensSample)
    {
        PathReservoir r;
        r.M = 1;
        r.weight = state.weight;
        r.integrand = state.integrand;
        r.lightPdf = state.lightPdf;
        r.rcNextVertexBounceSeed = state.rcNextVertexBounceSeed;
#if USE_AREA_RESERVOIRS
        r.filmLensSample = filmLensSample;
#endif
        r.initRandomSeed = initRandomSeed;
        r.reconnectionHit = packReservoirHitInfo(state.reconnectionHit);
        r.pathFlags = state.pathFlags;
        r.reconnectionJacobian = state.reconnectionJacobian;
        r.reconnectionIncidentRadiance = state.reconnectionIncidentRadiance;
        r.reconnectionWi = state.reconnectionWi;
        return r;
    }

    [mutating]
    bool add( const PathRISState state, uint initRandomSeed, float4 filmLensSample, inout SampleGenerator sg)
    {
        float w = state.weight;
        w = (isnan(w) || isinf(w)) ? 0.0f : w;
        this.M += 1;
        this.weight += w;

        bool sampleSelected = (sampleNext1D(sg) * this.weight < w);

        this.pathFlags.flags = sampleSelected ? state.pathFlags.flags : this.pathFlags.flags;
        this.rcNextVertexBounceSeed = sampleSelected ? state.rcNextVertexBounceSeed : this.rcNextVertexBounceSeed;
        this.initRandomSeed = sampleSelected ? initRandomSeed : this.initRandomSeed;
#if USE_AREA_RESERVOIRS
        this.filmLensSample = sampleSelected ? filmLensSample : this.filmLensSample;
#endif
        this.lightPdf = sampleSelected ? state.lightPdf : this.lightPdf;

        if (sampleSelected)
        {
            this.reconnectionHit = packReservoirHitInfo(state.reconnectionHit);
        }
        this.integrand = sampleSelected ? state.integrand : this.integrand;
        this.reconnectionJacobian = sampleSelected ? state.reconnectionJacobian : this.reconnectionJacobian;
        this.reconnectionIncidentRadiance = sampleSelected ? state.reconnectionIncidentRadiance : this.reconnectionIncidentRadiance;
        this.reconnectionWi = sampleSelected ? state.reconnectionWi : this.reconnectionWi;

        return sampleSelected;
    }

    [mutating]
    bool merge(float3 integrand, float in_Jacobian, PathReservoir inReservoir, inout SampleGenerator sg, float misWeight)
    {
        float w = PathRISState::toScalar(integrand) * in_Jacobian * inReservoir.weight * misWeight;
        w = (isnan(w) || isinf(w)) ? 0.0f : w;

        this.M += inReservoir.M;
        this.weight += w;

        bool sampleSelected = (sampleNext1D(sg) * this.weight < w);

        this.pathFlags.flags = sampleSelected ? inReservoir.pathFlags.flags : this.pathFlags.flags;
        this.rcNextVertexBounceSeed    = sampleSelected ? inReservoir.rcNextVertexBounceSeed : this.rcNextVertexBounceSeed;
        this.initRandomSeed = sampleSelected ? inReservoir.initRandomSeed : this.initRandomSeed;
        this.lightPdf        = sampleSelected ? inReservoir.lightPdf : this.lightPdf;
#if USE_AREA_RESERVOIRS
        this.filmLensSample  = sampleSelected ? inReservoir.filmLensSample : this.filmLensSample;
#endif
        this.reconnectionHit.data      = sampleSelected ? inReservoir.reconnectionHit.data : this.reconnectionHit.data;
        this.integrand       = sampleSelected ? integrand : this.integrand;
        this.reconnectionJacobian = sampleSelected ? inReservoir.reconnectionJacobian : this.reconnectionJacobian;
        this.reconnectionIncidentRadiance = sampleSelected ? inReservoir.reconnectionIncidentRadiance : this.reconnectionIncidentRadiance;
#if !USE_RESERVOIR_COMPRESSION
        this.reconnectionWi            = sampleSelected ? inReservoir.reconnectionWi : this.reconnectionWi;
#else
        this.packedRcWi      = sampleSelected ? inReservoir.packedRcWi : this.packedRcWi;
#endif

        return sampleSelected;
    }

    [mutating]
    void mergeDirectOverwrite(float3 integrand, PathReservoir inReservoir)
    {
        this.pathFlags.flags              = inReservoir.pathFlags.flags;
        this.rcNextVertexBounceSeed       = inReservoir.rcNextVertexBounceSeed;
        this.initRandomSeed               = inReservoir.initRandomSeed;
        this.lightPdf                     = inReservoir.lightPdf;
#if USE_AREA_RESERVOIRS
        this.filmLensSample               = inReservoir.filmLensSample;
#endif
        this.reconnectionHit.data         = inReservoir.reconnectionHit.data;
        this.integrand                    = integrand;
        this.reconnectionJacobian         = inReservoir.reconnectionJacobian;
        this.reconnectionIncidentRadiance = inReservoir.reconnectionIncidentRadiance;
#if !USE_RESERVOIR_COMPRESSION
        this.reconnectionWi               = inReservoir.reconnectionWi;
#else
        this.packedRcWi                   = inReservoir.packedRcWi;
#endif
    }

    [mutating]
    bool mergeWithoutWriting(float pHatWithJacobian, PathReservoir inReservoir, inout SampleGenerator sg, float misWeight)
    {
        float w = pHatWithJacobian * inReservoir.weight * misWeight;
        w = (isnan(w) || isinf(w)) ? 0.0f : w;
        this.M += inReservoir.M;
        this.weight += w;
        bool sampleSelected = (sampleNext1D(sg) * this.weight < w);
#if USE_AREA_RESERVOIRS
        this.filmLensSample = sampleSelected ? inReservoir.filmLensSample : this.filmLensSample;
#endif
        return sampleSelected;
    }

    [mutating]
    void finalizeReservoirWeight()
    {
        float p_hat = PathRISState::toScalar(integrand);
        this.weight = (p_hat == 0.f) ? 0.f : (this.weight / p_hat);
    }
}


// stores information about reconnection vertex (that can be reused by later vertices)
struct PathReconnectionInfo
{
    ReSTIRPathFlags pathFlags = {};  
    uint rcNextVertexBounceSeed = 0;
    uint initRandomSeed = 0;
    float3 rcThroughput = 0; // path throughput after rcVertex
    HitInfo reconnectionHit = {}; 
    float reconnectionJacobian = 0;
    float3 reconnectionWi = 0; 
    float footprintFactor = 0;
}

// this is basically a reservoir without M and initRandomSeed
struct PathRISState
{
    float weight = 0;
    float3 integrand = 0; 
    ReSTIRPathFlags pathFlags = {};
    float lightPdf = 0; 
    uint rcNextVertexBounceSeed = 0;
    HitInfo reconnectionHit = {};
    float reconnectionJacobian = 0;
    float3 reconnectionIncidentRadiance = 0; 
    float3 reconnectionWi = 0; 

    [mutating] void init()
    {
        weight = 0.f;
    }

    static float gaussrand(float mean, float stddev, inout SampleGenerator sg)
    {
        // Sampling from the normal distribution (Box-Muller method)
        const float2 rand = sampleNext2D(sg);
        float Z = sqrt(-2.0 * log(rand.x)) * cos(2.0 * M_PI * rand.y);
        float X = Z * stddev * mean + mean; // want the standard deviation to scale with mean to provide a soft threshold
        return X;
    }

    [mutating]
    bool addNEECandidate(float u, PathReconnectionInfo pathRcInfo, bool isRcVertex, uint pathLength,
                      float3 pathContribution, float3 wi, float3 Le, float3 Lr, float sourcePdf, float lightPdf, uint lightType)
    {
        bool selected = addPSSSample(u, pathContribution);
        uint rcVertexLength = pathRcInfo.pathFlags.rcVertexLength();
        rcVertexLength = isRcVertex ? pathLength : rcVertexLength;
        this.pathFlags.flags = selected ? pathRcInfo.pathFlags.flags : this.pathFlags.flags;
        this.pathFlags.insertPathLength(selected ? pathLength : this.pathFlags.pathLength());
        this.pathFlags.insertRcVertexLength(selected ? rcVertexLength : this.pathFlags.rcVertexLength());
        this.pathFlags.insertLastVertexNEE(selected || this.pathFlags.lastVertexNEE());
        this.reconnectionIncidentRadiance = selected ? (isRcVertex ? Le : pathRcInfo.rcThroughput * Lr) : this.reconnectionIncidentRadiance;
        this.reconnectionWi = selected ? (isRcVertex ? wi : pathRcInfo.reconnectionWi) : this.reconnectionWi;
        this.reconnectionHit.data = selected ? pathRcInfo.reconnectionHit.data : this.reconnectionHit.data;
        this.reconnectionJacobian = selected ? pathRcInfo.reconnectionJacobian : this.reconnectionJacobian;
        this.rcNextVertexBounceSeed = selected ? pathRcInfo.rcNextVertexBounceSeed : this.rcNextVertexBounceSeed;
        // integrand in our mixed PSS-solid angle measure
        this.integrand = selected ? pathContribution * sourcePdf : this.integrand;
        this.lightPdf = selected ? lightPdf : this.lightPdf;
        this.pathFlags.insertLightType(selected ? lightType : this.pathFlags.lightType());

        return selected;
    }

    [mutating]
    bool addBSDFHitCandidate(float u, PathReconnectionInfo pathRcInfo, uint pathLength,
                       float3 pathContribution, float3 wi, float3 Le, float3 Lr, float sourcePdf, float lightPdf, uint lightType)
    {

        bool selected = addPSSSample(u, pathContribution);
        uint rcVertexLength = pathRcInfo.pathFlags.rcVertexLength();
        bool isRcVertex = pathLength == rcVertexLength;

        this.pathFlags.flags = selected ? pathRcInfo.pathFlags.flags : this.pathFlags.flags;
        this.pathFlags.insertPathLength(selected ? pathLength : this.pathFlags.pathLength());
        this.pathFlags.insertLastVertexNEE(!selected && this.pathFlags.lastVertexNEE());
        this.reconnectionIncidentRadiance = selected ? (isRcVertex ? Le : (pathRcInfo.rcThroughput * Lr)) : this.reconnectionIncidentRadiance;

        this.reconnectionWi = selected ? pathRcInfo.reconnectionWi : this.reconnectionWi;
        this.reconnectionHit.data = selected ? pathRcInfo.reconnectionHit.data : this.reconnectionHit.data;
        this.reconnectionJacobian = selected ? pathRcInfo.reconnectionJacobian : this.reconnectionJacobian;
        this.rcNextVertexBounceSeed = selected ? pathRcInfo.rcNextVertexBounceSeed : this.rcNextVertexBounceSeed;

        // integrand in our mixed PSS-solid angle measure
        this.integrand = selected ? pathContribution * sourcePdf : this.integrand;
        this.lightPdf = selected ? lightPdf : this.lightPdf;
        this.pathFlags.insertLightType(selected ? lightType : this.pathFlags.lightType());

        return selected;
    }

    [mutating]
    bool addPrimaryVisibleEmitterCandidate(float u, float3 pathContribution, float sourcePdf)
    {
        bool selected = addPSSSample(u, pathContribution);
        this.pathFlags.flags = selected ? 0 : this.pathFlags.flags;
        this.pathFlags.insertIsVeryDirectLight(selected ? true : this.pathFlags.isVeryDirectLight());
        // integrand in our mixed PSS-solid angle measure
        this.integrand = selected ? pathContribution * sourcePdf : this.integrand;
        return selected;
    }


    [mutating]
    void markAsRcVertex(uint pathLength, HitInfo hit, LastVertexState lastVertexState,
                            float lightPdf, uint lightType, float3 Le, float3 wi, float reconnectionJacobian)
    {
        this.pathFlags.insertRcVertexLength(pathLength);

        this.reconnectionHit = hit;
        this.pathFlags.insertIsDeltaEvent(lastVertexState.isLastVertexDelta(), true);
        this.pathFlags.insertIsTransmissionEvent(lastVertexState.isLastVertexTransmission(), true);
        this.pathFlags.insertBSDFComponentType(lastVertexState.lastBSDFComponent(), true);
        this.pathFlags.insertSupportBSDFIndexingBeforeRcVertex(lastVertexState.lastVertexSupportBSDFIndexing());
        this.lightPdf = lightPdf;

        this.reconnectionJacobian = reconnectionJacobian;

        this.pathFlags.insertLightType(lightType);
        this.reconnectionIncidentRadiance = Le;
        this.reconnectionWi = wi;
    }

    [mutating]
    bool addPSSSample(float u, float3 pathContribution)
    {
        float w = toScalar(pathContribution);

        w = isnan(w) ? 0.0f : w;
        this.weight += w;

        return (u * this.weight < w);
    }

    static float toScalar(float3 color)
    {
        return dot(color, float3(0.299, 0.587, 0.114)); // luminance
    }
}

struct ReconnectionData
{
    HitInfo rcPrevHit; 
    float3 pathThroughput; 
#if !USE_RESERVOIR_COMPRESSION
    float3 rcPrevWo;     
#else
    uint packedRcPrevWo;   
    property float3 rcPrevWo
    {
        get { return decodeNormal2x16(packedRcPrevWo); }
        set { packedRcPrevWo = encodeNormal2x16(newValue); }
    }

#endif
    float rcFootprintFactor; // for checking invertibility for footprint thresholds

    __init(HitInfo rcPrevHit, float3 rcPrevWo, float3 pathThroughput, float rcFootprintFactor)
    {
        this.rcPrevHit = rcPrevHit;
        this.rcPrevWo = rcPrevWo;
        this.pathThroughput = pathThroughput;
        this.rcFootprintFactor = rcFootprintFactor;
    }

    [mutating]
    void init()
    {
        rcPrevHit.data = {};
        rcPrevHit.data[0] = uint(HitType::Triangle) << HitInfo::kTypeOffset;
        rcPrevWo = float3(0.f);
        pathThroughput = float3(1.f);
        rcFootprintFactor = float(0.f);
    }
}

struct NeighborValidMask
{
    uint data;
    float cSum;
    bool isValid(int i)
    {
        return data >> i & 1;
    }

    [mutating]
    void setValid(int i, bool val)
    {
        data &= ~(1 << i);
        data |= int(val) << i;
    }

    [mutating]
    void setCSum(float val)
    {
        cSum = val;
    }
}

struct PrecomputedShiftData
{
    float3 pathThroughput;
    float Jacobian;
    float reconnectionJacobian;
}
