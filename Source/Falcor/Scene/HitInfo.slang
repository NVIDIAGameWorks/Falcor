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
#include "Scene/SceneDefines.slangh"
import Utils.Math.FormatConversion;
__exported import Scene.HitInfoType;
__exported import Scene.SceneTypes;
__exported import Scene.SDFs.SDFGridHitData;

/** Ray hit information.

    'HitInfo' is a polymorphic container for storing ray hit information.
    'PackedHitInfo' holds the data in 'HitInfo'.
    The upper bits of the first uint are used to store the type of the hit.
    A zero initialized value represent hit type 'None'.

    HitInfo stores data for one of the following type specific structs:
    TriangleHit, DisplacedTriangleHit, CurveHit, CurveOTSHit, SDFGridHit, VolumeHit

    TriangleHit, DisplacedTriangleHit, CurveHit, and CurveOTSHit all store the hit point in terms of
    an instance ID and a primitive index, together with barycentrics.

    DisplacedTriangleHit additionally stores the displacement offset.

    To use HitInfo, the host needs to set the following defines:

    - HIT_INFO_DEFINES              Marks that the rest of the defines are available.
    - HIT_INFO_USE_COMPRESSION      Use compressed format (64 bits instead of 128 bits).
    - HIT_INFO_TYPE_BITS            Bits needed to encode the hit type.
    - HIT_INFO_INSTANCE_ID_BITS     Bits needed to encode the instance ID of the hit.
    - HIT_INFO_PRIMITIVE_INDEX_BITS Bits needed to encode the primitive index of the hit.

    If a bit size define is zero, no bits are needed (the field has only one possible value = 0).
*/

#if HIT_INFO_USE_COMPRESSION
typedef uint2 PackedHitInfo;
#else
typedef uint4 PackedHitInfo;
#endif

/** Geometry hit information (base class).
*/
struct GeometryHit
{
    GeometryInstanceID instanceID;
    uint primitiveIndex;
    float2 barycentrics;

    /** Return the barycentric weights.
    */
    float3 getBarycentricWeights()
    {
        return float3(1.f - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);
    }
};

/** Triangle hit information.

    Encoding (without compression):
    | header  | barycentrics.x | barycentrics.y |
    | 64 bits | 32 bit float   | 32 bit float   |

    Encoding (with compression):
    | header     | barycentrics.x | barycentrics.y |
    | 32 bits    | 16 bit unorm   | 16 bit unorm   |
*/
struct TriangleHit : GeometryHit
{
    __init(const PackedHitInfo packed)
    {
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
        HitInfo::unpackHeader(packed, instanceID, primitiveIndex);
#if HIT_INFO_USE_COMPRESSION
        barycentrics = unpackUnorm2x16(packed[1]);
#else
        barycentrics.x = asfloat(packed[2]);
        barycentrics.y = asfloat(packed[3]);
#endif
#else
        this = {};
#endif
    }

    PackedHitInfo pack()
    {
        PackedHitInfo packed = {};
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
        HitInfo::packHeader(packed, HitType::Triangle, instanceID, primitiveIndex);
#if HIT_INFO_USE_COMPRESSION
        packed[1] = packUnorm2x16_unsafe(barycentrics);
#else
        packed[2] = asuint(barycentrics.x);
        packed[3] = asuint(barycentrics.y);
#endif
#endif
        return packed;
    }
};

/** Displaced triangle hit information.

    Encoding:
    | header  | barycentrics.x | barycentrics.y | displacement |
    | 64 bits | 24 bit unorm   | 24 bit unorm   | 16 bit float |
*/
struct DisplacedTriangleHit : GeometryHit
{
    float displacement;

    __init(const PackedHitInfo packed)
    {
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_DISPLACED_TRIANGLE_MESH)
        HitInfo::unpackHeader(packed, instanceID, primitiveIndex);
        const uint ux = (packed[2] >> 8);
        const uint uy = ((packed[2] & 0xff) << 16) | (packed[3] >> 16);
        barycentrics = float2(ux, uy) * (1.f / 16777215);
        displacement = f16tof32(packed[3]);
#else
        this = {};
#endif
    }

    PackedHitInfo pack()
    {
        PackedHitInfo packed = {};
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_DISPLACED_TRIANGLE_MESH)
        HitInfo::packHeader(packed, HitType::DisplacedTriangle, instanceID, primitiveIndex);
        const uint2 u = trunc(barycentrics * 16777215.f + 0.5f);
        packed[2] = (u.x << 8) | (u.y >> 16);
        packed[3] = (u.y << 16) | f32tof16(displacement);
#endif
        return packed;
    }
};

/** Curve hit information.

    Encoding:
    | header  | barycentrics.x | barycentrics.y |
    | 64 bits | 32 bit float   | 32 bit float   |
*/
struct CurveHit : GeometryHit
{
    __init(const PackedHitInfo packed)
    {
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_CURVE)
        HitInfo::unpackHeader(packed, instanceID, primitiveIndex);
        barycentrics.x = asfloat(packed[2]);
        barycentrics.y = asfloat(packed[3]);
#else
        this = {};
#endif
    }

    PackedHitInfo pack()
    {
        PackedHitInfo packed = {};
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_CURVE)
        HitInfo::packHeader(packed, HitType::Curve, instanceID, primitiveIndex);
        packed[2] = asuint(barycentrics.x);
        packed[3] = asuint(barycentrics.y);
#endif
        return packed;
    }
};


/** SDF grid hit information.

    Encoding:
    | header  | extra  | extra  |
    | 64 bits | 32 bit | 32 bit |
*/
struct SDFGridHit
{
    GeometryInstanceID instanceID;
    SDFGridHitData hitData;

    __init(const PackedHitInfo packed)
    {
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_SDF_GRID)
        uint primitiveData;
        HitInfo::unpackHeader(packed, instanceID, primitiveData);


#if SCENE_SDF_GRID_IMPLEMENTATION == SCENE_SDF_GRID_IMPLEMENTATION_NDSDF
        hitData.lod = primitiveData;
        hitData.hitT = asfloat(packed[2]);
#elif SCENE_SDF_GRID_IMPLEMENTATION == SCENE_SDF_GRID_IMPLEMENTATION_SVS || SCENE_SDF_GRID_IMPLEMENTATION == SCENE_SDF_GRID_IMPLEMENTATION_SBS
        hitData.primitiveID = primitiveData;
        hitData.hitT = asfloat(packed[2]);
#elif SCENE_SDF_GRID_IMPLEMENTATION == SCENE_SDF_GRID_IMPLEMENTATION_SVO
        hitData.svoIndex = primitiveData;
        hitData.hitT = asfloat(packed[2]);
#endif


#else
        this = {};
#endif // SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_SDF_GRID)
    }

    PackedHitInfo pack()
    {
        PackedHitInfo packed = {};

#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_SDF_GRID)
        uint primitiveData = {};

#if SCENE_SDF_GRID_IMPLEMENTATION == SCENE_SDF_GRID_IMPLEMENTATION_NDSDF
        primitiveData = hitData.lod;
        packed[2] = asuint(hitData.hitT);
#elif SCENE_SDF_GRID_IMPLEMENTATION == SCENE_SDF_GRID_IMPLEMENTATION_SVS || SCENE_SDF_GRID_IMPLEMENTATION == SCENE_SDF_GRID_IMPLEMENTATION_SBS
        primitiveData = hitData.primitiveID;
        packed[2] = asuint(hitData.hitT);
#elif SCENE_SDF_GRID_IMPLEMENTATION == SCENE_SDF_GRID_IMPLEMENTATION_SVO
        primitiveData = hitData.svoIndex;
        packed[2] = asuint(hitData.hitT);
#endif


        HitInfo::packHeader(packed, HitType::SDFGrid, instanceID, primitiveData);
#endif // SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_SDF_GRID)

        return packed;
    }
};

/** Volume hit information.

    Encoding (without compression):
    | header  | t            | g            |
    | 64 bits | 32 bit float | 32 bit float |

    Encoding (with compression):
    | header  | g            | t            |
    | 16 bits | 16 bit float | 32 bit float |
*/
struct VolumeHit
{
    float t;
    float g;

    __init(const PackedHitInfo packed)
    {
#if HIT_INFO_USE_COMPRESSION
        t = asfloat(packed[1]);
        g = f16tof32(packed[0] & 0xffff);
#else
        t = asfloat(packed[1]);
        g = asfloat(packed[2]);
#endif
    }

    PackedHitInfo pack()
    {
        PackedHitInfo packed = {};
        HitInfo::packHeader(packed, HitType::Volume);
#if HIT_INFO_USE_COMPRESSION
        packed[1] = asuint(t);
        packed[0] |= f32tof16(g) & 0xffff;
#else
        packed[1] = asuint(t);
        packed[2] = asuint(g);
#endif
        return packed;
    }
};

/** Polymorphic hit information type.
*/
struct HitInfo
{
#ifdef HIT_INFO_DEFINES
    static const uint kTypeBits = HIT_INFO_TYPE_BITS;
    static const uint kInstanceIDBits = HIT_INFO_INSTANCE_ID_BITS;
    static const uint kPrimitiveIndexBits = HIT_INFO_PRIMITIVE_INDEX_BITS;
#else
    static const uint kTypeBits = 1;
    static const uint kInstanceIDBits = 1;
    static const uint kPrimitiveIndexBits = 1;
#endif

    static const uint kTypeOffset = 32u - kTypeBits;
    static const uint kInstanceIDOffset = kPrimitiveIndexBits;

    static const uint kHeaderBits = kTypeBits + kInstanceIDBits + kPrimitiveIndexBits;

    PackedHitInfo data;

    /** Initialize empty (invalid) hit info.
    */
    __init()
    {
        data = {};
    }

    /** Initialize hit info from a packed hit info.
        \param[in] packed Packed hit information.
    */
    __init(const PackedHitInfo packed)
    {
        data = packed;
    }

    /** Initialize hit info from a triangle hit.
        \param[in] triangleHit Triangle hit information.
    */
    __init(const TriangleHit triangleHit)
    {
        data = triangleHit.pack();
    }

    /** Initialize hit info from a displaced triangle hit.
        \param[in] displacedTriangleHit Displaced triangle hit information.
    */
    __init(const DisplacedTriangleHit displacedTriangleHit)
    {
        data = displacedTriangleHit.pack();
    }

    /** Initialize hit info from a curve hit.
        \param[in] curveHit Curve hit information.
    */
    __init(const CurveHit curveHit)
    {
        data = curveHit.pack();
    }


    /** Initialize hit info from a SDF grid hit.
        \param[in] sdfGridHit Curve hit information.
    */
    __init(const SDFGridHit sdfGridHit)
    {
        data = sdfGridHit.pack();
    }

    /** Initialize hit info from a volume hit.
        \param[in] volumeHit Volume hit information.
    */
    __init(const VolumeHit volumeHit)
    {
        data = volumeHit.pack();
    }

    /** Return true if object represents a valid hit.
    */
    bool isValid()
    {
        return getType() != HitType::None;
    }

    /** Return hit type.
    */
    HitType getType()
    {
        return HitType(data[0] >> kTypeOffset);
    }

    /** Return the triangle hit.
        Only valid if type is HitType::Triangle.
    */
    TriangleHit getTriangleHit()
    {
        return TriangleHit(data);
    }

    /** Return the displaced triangle hit.
        Only valid if type is HitType::DisplacedTriangle.
    */
    DisplacedTriangleHit getDisplacedTriangleHit()
    {
        return DisplacedTriangleHit(data);
    }

    /** Return the curve hit.
        Only valid if type is HitType::Curve.
    */
    CurveHit getCurveHit()
    {
        return CurveHit(data);
    }


    /** Return the SDF grid hit.
        Only valid if type is HitType::SDFGrid.
    */
    SDFGridHit getSDFGridHit()
    {
        return SDFGridHit(data);
    }

    /** Return the volume hit.
        Only valid if type is HitType::Volume.
    */
    VolumeHit getVolumeHit()
    {
        return VolumeHit(data);
    }

    /** Return the packed hit info.
        \return Packed hit info.
    */
    PackedHitInfo getData()
    {
        return data;
    }

    /** Get the geometry instance id from the packed header.
        \return The geometry instance id.
    */
    GeometryInstanceID getInstanceID()
    {
        GeometryInstanceID instanceID;
        uint primitiveIndex;
        unpackHeader(data, instanceID, primitiveIndex);
        return instanceID;
    }

    /** Get the primitive index from the packed header.
        \return The primitive index.
    */
    uint getPrimitiveIndex()
    {
        GeometryInstanceID instanceID;
        uint primitiveIndex;
        unpackHeader(data, instanceID, primitiveIndex);
        return primitiveIndex;
    }

    static void packHeader(inout PackedHitInfo packed, const HitType type)
    {
        packed[0] = uint(type) << kTypeOffset;
    }

    static void packHeader(inout PackedHitInfo packed, const HitType type, const GeometryInstanceID instanceID, const uint primitiveIndex)
    {
        if (kHeaderBits <= 32)
        {
            packed[0] = (uint(type) << kTypeOffset) | (instanceID.index << kInstanceIDOffset) | primitiveIndex;
        }
        else
        {
            packed[0] = (uint(type) << kTypeOffset) | instanceID.index;
            packed[1] = primitiveIndex;
        }
    }

    static void unpackHeader(const PackedHitInfo packed, out GeometryInstanceID instanceID, out uint primitiveIndex)
    {
        if (kHeaderBits <= 32)
        {
            instanceID.index = { (packed[0] >> kInstanceIDOffset) & ((1 << kInstanceIDBits) - 1) };
            primitiveIndex = packed[0] & ((1 << kPrimitiveIndexBits) - 1);
        }
        else
        {
            instanceID.index = { packed[0] & ((1 << kInstanceIDBits) - 1) };
            primitiveIndex = packed[1];
        }
    }
};
