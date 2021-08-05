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
#include "Scene/ScenePrimitiveDefines.slangh"
import Utils.Math.FormatConversion;
__exported import Scene.HitInfoType;
__exported import Scene.SceneTypes;

/** Ray hit information.

    'HitInfo' is a polymorphic container for storing ray hit information.
    'PackedHitInfo' holds a packed version of the data in 'HitInfo'.
    Both use the upper bits of the first uint to store the type of the hit.

    HitInfo stores one of the following type specific structs:
    TriangleHit, DisplacedTriangleHit, CurveHit

    TriangleHit, DisplacedTriangleHit and CurveHit all store the hit point in terms of
    an instance ID and a primitive index, together with barycentrics.

    DisplacedTriangleHit additionally stores the displacement offset.

    If the host sets the following defines, the HitInfo struct size is optimized and helpers
    for packing/unpacking are added. Use 'PackedHitInfo' in your code.
    A zero initialized struct represent hit type 'None'.

    - HIT_INFO_DEFINES              Marks that the rest of the defines are available.
    - HIT_INFO_DATA_SIZE            Number of uints of data in HitInfo.
    - HIT_INFO_PACKED_DATA_SIZE     Number of uints to encode the HitInfo.
    - HIT_TYPE_BITS                 Bits needed to encode the hit type.
    - HIT_INSTANCE_INDEX_BITS       Bits needed to encode the mesh instance ID of the hit.
    - HIT_PRIMITIVE_INDEX_BITS      Bits needed to encode the primitive index of the hit.

    If a bit size define is zero, no bits are needed (the field has only one possible value = 0).

*/

#if defined(HIT_INFO_PACKED_DATA_SIZE)
typedef vector<uint, HIT_INFO_PACKED_DATA_SIZE> PackedHitInfo;
#else
typedef uint4 PackedHitInfo; // Dummy type.
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
*/
struct TriangleHit : GeometryHit
{
};

/** Displaced triangle hit information.
*/
struct DisplacedTriangleHit : GeometryHit
{
    float displacement;
};

/** Curve hit information.
*/
struct CurveHit : GeometryHit
{
};

/** Polymorphic hit information type.
*/
struct HitInfo
{
#ifdef HIT_INFO_DEFINES
    static const uint kDataSize = HIT_INFO_DATA_SIZE;
    static const uint kTypeBits = HIT_TYPE_BITS;

    static const uint kInstanceIndexBits = HIT_INSTANCE_INDEX_BITS;
    static const uint kPrimitiveIndexBits = HIT_PRIMITIVE_INDEX_BITS;
    static const uint kInstanceIndexOffset = kPrimitiveIndexBits;

#else
    static const uint kDataSize = 4;
    static const uint kTypeBits = 2;
#endif

    static const uint kTypeOffset = 32u - kTypeBits;
    static const uint kData0Mask = (1u << kTypeOffset) - 1u;

    uint data[kDataSize];

    /** Initialize empty (invalid) hit info.
    */
    __init()
    {
        clear();
    }

    /** Initialize hit info from a packed hit info.
        \param[in] packed Packed hit information.
    */
    __init(const PackedHitInfo packed)
    {
        unpack(packed);
    }

    /** Initialize hit info from a triangle hit.
        \param[in] triangleHit Triangle hit information.
    */
    __init(const TriangleHit triangleHit)
    {
        data[0] = (uint(HitType::Triangle) << kTypeOffset) | (triangleHit.instanceID.index & kData0Mask);
        data[1] = triangleHit.primitiveIndex;
        data[2] = asuint(triangleHit.barycentrics.x);
        data[3] = asuint(triangleHit.barycentrics.y);
    }

    /** Initialize hit info from a displaced triangle hit.
        \param[in] displacedTriangleHit Displaced triangle hit information.
    */
    __init(const DisplacedTriangleHit displacedTriangleHit)
    {
#if IS_SET(SCENE_PRIMITIVE_TYPE_FLAGS, PRIMITIVE_TYPE_DISPLACED_TRIANGLE_MESH)
        data[0] = (uint(HitType::DisplacedTriangle) << kTypeOffset) | (displacedTriangleHit.instanceID.index & kData0Mask);
        data[1] = displacedTriangleHit.primitiveIndex;
        data[2] = asuint(displacedTriangleHit.barycentrics.x);
        data[3] = asuint(displacedTriangleHit.barycentrics.y);
        data[4] = asuint(displacedTriangleHit.displacement);
#endif
    }

    /** Initialize hit info from a curve hit.
        \param[in] curveHit Curve hit information.
    */
    __init(const CurveHit curveHit)
    {
#if IS_SET(SCENE_PRIMITIVE_TYPE_FLAGS, PRIMITIVE_TYPE_CURVE)
        data[0] = (uint(HitType::Curve) << kTypeOffset) | (curveHit.instanceID.index & kData0Mask);
        data[1] = curveHit.primitiveIndex;
        data[2] = asuint(curveHit.barycentrics.x);
        data[3] = asuint(curveHit.barycentrics.y);
#endif
    }

    /** Clear hit. This marks the hit as invalid.
    */
    [mutating] void clear()
    {
        data[0] = 0;
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
        TriangleHit triangleHit;
        triangleHit.instanceID.index = data[0] & kData0Mask;
        triangleHit.primitiveIndex = data[1];
        triangleHit.barycentrics.x = asfloat(data[2]);
        triangleHit.barycentrics.y = asfloat(data[3]);
        return triangleHit;
    }

    /** Return the displaced triangle hit.
        Only valid if type is HitType::DisplacedTriangle.
    */
    DisplacedTriangleHit getDisplacedTriangleHit()
    {
        DisplacedTriangleHit displacedTriangleHit = {};
#if IS_SET(SCENE_PRIMITIVE_TYPE_FLAGS, PRIMITIVE_TYPE_DISPLACED_TRIANGLE_MESH)
        displacedTriangleHit.instanceID.index = data[0] & kData0Mask;
        displacedTriangleHit.primitiveIndex = data[1];
        displacedTriangleHit.barycentrics.x = asfloat(data[2]);
        displacedTriangleHit.barycentrics.y = asfloat(data[3]);
        displacedTriangleHit.displacement = asfloat(data[4]);
#endif
        return displacedTriangleHit;
    }

    /** Return the curve hit.
        Only valid if type is HitType::Curve.
    */
    CurveHit getCurveHit()
    {
        CurveHit curveHit = {};
#if IS_SET(SCENE_PRIMITIVE_TYPE_FLAGS, PRIMITIVE_TYPE_CURVE)
        curveHit.instanceID.index = data[0] & kData0Mask;
        curveHit.primitiveIndex = data[1];
        curveHit.barycentrics.x = asfloat(data[2]);
        curveHit.barycentrics.y = asfloat(data[3]);
#endif
        return curveHit;
    }

    /** Set from a packed hit info.
        \param[in] packed Packed hit info.
    */
    [mutating] void unpack(const PackedHitInfo packed)
    {
#ifdef HIT_INFO_DEFINES
        const HitType type = HitType(packed[0] >> kTypeOffset);
        if (type == HitType::Triangle || type == HitType::DisplacedTriangle || type == HitType::Curve)
        {
#if ((HIT_TYPE_BITS) + (HIT_INSTANCE_INDEX_BITS) + (HIT_PRIMITIVE_INDEX_BITS)) <= 32
            uint instanceID = { (packed[0] >> kInstanceIndexOffset) & ((1 << kInstanceIndexBits) - 1) };
            uint primitiveIndex = packed[0] & ((1 << kPrimitiveIndexBits) - 1);
            float2 barycentrics = unpackUnorm2x16(packed[1]);
#else
            uint instanceID = { packed[0] & ((1 << kInstanceIndexBits) - 1) };
            uint primitiveIndex = packed[1];
            float2 barycentrics = unpackUnorm2x16(packed[2]);
#endif
            data[0] = (uint(type) << kTypeOffset) | (instanceID & kData0Mask);
            data[1] = primitiveIndex;
            data[2] = asuint(barycentrics.x);
            data[3] = asuint(barycentrics.y);

#if IS_SET(SCENE_PRIMITIVE_TYPE_FLAGS, PRIMITIVE_TYPE_DISPLACED_TRIANGLE_MESH)
            if (type == HitType::DisplacedTriangle)
            {
#if ((HIT_TYPE_BITS) + (HIT_INSTANCE_INDEX_BITS) + (HIT_PRIMITIVE_INDEX_BITS)) <= 32
                data[4] = packed[2];
#else
                data[4] = packed[3];
#endif
            }
#endif
        }
        else
#endif
        {
            clear();
        }
    }

    /** Return the packed hit info.
        \return Packed hit info.
    */
    PackedHitInfo pack()
    {
        PackedHitInfo packed = {};
#ifdef HIT_INFO_DEFINES
        const HitType type = getType();
        if (type == HitType::Triangle || type == HitType::DisplacedTriangle || type == HitType::Curve)
        {
            const uint instanceID = data[0] & kData0Mask;
            const uint primitiveIndex = data[1];
            const float2 barycentrics = { asfloat(data[2]), asfloat(data[3]) };
#if ((HIT_TYPE_BITS) + (HIT_INSTANCE_INDEX_BITS) + (HIT_PRIMITIVE_INDEX_BITS)) <= 32
            packed[0] = (uint(type) << kTypeOffset) | (instanceID << kInstanceIndexOffset) | primitiveIndex;
            packed[1] = packUnorm2x16_unsafe(barycentrics);
#else
            packed[0] = (uint(type) << kTypeOffset) | instanceID;
            packed[1] = primitiveIndex;
            packed[2] = packUnorm2x16_unsafe(barycentrics);
#endif

#if IS_SET(SCENE_PRIMITIVE_TYPE_FLAGS, PRIMITIVE_TYPE_DISPLACED_TRIANGLE_MESH)
            if (type == HitType::DisplacedTriangle)
            {
#if ((HIT_TYPE_BITS) + (HIT_INSTANCE_INDEX_BITS) + (HIT_PRIMITIVE_INDEX_BITS)) <= 32
                packed[2] = data[4];
#else
                packed[3] = data[4];
#endif
            }
#endif
        }
#endif
        return packed;
    }
};
