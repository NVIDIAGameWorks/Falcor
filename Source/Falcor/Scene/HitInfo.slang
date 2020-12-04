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
import Utils.Math.FormatConversion;
__exported import Scene.HitInfoType;

/** Ray hit information.

    The fields in the 'HitInfo' struct uniquely identifies a hit point in terms
    of instance ID and primitive index, together with barycentrics.

    If the host sets the following defines, the struct includes helpers
    for packing/unpacking the hit information. Use 'PackedHitInfo' in your code.

    - HIT_INSTANCE_TYPE_BITS    Bits needed to encode the instance type.
    - HIT_INSTANCE_INDEX_BITS   Bits needed to encode the mesh instance ID of the hit.
    - HIT_PRIMITIVE_INDEX_BITS  Bits needed to encode the primitive index of the hit.

    If a define is zero, no bits are needed (the field has only one possible value = 0).

*/

#if defined(HIT_INSTANCE_TYPE_BITS) && defined(HIT_INSTANCE_INDEX_BITS) && defined(HIT_PRIMITIVE_INDEX_BITS)
#if ((HIT_INSTANCE_TYPE_BITS) + (HIT_INSTANCE_INDEX_BITS) + (HIT_PRIMITIVE_INDEX_BITS)) <= 32
    typedef uint2 PackedHitInfo;
#elif ((HIT_INSTANCE_TYPE_BITS) + (HIT_INSTANCE_INDEX_BITS)) <= 32 && (HIT_PRIMITIVE_INDEX_BITS) <= 32
    typedef uint3 PackedHitInfo;
#else
    #error HitInfo instance or primitive index bits exceed the maximum supported
#endif
#endif

struct HitInfo
{
    InstanceType    type = InstanceType::TriangleMesh;  ///< Type of instance.
    uint            instanceID;                         ///< Instance ID at hit.
    uint            primitiveIndex;                     ///< Primitive index at hit.
    float2          barycentrics;                       ///< Barycentric coordinates at ray hit, always in [0,1].

    static const uint kInvalidIndex = 0xffffffff;

    /** Return instance type that was hit.
    */
    InstanceType getType()
    {
        return type;
    }

    /** Return the barycentric weights.
    */
    float3 getBarycentricWeights()
    {
        return float3(1.f - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);
    }

#if defined(HIT_INSTANCE_TYPE_BITS) && defined(HIT_INSTANCE_INDEX_BITS) && defined(HIT_PRIMITIVE_INDEX_BITS)

    static const uint kInstanceTypeBits = HIT_INSTANCE_TYPE_BITS;
    static const uint kInstanceIndexBits = HIT_INSTANCE_INDEX_BITS;
    static const uint kPrimitiveIndexBits = HIT_PRIMITIVE_INDEX_BITS;

#if ((HIT_INSTANCE_TYPE_BITS) + (HIT_INSTANCE_INDEX_BITS) + (HIT_PRIMITIVE_INDEX_BITS)) <= 32

    static const uint kInstanceIndexOffset = kPrimitiveIndexBits;
    static const uint kInstanceTypeOffset = kInstanceIndexOffset + kInstanceIndexBits;

    /** Encode hit information to packed 64-bit format.
    */
    PackedHitInfo encode()
    {
        PackedHitInfo packed;
        packed.x = (uint(type) << kInstanceTypeOffset) | (instanceID << kInstanceIndexOffset) | primitiveIndex;
        packed.y = packUnorm2x16_unsafe(barycentrics);
        return packed;
    }

    /** Decode hit information from packed 64-bit format.
        \return True if the hit information is valid.
    */
    [mutating] bool decode(PackedHitInfo packed)
    {
        type = InstanceType(packed.x >> kInstanceTypeOffset);
        instanceID = (packed.x >> kInstanceIndexOffset) & ((1 << kInstanceIndexBits) - 1);
        primitiveIndex = packed.x & ((1 << kPrimitiveIndexBits) - 1);
        barycentrics = unpackUnorm2x16(packed.y);
        return packed.x != kInvalidIndex;
    }

#elif ((HIT_INSTANCE_TYPE_BITS) + (HIT_INSTANCE_INDEX_BITS)) <= 32 && (HIT_PRIMITIVE_INDEX_BITS) <= 32

    /** Encode hit information to packed 96-bit format.
    */
    PackedHitInfo encode()
    {
        PackedHitInfo packed;
        packed.x = (uint(type) << kInstanceIndexBits) | instanceID;
        packed.y = primitiveIndex;
        packed.z = packUnorm2x16_unsafe(barycentrics);
        return packed;
    }

    /** Decode hit information from packed 96-bit format.
        \return True if the hit information is valid.
    */
    [mutating] bool decode(PackedHitInfo packed)
    {
        type = InstanceType(packed.x >> kInstanceIndexBits);
        instanceID = packed.x & ((1 << kInstanceIndexBits) - 1);
        primitiveIndex = packed.y;
        barycentrics = unpackUnorm2x16(packed.z);
        return packed.x != kInvalidIndex;
    }

#endif
#endif
};
