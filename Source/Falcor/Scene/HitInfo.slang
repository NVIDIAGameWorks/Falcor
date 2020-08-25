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

/** Ray hit information.

    The fields in this struct uniquely identifies a hit point in terms
    of mesh instance ID and primitive index, together with barycentrics.

    If the host sets the following defines, the struct includes helpers
    for packing/unpacking the hit information into a 64-bit value.

    - HIT_INSTANCE_INDEX_BITS   Bits needed to encode the mesh instance ID of the hit
    - HIT_TRIANGLE_INDEX_BITS   Bits needed to encode the primitive index of the hit

    Together the bit allocations must be 32 or less.

*/
struct HitInfo
{
    uint    meshInstanceID;     ///< Mesh instance ID at hit.
    uint    primitiveIndex;     ///< Primitive index at hit.
    float2  barycentrics;       ///< Barycentric coordinates at ray hit, always in [0,1].

    static const uint kInvalidIndex = 0xffffffff;

    /** Return the barycentric weights.
    */
    float3 getBarycentricWeights()
    {
        return float3(1.f - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);
    }

#if defined(HIT_INSTANCE_INDEX_BITS) && defined(HIT_TRIANGLE_INDEX_BITS)
#if ((HIT_INSTANCE_INDEX_BITS) + (HIT_TRIANGLE_INDEX_BITS)) > 32
    #error HitInfo instance/primitive index bits exceed 32 bits
#endif

    /** Encode hit information to packed format.
    */
    uint2 encode()
    {
        uint2 packed;
        packed.x = (meshInstanceID << (HIT_TRIANGLE_INDEX_BITS)) | primitiveIndex;
        packed.y = packUnorm2x16_unsafe(barycentrics);
        return packed;
    }

    /** Decode hit information from packed format.
        \return True if the hit information is valid.
    */
    [mutating] bool decode(uint2 packed)
    {
        meshInstanceID = packed.x >> (HIT_TRIANGLE_INDEX_BITS);
        primitiveIndex = packed.x & ((1 << (HIT_TRIANGLE_INDEX_BITS)) - 1);
        barycentrics = unpackUnorm2x16(packed.y);
        return packed.x != kInvalidIndex;
    }
#endif
};
