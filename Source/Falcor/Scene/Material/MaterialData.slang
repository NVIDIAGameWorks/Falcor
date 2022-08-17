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
#pragma once
#include "Utils/HostDeviceShared.slangh"

#ifdef HOST_CODE
#include "MaterialTypes.slang"
#else
__exported import MaterialTypes;
#endif

BEGIN_NAMESPACE_FALCOR

/** This is a host/device structure for material header data for all material types (8B).
*/
struct MaterialHeader
{
    uint2 packedData = {};

    static const uint kMaterialTypeBits = 16;
    static const uint kNestedPriorityBits = 4;
    static const uint kLobeTypeBits = 8; // Only 6 bits needed if packing LobeType
    static const uint kSamplerIDBits = 8;
    static const uint kAlphaModeBits = 1;
    static const uint kAlphaThresholdBits = 16; // Using float16_t format

    // packedData.x bit layout
    static const uint kMaterialTypeOffset = 0;
    static const uint kNestedPriorityOffset = kMaterialTypeOffset + kMaterialTypeBits;
    static const uint kLobeTypeOffset = kNestedPriorityOffset + kNestedPriorityBits;
    static const uint kDoubleSidedFlagOffset = kLobeTypeOffset + kLobeTypeBits;
    static const uint kThinSurfaceFlagOffset = kDoubleSidedFlagOffset + 1;
    static const uint kEmissiveFlagOffset = kThinSurfaceFlagOffset + 1;
    static const uint kIsBasicMaterialFlagOffset = kEmissiveFlagOffset + 1;

    static const uint kTotalHeaderBitsX = kIsBasicMaterialFlagOffset + 1;

    // packedData.y bit layout
    static const uint kAlphaThresholdOffset = 0;
    static const uint kAlphaModeOffset = kAlphaThresholdOffset + kAlphaThresholdBits;
    static const uint kSamplerIDOffset = kAlphaModeOffset + kAlphaModeBits;
    static const uint kEnableLightProfileOffset = kSamplerIDOffset + kSamplerIDBits;
    static const uint kDeltaSpecularFlagOffset = kEnableLightProfileOffset + 1;

    static const uint kTotalHeaderBitsY = kDeltaSpecularFlagOffset + 1;


    /** Set material type.
    */
    SETTER_DECL void setMaterialType(MaterialType type) { packedData.x = PACK_BITS(kMaterialTypeBits, kMaterialTypeOffset, packedData.x, (uint)type); }

    /** Get material type.
    */
    MaterialType getMaterialType() CONST_FUNCTION { return MaterialType(EXTRACT_BITS(kMaterialTypeBits, kMaterialTypeOffset, packedData.x)); }

    /** Set alpha testing mode.
    */
    SETTER_DECL void setAlphaMode(AlphaMode mode) { packedData.y = PACK_BITS(kAlphaModeBits, kAlphaModeOffset, packedData.y, (uint)mode); }

    /** Get material type.
    */
    AlphaMode getAlphaMode() CONST_FUNCTION { return AlphaMode(EXTRACT_BITS(kAlphaModeBits, kAlphaModeOffset, packedData.y)); }

    /** Set alpha threshold.
    */
    SETTER_DECL void setAlphaThreshold(float16_t alphaThreshold) { packedData.y = PACK_BITS_UNSAFE(kAlphaThresholdBits, kAlphaThresholdOffset, packedData.y, (uint)asuint16(alphaThreshold)); }

    /** Get alpha threshold.
    */
    float16_t getAlphaThreshold() CONST_FUNCTION { return asfloat16((uint16_t)EXTRACT_BITS(kAlphaThresholdBits, kAlphaThresholdOffset, packedData.y)); }

    /** Set the nested priority used for nested dielectrics.
    */
    SETTER_DECL void setNestedPriority(uint32_t priority) { packedData.x = PACK_BITS(kNestedPriorityBits, kNestedPriorityOffset, packedData.x, priority); }

    /** Get the nested priority used for nested dielectrics.
        \return Nested priority, with 0 reserved for the highest possible priority.
    */
    uint getNestedPriority() CONST_FUNCTION { return EXTRACT_BITS(kNestedPriorityBits, kNestedPriorityOffset, packedData.x); }

    /** Set active BxDF lobes.
        \param[in] activeLobes Bit mask of active lobes. See LobeType.
    */
    SETTER_DECL void setActiveLobes(uint activeLobes) { packedData.x = PACK_BITS(kLobeTypeBits, kLobeTypeOffset, packedData.x, activeLobes); }

    /** Get active BxDF lobes.
        \return Bit mask of active lobes. See LobeType.
    */
    uint getActiveLobes() CONST_FUNCTION { return EXTRACT_BITS(kLobeTypeBits, kLobeTypeOffset, packedData.x); }

    /** Set default texture sampler ID.
    */
    SETTER_DECL void setDefaultTextureSamplerID(uint samplerID) { packedData.y = PACK_BITS(kSamplerIDBits, kSamplerIDOffset, packedData.y, samplerID); }

    /** Get default texture sampler ID.
    */
    uint getDefaultTextureSamplerID() CONST_FUNCTION { return EXTRACT_BITS(kSamplerIDBits, kSamplerIDOffset, packedData.y); }

    /** Set double-sided flag.
    */
    SETTER_DECL void setDoubleSided(bool doubleSided) { packedData.x = PACK_BITS(1, kDoubleSidedFlagOffset, packedData.x, doubleSided ? 1 : 0); }

    /** Get double-sided flag.
    */
    bool isDoubleSided() CONST_FUNCTION { return packedData.x & (1u << kDoubleSidedFlagOffset); }

    /** Set thin surface flag.
    */
    SETTER_DECL void setThinSurface(bool thinSurface) { packedData.x = PACK_BITS(1, kThinSurfaceFlagOffset, packedData.x, thinSurface ? 1 : 0); }

    /** Get thin surface flag.
    */
    bool isThinSurface() CONST_FUNCTION { return packedData.x & (1u << kThinSurfaceFlagOffset); }

    /** Set emissive flag.
    */
    SETTER_DECL void setEmissive(bool isEmissive) { packedData.x = PACK_BITS(1, kEmissiveFlagOffset, packedData.x, isEmissive ? 1 : 0); }

    /** Get emissive flag.
    */
    bool isEmissive() CONST_FUNCTION { return packedData.x & (1u << kEmissiveFlagOffset); }

    /** Set basic material flag. This flag is an optimization to allow quick type checking.
    */
    SETTER_DECL void setIsBasicMaterial(bool isBasicMaterial) { packedData.x = PACK_BITS(1, kIsBasicMaterialFlagOffset, packedData.x, isBasicMaterial ? 1 : 0); }

    /** Get basic material flag. This flag is an optimization to allow quick type checking.
    */
    bool isBasicMaterial() CONST_FUNCTION { return packedData.x & (1u << kIsBasicMaterialFlagOffset); }

    /** Set light profile enable flag. // DEMO21
    */
    SETTER_DECL void setEnableLightProfile(bool enableLightProfile) { packedData.y = PACK_BITS(1, kEnableLightProfileOffset, packedData.y, enableLightProfile ? 1 : 0); }

    /** Get light profile enable flag.
    */
    bool isLightProfileEnabled() CONST_FUNCTION { return packedData.y & (1u << kEnableLightProfileOffset); }

    /** Set delta specular flag. This flag is set if the material only has delta specular reflection/transmission (and can skip NEE).
    */
    SETTER_DECL void setDeltaSpecular(bool deltaSpecular) { packedData.y = PACK_BITS(1, kDeltaSpecularFlagOffset, packedData.y, deltaSpecular ? 1 : 0); }

    /** Get delta specular flag. This flag is set if the material only has delta specular reflection/transmission (and can skip NEE).
    */
    bool isDeltaSpecular() CONST_FUNCTION { return packedData.y & (1u << kDeltaSpecularFlagOffset); }

#ifdef HOST_CODE
    friend bool operator==(const MaterialHeader& lhs, const MaterialHeader& rhs);
    friend bool operator!=(const MaterialHeader& lhs, const MaterialHeader& rhs) { return !(lhs == rhs); }
#endif
};

/** This is a host/device structure for material payload data (120B).
    The format of the data depends on the material type.
*/
struct MaterialPayload
{
    uint data[30];
};

/** This is a host/device structure representing a blob of material data (128B).
    The material data blob consists of a header and a payload. The header is always valid.
    The blob is sized to fit into one cacheline for efficient access. Do not change the size.
    All material types should keep their main material data within this footprint.
    Material that need more can reference additional sideband data stored elsewhere.
*/
struct MaterialDataBlob
{
    MaterialHeader header; // 8B
    MaterialPayload payload; // 120B
};

END_NAMESPACE_FALCOR
