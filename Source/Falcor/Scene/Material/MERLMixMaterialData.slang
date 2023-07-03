/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "Scene/Material/TextureHandle.slang"
#include "Scene/Material/MaterialTypes.slang"
#include "Scene/Material/MaterialData.slang"
#else
__exported import Scene.Material.TextureHandle;
__exported import Scene.Material.MaterialTypes;
__exported import Scene.Material.MaterialData;
#endif

BEGIN_NAMESPACE_FALCOR

/** This is a host/device structure that describes a measured MERLMix material.
*/
struct MERLMixMaterialData
{
    // MaterialHeader (16B) is stored just before this struct in memory.
    uint flags = 0;                 ///< Material flags and packed sampler IDs. See accessors below.
    uint brdfCount = 0;             ///< Number of loaded BRDFs.
    uint byteStride = 0;            ///< Stride in bytes between each BRDF in the data buffer.
    uint bufferID = 0;              ///< Buffer ID in material system where BRDF data is stored.
    uint extraDataOffset = 0;       ///< Offset in bytes to where extra data for sampling is stored.
    uint extraDataStride = 0;       ///< Stride in bytes between each struct of extra data in the data buffer.

    // Texture handles (4B each).
    TextureHandle texNormalMap;
    TextureHandle texIndexMap;
    TextureHandle texAlbedoLUT;     ///< Texture handle for albedo LUT.

    static constexpr uint kAlbedoLUTSize = 256;
    static constexpr uint kNormalMapTypeBits = 2;
    static constexpr uint kNormalMapTypeOffset = 0;
    static constexpr uint kLUTSamplerIDOffset = kNormalMapTypeOffset + kNormalMapTypeBits;
    static constexpr uint kIndexSamplerIDOffset = kLUTSamplerIDOffset + MaterialHeader::kSamplerIDBits;

    SETTER_DECL void setNormalMapType(NormalMapType type) { flags = PACK_BITS(kNormalMapTypeBits, kNormalMapTypeOffset, flags, (uint)type); }
    NormalMapType getNormalMapType() CONST_FUNCTION { return NormalMapType(EXTRACT_BITS(kNormalMapTypeBits, kNormalMapTypeOffset, flags)); }

    SETTER_DECL void setLUTSamplerID(uint samplerID) { flags = PACK_BITS(MaterialHeader::kSamplerIDBits, kLUTSamplerIDOffset, flags, samplerID); }
    uint getLUTSamplerID() CONST_FUNCTION { return EXTRACT_BITS(MaterialHeader::kSamplerIDBits, kLUTSamplerIDOffset, flags); }

    SETTER_DECL void setIndexSamplerID(uint samplerID) { flags = PACK_BITS(MaterialHeader::kSamplerIDBits, kIndexSamplerIDOffset, flags, samplerID); }
    uint getIndexSamplerID() CONST_FUNCTION { return EXTRACT_BITS(MaterialHeader::kSamplerIDBits, kIndexSamplerIDOffset, flags); }
};

END_NAMESPACE_FALCOR
