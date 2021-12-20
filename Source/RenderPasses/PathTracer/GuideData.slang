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
import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;

static const float kEnvMapDepth = 100000000.0f; // Same constant as defined in GBufferRT.slang

/** Struct holding guide data for denoising.
*/
struct GuideData
{
    uint4 data;
    float3 reflectionPos;
    float dummy;

    float3 getAlbedo()
    {
        return unpackR11G11B10(data[0]);
    }

    [mutating] void setAlbedo(float3 albedo)
    {
        data[0] = packR11G11B10(albedo);
    }

    float3 getNormal()
    {
        return decodeNormal2x16(data[1]);
    }

    [mutating] void setNormal(float3 normal)
    {
        data[1] = encodeNormal2x16(normal);
    }

    float3 getSpecularAlbedo()
    {
        return unpackR11G11B10(data[2]);
    }

    [mutating] void setSpecularAlbedo(float3 specular)
    {
        data[2] = packR11G11B10(specular);
    }

    float3 getIndirectAlbedo()
    {
        return unpackR11G11B10(data[3]);
    }

    [mutating] void setIndirectAlbedo(float3 albedo)
    {
        data[3] = packR11G11B10(albedo);
    }

    float3 getReflectionPos()
    {
        return reflectionPos;
    }

    [mutating] void setReflectionPos(float3 pos)
    {
        reflectionPos = pos;
    }
};
