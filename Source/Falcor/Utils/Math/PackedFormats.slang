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
 **************************************************************************/
import Utils.Math.MathHelpers;
import Utils.Math.FormatConversion;

/** Encode a normal packed as 2x 16-bit snorms in the octahedral mapping.
*/
uint encodeNormal2x16(float3 normal)
{
    float2 octNormal = ndir_to_oct_snorm(normal);
    return packSnorm2x16(octNormal);
}

/** Decode a normal packed as 2x 16-bit snorms in the octahedral mapping.
*/
float3 decodeNormal2x16(uint packedNormal)
{
    float2 octNormal = unpackSnorm2x16(packedNormal);
    return oct_to_ndir_snorm(octNormal);
}

/** Encode a normal packed as 3x 16-bit snorms. Note: The high 16 bits of the second dword are unused.
*/
uint2 encodeNormal3x16(float3 normal)
{
    uint2 packedNormal;
    packedNormal.x = packSnorm2x16(normal.xy);
    packedNormal.y = packSnorm16(normal.z);
    return packedNormal;
}

/** Decode a normal packed as 3x 16-bit snorms. Note: The high 16 bits of the second dword are unused.
*/
float3 decodeNormal3x16(uint2 packedNormal)
{
    float3 normal;
    normal.xy = unpackSnorm2x16(packedNormal.x);
    normal.z = unpackSnorm16(packedNormal.y);
    return normalize(normal);
}
