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
import NRDData;

#if !defined(OUTPUT_NRD_DATA) || !defined(OUTPUT_NRD_ADDITIONAL_DATA)
#error NRDBuffers.slang is missing defines
#endif

static const bool kOutputNRDData = OUTPUT_NRD_DATA;
static const bool kOutputNRDAdditionalData = OUTPUT_NRD_ADDITIONAL_DATA;

struct NRDBuffers
{
    RWStructuredBuffer<NRDRadiance> sampleRadiance;             ///< Output per-sample NRD radiance data. Only valid if kOutputNRDData == true.
    RWStructuredBuffer<float>       sampleHitDist;              ///< Output per-sample NRD hit distance data. Only valid if kOutputNRDData == true.
    RWStructuredBuffer<float4>      sampleEmission;             ///< Output per-sample NRD emission data. Only valid if kOutputNRDData == true.
    RWStructuredBuffer<float4>      sampleReflectance;          ///< Output per-sample NRD reflectance data. Only valid if kOutputNRDData == true.

    RWTexture2D<float4> primaryHitEmission;                     ///< Output per-pixel primary hit emission. Only valid if kOutputNRDData == true.
    RWTexture2D<float4> primaryHitDiffuseReflectance;           ///< Output per-pixel primary hit diffuse reflectance. Only valid if kOutputNRDData == true.
    RWTexture2D<float4> primaryHitSpecularReflectance;          ///< Output per-pixel primary hit specular reflectance. Only valid if kOutputNRDData == true.

    RWTexture2D<float4> deltaReflectionReflectance;             ///< Output per-pixel delta reflection reflectance. Only valid if kOutputNRDAdditionalData == true.
    RWTexture2D<float4> deltaReflectionEmission;                ///< Output per-pixel delta reflection emission. Only valid if kOutputNRDAdditionalData == true.
    RWTexture2D<float4> deltaReflectionNormWRoughMaterialID;    ///< Output per-pixel delta reflection world normal, roughness, and material ID. Only valid if kOutputNRDAdditionalData == true.
    RWTexture2D<float>  deltaReflectionPathLength;              ///< Output per-pixel delta reflection path length. Only valid if kOutputNRDAdditionalData == true.
    RWTexture2D<float>  deltaReflectionHitDist;                 ///< Output per-pixel delta reflection hit distance. Only valid if kOutputNRDAdditionalData == true.

    RWTexture2D<float4> deltaTransmissionReflectance;           ///< Output per-pixel delta transmission reflectance. Only valid if kOutputNRDAdditionalData == true.
    RWTexture2D<float4> deltaTransmissionEmission;              ///< Output per-pixel delta transmission emission. Only valid if kOutputNRDAdditionalData == true.
    RWTexture2D<float4> deltaTransmissionNormWRoughMaterialID;  ///< Output per-pixel delta transmission world normal, roughness, and material ID. Only valid if kOutputNRDAdditionalData == true.
    RWTexture2D<float>  deltaTransmissionPathLength;            ///< Output per-pixel delta transmission path length. Only valid if kOutputNRDAdditionalData == true.
    RWTexture2D<float4> deltaTransmissionPosW;                  ///< Output per-pixel delta transmission world position. Only valid if kOutputNRDAdditionalData == true.
};
