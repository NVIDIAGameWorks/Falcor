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
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

#ifdef HOST_CODE
#define SamplerState std::shared_ptr<Sampler>
#define Texture2D std::shared_ptr<Texture>
#endif

/** This is a host/device structure that describes light probe resources.
*/
struct LightProbeResources
{
    Texture2D origTexture;      ///< The original texture
    Texture2D diffuseTexture;   ///< Texture containing pre-integrated diffuse (LD) term
    Texture2D specularTexture;  ///< Texture containing pre-integrated specular (LD) term
    SamplerState sampler;
};

/** This is a host/device structure that describes shared light probe resources.
*/
struct LightProbeSharedResources
{
    Texture2D dfgTexture;       ///< Texture containing shared pre-integrated (DFG) term
    SamplerState dfgSampler;
};

/** This is a host/device structure that describes light probe data.
*/
struct LightProbeData
{
    float3 posW         = float3(0);
    float radius        = -1.0f;
    float3 intensity    = float3(1.0f);

    LightProbeResources resources;
    LightProbeSharedResources sharedResources;
};

#ifdef HOST_CODE
#undef SamplerState
#undef Texture2D
#endif

END_NAMESPACE_FALCOR
