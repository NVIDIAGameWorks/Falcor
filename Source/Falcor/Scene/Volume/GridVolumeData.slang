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
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

/** This is a host/device structure that describes a grid volume.
*/
struct GridVolumeData
{
    float4x4 transform;                             ///< Local-space to world-space transform.
    float4x4 invTransform;                          ///< World-space to local-space transform.
    float3  boundsMin               = float3(0);    ///< World-space bounds (minimum).
    float   densityScale            = 1.f;          ///< Density scale factor.
    float3  boundsMax               = float3(0);    ///< World-space bounds (maximum).
    float   emissionScale           = 1.f;          ///< Emission scale factor.
    uint    densityGrid             = 0;            ///< Index of the density grid.
    uint    emissionGrid            = 0;            ///< Index of the emission grid.
    uint    flags                   = 0;            ///< Flags (contains only emission mode for now).
    float   anisotropy              = 0.f;          ///< Phase function anisotropy (g).
    float3  albedo                  = float3(1);    ///< Medium scattering albedo.
    float   emissionTemperature     = 0.f;          ///< Emission base temperature (K).
};

END_NAMESPACE_FALCOR
