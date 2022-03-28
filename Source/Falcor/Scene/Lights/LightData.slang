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
#include "Utils/Math/MathConstants.slangh"

BEGIN_NAMESPACE_FALCOR

/** Types of light sources. Used in LightData structure.
*/
enum class LightType : uint32_t
{
    Point,          ///< Point light source, can be a spot light if its opening angle is < 2pi
    Directional,    ///< Directional light source
    Distant,        ///< Distant light that subtends a non-zero solid angle
    Rect,           ///< Quad shaped area light source
    Disc,           ///< Disc shaped area light source
    Sphere,         ///< Spherical area light source
};

/** This is a host/device structure that describes analytic light sources.
*/
struct LightData
{
    float3   posW               = float3(0, 0, 0);  ///< World-space position of the center of a light source
    uint32_t type               = uint(LightType::Point); ///< Type of the light source (see above)
    float3   dirW               = float3(0, -1, 0); ///< World-space orientation of the light source (normalized).
    float    openingAngle       = float(M_PI);      ///< For point (spot) light: Opening half-angle of a spot light cut-off, pi by default (full sphere).
    float3   intensity          = float3(1, 1, 1);  ///< Emitted radiance of th light source
    float    cosOpeningAngle    = -1.f;             ///< For point (spot) light: cos(openingAngle), -1 by default because openingAngle is pi by default
    float    cosSubtendedAngle  = 0.9999893f;       ///< For distant light; cosine of the half-angle subtended by the light. Default corresponds to the sun as viewed from earth
    float    penumbraAngle      = 0.f;              ///< For point (spot) light: Opening half-angle of penumbra region in radians, usually does not exceed openingAngle. 0.f by default, meaning a spot light with hard cut-off
    float2   _pad0;

    // Extra parameters for analytic area lights
    float3   tangent            = float3(0);        ///< Tangent vector of the light shape
    float    surfaceArea        = 0.f;              ///< Surface area of the light shape
    float3   bitangent          = float3(0);        ///< Bitangent vector of the light shape
    float    _pad1;
    float4x4 transMat           = {};               ///< Transformation matrix of the light shape, from local to world space.
    float4x4 transMatIT         = {};               ///< Inverse-transpose of transformation matrix of the light shape
};

END_NAMESPACE_FALCOR
