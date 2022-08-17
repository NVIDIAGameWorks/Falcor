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
#include "Utils/SDF/SDFOperationType.slang"
#else
__exported import Utils.SDF.SDFOperationType;
#endif

BEGIN_NAMESPACE_FALCOR

enum class SDF2DShapeType
{
    Circle,
    Square,
    Diamond,
    Heart,
    Chevron,
    Ring,
    Tag,
    Cross,
    Asterisk,
    Infinity,
    Pin,
    Arrow,
    RoundedBox,
    Triangle,
    RoundedLine,
    Vector,
    MarkerOpMarker,
    ArrowFromTwoTris,
    CircleSector
};

enum class ExcludeBorderFlags : uint32_t
{
    None = 0,
    Top = 1,
    Right = 2,
    Bottom = 4,
    Left = 8
};
#ifdef HOST_CODE
FALCOR_ENUM_CLASS_OPERATORS(ExcludeBorderFlags)
#endif

struct Marker2DPayload
{
    uint data[20];
};

struct Marker2DDataBlob
{
    SDF2DShapeType  type;
    Marker2DPayload payload;
};

struct Marker2DTransform
{
    float   scale;
    float   rotation;
    float2  translation;
};

struct SimpleMarker2DData
{
    Marker2DTransform transform;
    float4 color;
};

struct Marker2DRoundedLine
{
    float2  positionA;
    float2  positionB;
    float   width;
};

struct RoundedLineMarker2DData
{
    Marker2DRoundedLine line;
    float4 color;
};

struct VectorMarker2DData
{
    Marker2DRoundedLine line;
    float arrowHeight;
    float4 color;
};

struct TriangleMarker2DData
{
    float2 positionA;
    float2 positionB;
    float2 positionC;
    float4 color;
};

struct RoundedBoxMarker2DData
{
    Marker2DTransform transform;
    float2 halfSides;
    float4 color;
};

struct BasicMarker2D
{
    float2 position;
    SDF2DShapeType type;
    float size;
};

struct MarkerOpMarker2DData
{
    SDFOperationType operation;
    BasicMarker2D markerA;
    BasicMarker2D markerB;
    float4 color;
    float4 dimmedColor;
};

struct ArrowFromTwoTrisMarker2DData
{
    Marker2DRoundedLine line;
    float headLength;
    float headWidth;
    float4 color;
};

struct CircleSectorMarker2DData
{
    float2 position;
    float rotation;
    float angle;
    float maxRadius;
    float minRadius;
    float4 color;
    float4 borderColor;     ///< The color of the border in x,y,z and the border thickness in w. Uses the same alpha as color.a
    uint excludeBorders;    ///< Flags for which borders should be excluded. Flags can be any of ExcludeBorderFlags OR'ed together.
};

END_NAMESPACE_FALCOR
