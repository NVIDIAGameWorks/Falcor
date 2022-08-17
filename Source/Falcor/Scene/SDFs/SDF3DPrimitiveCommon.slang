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

enum class SDF3DShapeType
{
    Sphere,         ///< A sphere, defined by a radius (1 float).
    Ellipsoid,      ///< An ellipsoid, defined by three axis aligned radii (3 floats).
    Box,            ///< A box, defined by three axis aligned half extents (3 floats).
    Torus,          ///< A torus, defined by a radius (1 floats). Note that this type requires primitive blobbing to have thickness.
    Cone,           ///< A cone defined by tan of the cone angle and a height (1 floats).
    Capsule,        ///< A capsule, defined by a half length (1 float). Note that this type requires primitive blobbing to have thickness.

    Count
};

struct SDF3DPrimitive
{
    SDF3DShapeType shapeType;           ///< The shape type.
    float3 shapeData;                   ///< The shape data used to create the shape.
    float shapeBlobbing;                ///< Blobbing that should be applied to the shape.
    SDFOperationType operationType;     ///< The operation type.
    float operationSmoothing;           ///< Smoothing that should be applied to the operation.
    float3 translation;                 ///< Translation.
    // float3x3 here is okay, as this is not used in CB, only structured buffers, so no padding issues.
    float3x3 invRotationScale;          ///< Inverted rotation and scale matrix.
};

END_NAMESPACE_FALCOR
