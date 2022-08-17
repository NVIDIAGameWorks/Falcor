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

__exported import Scene.SDFs.SDF3DPrimitiveCommon;
import Utils.SDF.SDF3DShapes;
import Utils.SDF.SDFOperations;
import Utils.Math.IntervalArithmetic;

BEGIN_NAMESPACE_FALCOR

extension SDF3DPrimitive
{
    float eval(float3 p)
    {
        return evalShape(p, shapeType, shapeData, shapeBlobbing, translation, invRotationScale);
    }

    float eval(float3 p, float d)
    {
        float dShape = evalShape(p, shapeType, shapeData, shapeBlobbing, translation, invRotationScale);
        return evalOperation(operationType, d, dShape, operationSmoothing);
    }

    float2 evalInterval(float3 pCenter, float3 pHalfExtent, float2 d)
    {
        float2 dShape = evalIntervalShape(pCenter, pHalfExtent, shapeType, shapeData, shapeBlobbing, translation, invRotationScale);
        return evalIntervalOperation(operationType, d, dShape, operationSmoothing);
    }

    static float evalShape(float3 p, SDF3DShapeType shapeType, float3 shapeData, float blobbing, float3 translation, float3x3 invRotationScale)
    {
        // OLD CODE: invRotationScale was tranpose w.r.t. CPU: p = mul(invRotationScale, p - translation);
        // The old code did M.v, where M was actually transposed w.r.t. what was on the CPU because of the col-major CPU and row-major GPU.
        // Matrices are now row-major on both, meaning there is no more any hidden tranpose.
        // However, to keep the exact same semantic as before, we explicitly transpose the invRotationScale here.
        // We choose to transpose rather than do v.M to make the weirdness even more obvious.
        // tdavidovic: I think this is a bug and the transpose shouldn't be there, but don't have a test case with a ground truth to verify
        p = mul(transpose(invRotationScale), p - translation);
        float d = FLT_MAX;

        switch (shapeType)
        {
        case SDF3DShapeType::Sphere:    d = sdfSphere(p, shapeData.x); break;
        case SDF3DShapeType::Ellipsoid: d = sdfEllipsoid(p, shapeData.xyz); break;
        case SDF3DShapeType::Box:       d = sdfBox(p, shapeData.xyz); break;
        case SDF3DShapeType::Torus:     d = sdfTorus(p, shapeData.x); break;
        case SDF3DShapeType::Cone:      d = sdfCone(p, shapeData.x, shapeData.y); break;
        case SDF3DShapeType::Capsule:   d = sdfCapsule(p, shapeData.x); break;
        }

        // Apply blobbing;
        return d - blobbing;
    }

    static float2 evalIntervalShape(float3 pCenter, float3 pHalfExtent, SDF3DShapeType shapeType, float3 shapeData, float blobbing, float3 translation, float3x3 invRotationScale)
    {
        // OLD CODE: invRotationScale was tranpose w.r.t. CPU: pCenter = mul(invRotationScale, pCenter - translation);
        // The old code did M.v, where M was actually transposed w.r.t. what was on the CPU because of the col-major CPU and row-major GPU.
        // Matrices are now row-major on both, meaning there is no more any hidden tranpose.
        // However, to keep the exact same semantic as before, we explicitly transpose the invRotationScale here.
        // We choose to transpose rather than do v.M to make the weirdness even more obvious.
        // tdavidovic: I think this is a bug and the transpose shouldn't be there, but don't have a test case with a ground truth to verify
        pCenter = mul(transpose(invRotationScale), pCenter - translation);
        float3 pMin = pCenter - pHalfExtent;
        float3 pMax = pCenter + pHalfExtent;

        float2 d = float2(FLT_MAX);

        switch (shapeType)
        {
        case SDF3DShapeType::Sphere:    d = sdfIntervalSphere(pMin, pMax, shapeData.x); break;
        case SDF3DShapeType::Ellipsoid: d = sdfIntervalEllipsoid(pMin, pMax, shapeData.xyz); break;
        case SDF3DShapeType::Box:       d = sdfIntervalBox(pMin, pMax, shapeData.xyz); break;
        case SDF3DShapeType::Torus:     d = sdfIntervalTorus(pMin, pMax, shapeData.x); break;
        case SDF3DShapeType::Cone:      d = sdfIntervalCone(pMin, pMax, shapeData.x, shapeData.y); break;
        case SDF3DShapeType::Capsule:   d = sdfIntervalCapsule(pMin, pMax, shapeData.x); break;
        }

        // Apply blobbing;
        return ivlSub(d, blobbing);
    }

    static float evalOperation(SDFOperationType operationType, float d, float dShape, float smoothing)
    {
        switch (operationType)
        {
        case SDFOperationType::Union:                 d = sdfUnion(d, dShape); break;
        case SDFOperationType::Subtraction:           d = sdfSubtraction(d, dShape); break;
        case SDFOperationType::Intersection:          d = sdfIntersection(d, dShape); break;
        case SDFOperationType::SmoothUnion:           d = sdfSmoothUnion(d, dShape, smoothing); break;
        case SDFOperationType::SmoothSubtraction:     d = sdfSmoothSubtraction(d, dShape, smoothing); break;
        case SDFOperationType::SmoothIntersection:    d = sdfSmoothIntersection(d, dShape, smoothing); break;
        }

        return d;
    }

    static float2 evalIntervalOperation(SDFOperationType operationType, float2 d, float2 dShape, float smoothing)
    {
        switch (operationType)
        {
        case SDFOperationType::Union:                 d = sdfIntervalUnion(d, dShape); break;
        case SDFOperationType::Subtraction:           d = sdfIntervalSubtraction(d, dShape); break;
        case SDFOperationType::Intersection:          d = sdfIntervalIntersection(d, dShape); break;
        case SDFOperationType::SmoothUnion:           d = sdfIntervalSmoothUnion(d, dShape, smoothing); break;
        case SDFOperationType::SmoothSubtraction:     d = sdfIntervalSmoothSubtraction(d, dShape, smoothing); break;
        case SDFOperationType::SmoothIntersection:    d = sdfIntervalSmoothIntersection(d, dShape, smoothing); break;
        }

        return d;
    }
};

END_NAMESPACE_FALCOR
