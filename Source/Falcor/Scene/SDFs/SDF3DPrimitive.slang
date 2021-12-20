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
#include "Utils/Math/MathConstants.slangh"

#ifndef HOST_CODE
import Utils.SDF.SDF3DShapes;
import Utils.SDF.SDFOperations;
import Utils.Math.IntervalArithmetic;
#else
#include "Utils/SDF/SDFOperations.slang"
#include "Scene/Transform.h"
#endif

BEGIN_NAMESPACE_FALCOR

enum class SDF3DShapeType
{
    Sphere      = 0,        ///< A sphere, defined by a radius (1 float).
    Ellipsoid   = 1,        ///< An ellipsoid, defined by three axis aligned radii (3 floats).
    Box         = 2,        ///< A box, defined by three axis aligned half extents (3 floats).
    Torus       = 3,        ///< A torus, defined by a radius (1 floats). Note that this type requires primitive blobbing to have thickness.
    Cone        = 4,        ///< A cone defined by tan of the cone angle and a height (1 floats).
    Capsule     = 5,        ///< A capsule, defined by a half length (1 float). Note that this type requires primitive blobbing to have thickness.

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
    float3x3 invRotationScale;          ///< Inverted rotation and scale matrix.


#ifdef HOST_CODE
    SDF3DPrimitive& initCommon(float blobbing, float operationSmoothing, SDFOperationType operationType, const Transform& transform)
    {
        this->operationType = operationType;
        this->shapeBlobbing = blobbing;
        this->operationSmoothing = operationSmoothing;
        this->translation = transform.getTranslation();
        this->invRotationScale = glm::inverse(float3x3(transform.getMatrix()));
        return *this;
    }

    SDF3DPrimitive& initSphere(float radius, float blobbing, float operationSmoothing, SDFOperationType operationType, const Transform& transform)
    {
        this->shapeType            = SDF3DShapeType::Sphere;
        this->shapeData            = float3(radius, 0.0f, 0.0f);
        return initCommon(blobbing, operationSmoothing, operationType, transform);
    }

    SDF3DPrimitive& initEllipsoid(const float3& radii, float blobbing, float operationSmoothing, SDFOperationType operationType, const Transform& transform)
    {
        this->shapeType            = SDF3DShapeType::Ellipsoid;
        this->shapeData            = radii;
        return initCommon(blobbing, operationSmoothing, operationType, transform);
    }

    SDF3DPrimitive& initBox(const float3& halfExtents, float blobbing, float operationSmoothing, SDFOperationType operationType, const Transform& transform)
    {
        this->shapeType            = SDF3DShapeType::Box;
        this->shapeData            = halfExtents;
        return initCommon(blobbing, operationSmoothing, operationType, transform);
    }

    SDF3DPrimitive& initTorus(float radius, float blobbing, float operationSmoothing, SDFOperationType operationType, const Transform& transform)
    {
        this->shapeType            = SDF3DShapeType::Torus;
        this->shapeData            = float3(radius, 0.0f, 0.0f);
        return initCommon(blobbing, operationSmoothing, operationType, transform);
    }

    SDF3DPrimitive& initCone(float angle, float height, float blobbing, float operationSmoothing, SDFOperationType operationType, const Transform& transform)
    {
        this->shapeType            = SDF3DShapeType::Cone;
        this->shapeData            = float3(std::tanf(angle), height, 0.0f);
        return initCommon(blobbing, operationSmoothing, operationType, transform);
    }

    SDF3DPrimitive& initCapsule(float halfLength, float blobbing, float operationSmoothing, SDFOperationType operationType, const Transform& transform)
    {
        this->shapeType            = SDF3DShapeType::Capsule;
        this->shapeData            = float3(halfLength, 0.0f, 0.0f);
        return initCommon(blobbing, operationSmoothing, operationType, transform);
    }

    float calculateRange()
    {
        float range = 0.0f;
        switch (shapeType)
        {
        case SDF3DShapeType::Sphere:    range = shapeData.x; break;
        case SDF3DShapeType::Ellipsoid: range = std::max(std::max(shapeData.x, shapeData.y), shapeData.z); break;
        case SDF3DShapeType::Box:       range = std::max(std::max(shapeData.x, shapeData.y), shapeData.z); break;
        case SDF3DShapeType::Torus:     range = shapeData.x; break;
        case SDF3DShapeType::Cone: { float angle = atan(shapeData.y); range = std::max(shapeData.y, angle * shapeData.y / sqrt(4.0f * glm::pi<float>() * glm::pi<float>() - angle * angle)); break; }
        case SDF3DShapeType::Capsule:   range = 0.5f * shapeData.x; break;
        }

        range += shapeBlobbing;
        range += float(uint(operationType) >= uint(SDFOperationType::SmoothUnion)) * operationSmoothing;
        range *= 1.0f / std::max(std::max(invRotationScale[0][0], invRotationScale[1][1]), invRotationScale[2][2]);
        return range;
    }

#else
    float calculateRange()
    {
        float range = 0.0f;
        switch (shapeType)
        {
        case SDF3DShapeType::Sphere:    range = shapeData.x; break;
        case SDF3DShapeType::Ellipsoid: range = max(max(shapeData.x, shapeData.y), shapeData.z); break;
        case SDF3DShapeType::Box:       range = max(max(shapeData.x, shapeData.y), shapeData.z); break;
        case SDF3DShapeType::Torus:     range = shapeData.x; break;
        case SDF3DShapeType::Cone: { float angle = atan(shapeData.y); range = max(shapeData.y, angle * shapeData.y / sqrt(4.0f * float(M_PI) * float(M_PI) - angle * angle)); break; }
        case SDF3DShapeType::Capsule:   range = 0.5f * shapeData.x; break;
        }

        range += shapeBlobbing;
        range += float(uint(operationType) >= uint(SDFOperationType::SmoothUnion)) * operationSmoothing;
        range *= 1.0f / max(max(invRotationScale[0][0], invRotationScale[1][1]), invRotationScale[2][2]);
        return range;
    }

    float eval(float3 p, float d)
    {
        float dShape = evalShape(p, shapeType, shapeData, shapeBlobbing, translation, invRotationScale);
        return evalOperation(operationType, d, dShape, operationSmoothing);
    }

    float2 evalInterval(float3 pMin, float3 pMax, float2 d)
    {
        float2 dShape = evalIntervalShape(pMin, pMax, shapeType, shapeData, shapeBlobbing, translation, invRotationScale);
        return evalIntervalOperation(operationType, d, dShape, operationSmoothing);
    }

    static float evalShape(float3 p, SDF3DShapeType shapeType, float3 shapeData, float blobbing, float3 translation, float3x3 invRotationScale)
    {
        p = mul(invRotationScale, p - translation);
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
        pCenter = mul(invRotationScale, pCenter - translation);
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
#endif
};

#ifndef HOST_CODE
interface ISDF3DPrimitive
{
    float eval(float3 p, float d);
    float evalShape(float3 p);

    SDF3DPrimitive format();
};

struct SDF3DSpherePrimitive : ISDF3DPrimitive
{
    float radius;
    float blobbing;
    SDFOperationType operationType;
    float smoothing;
    float3 translation;
    float3x3 invRotationScale;

    __init(float radius, float blobbing, float operationSmoothing, SDFOperationType operationType, float3 translation, float3x3 invRotationScale)
    {
        this.radius = radius;
        this.blobbing = blobbing;
        this.smoothing = operationSmoothing;
        this.operationType = operationType;
        this.translation = translation;
        this.invRotationScale = invRotationScale;
    }

    float eval(float3 p, float d)
    {
        float dShape = evalShape(p);
        return SDF3DPrimitive.evalOperation(operationType, d, dShape, smoothing);
    }

    float evalShape(float3 p)
    {
        p = mul(invRotationScale, p - translation);
        return sdfSphere(p, radius) - blobbing;
    }

    SDF3DPrimitive format()
    {
        SDF3DPrimitive formattedPrimitive;
        formattedPrimitive.shapeType            = SDF3DShapeType::Sphere;
        formattedPrimitive.operationType        = operationType;
        formattedPrimitive.shapeData            = float3(radius, 0.0f, 0.0f);
        formattedPrimitive.shapeBlobbing        = blobbing;
        formattedPrimitive.operationSmoothing   = smoothing;
        formattedPrimitive.translation          = translation;
        formattedPrimitive.invRotationScale     = invRotationScale;
        return formattedPrimitive;
    }
};

struct SDF3DEllipsoidPrimitive : ISDF3DPrimitive
{
    float3 radii;
    float blobbing;
    float smoothing;
    SDFOperationType operationType;
    float3 translation;
    float3x3 invRotationScale;

    __init(float3 radii, float blobbing, float operationSmoothing, SDFOperationType operationType, float3 translation, float3x3 invRotationScale)
    {
        this.radii = radii;
        this.blobbing = blobbing;
        this.smoothing = operationSmoothing;
        this.operationType = operationType;
        this.translation = translation;
        this.invRotationScale = invRotationScale;
    }

    float eval(float3 p, float d)
    {
        float dShape = evalShape(p);
        return SDF3DPrimitive.evalOperation(operationType, d, dShape, smoothing);
    }

    float evalShape(float3 p)
    {
        p = mul(invRotationScale, p - translation);
        return sdfEllipsoid(p, radii) - blobbing;
    }

    SDF3DPrimitive format()
    {
        SDF3DPrimitive formattedPrimitive;
        formattedPrimitive.shapeType            = SDF3DShapeType::Ellipsoid;
        formattedPrimitive.operationType        = operationType;
        formattedPrimitive.shapeData            = radii;
        formattedPrimitive.shapeBlobbing        = blobbing;
        formattedPrimitive.operationSmoothing   = smoothing;
        formattedPrimitive.translation          = translation;
        formattedPrimitive.invRotationScale     = invRotationScale;
        return formattedPrimitive;
    }
};

struct SDF3DBoxPrimitive : ISDF3DPrimitive
{
    float3 halfExtents;
    float blobbing;
    float smoothing;
    SDFOperationType operationType;
    float3 translation;
    float3x3 invRotationScale;

    __init(float3 halfExtents, float blobbing, float operationSmoothing, SDFOperationType operationType, float3 translation, float3x3 invRotationScale)
    {
        this.halfExtents = halfExtents;
        this.blobbing = blobbing;
        this.smoothing = operationSmoothing;
        this.operationType = operationType;
        this.translation = translation;
        this.invRotationScale = invRotationScale;
    }

    float eval(float3 p, float d)
    {
        float dShape = evalShape(p);
        return SDF3DPrimitive.evalOperation(operationType, d, dShape, smoothing);
    }

    float evalShape(float3 p)
    {
        p = mul(invRotationScale, p - translation);
        return sdfBox(p, halfExtents) - blobbing;
    }

    SDF3DPrimitive format()
    {
        SDF3DPrimitive formattedPrimitive;
        formattedPrimitive.shapeType            = SDF3DShapeType::Box;
        formattedPrimitive.operationType        = operationType;
        formattedPrimitive.shapeData            = halfExtents;
        formattedPrimitive.shapeBlobbing        = blobbing;
        formattedPrimitive.operationSmoothing   = smoothing;
        formattedPrimitive.translation          = translation;
        formattedPrimitive.invRotationScale     = invRotationScale;
        return formattedPrimitive;
    }
};

struct SDF3DTorusPrimitive : ISDF3DPrimitive
{
    float radius;
    float blobbing;
    float smoothing;
    SDFOperationType operationType;
    float3 translation;
    float3x3 invRotationScale;

    __init(float radius, float blobbing, float operationSmoothing, SDFOperationType operationType, float3 translation, float3x3 invRotationScale)
    {
        this.radius = radius;
        this.blobbing = blobbing;
        this.smoothing = operationSmoothing;
        this.operationType = operationType;
        this.translation = translation;
        this.invRotationScale = invRotationScale;
    }

    float eval(float3 p, float d)
    {
        float dShape = evalShape(p);
        return SDF3DPrimitive.evalOperation(operationType, d, dShape, smoothing);
    }

    float evalShape(float3 p)
    {
        p = mul(invRotationScale, p - translation);
        return sdfTorus(p, radius) - blobbing;
    }

    SDF3DPrimitive format()
    {
        SDF3DPrimitive formattedPrimitive;
        formattedPrimitive.shapeType            = SDF3DShapeType::Torus;
        formattedPrimitive.operationType        = operationType;
        formattedPrimitive.shapeData            = float3(radius, 0.0f, 0.0f);
        formattedPrimitive.shapeBlobbing        = blobbing;
        formattedPrimitive.operationSmoothing   = smoothing;
        formattedPrimitive.translation          = translation;
        formattedPrimitive.invRotationScale     = invRotationScale;
        return formattedPrimitive;
    }
};

struct SDF3DConePrimitive : ISDF3DPrimitive
{
    float tanAngle;
    float height;
    float blobbing;
    float smoothing;
    SDFOperationType operationType;
    float3 translation;
    float3x3 invRotationScale;

    __init(float angle, float height, float blobbing, float operationSmoothing, SDFOperationType operationType, float3 translation, float3x3 invRotationScale)
    {
        this.tanAngle = tan(angle);
        this.height = height;
        this.blobbing = blobbing;
        this.smoothing = operationSmoothing;
        this.operationType = operationType;
        this.translation = translation;
        this.invRotationScale = invRotationScale;
    }

    float eval(float3 p, float d)
    {
        float dShape = evalShape(p);
        return SDF3DPrimitive.evalOperation(operationType, d, dShape, smoothing);
    }

    float evalShape(float3 p)
    {
        p = mul(invRotationScale, p - translation);
        return sdfCone(p, tanAngle, height) - blobbing;
    }

    SDF3DPrimitive format()
    {
        SDF3DPrimitive formattedPrimitive;
        formattedPrimitive.shapeType            = SDF3DShapeType::Cone;
        formattedPrimitive.operationType        = operationType;
        formattedPrimitive.shapeData            = float3(tanAngle, height, 0.0f);
        formattedPrimitive.shapeBlobbing        = blobbing;
        formattedPrimitive.operationSmoothing   = smoothing;
        formattedPrimitive.translation          = translation;
        formattedPrimitive.invRotationScale     = invRotationScale;
        return formattedPrimitive;
    }
};

struct SDF3DCapsulePrimitive : ISDF3DPrimitive
{
    float halfLength;
    float blobbing;
    float smoothing;
    SDFOperationType operationType;
    float3 translation;
    float3x3 invRotationScale;

    __init(float halfLength, float blobbing, float operationSmoothing, SDFOperationType operationType, float3 translation, float3x3 invRotationScale)
    {
        this.halfLength = halfLength;
        this.blobbing = blobbing;
        this.smoothing = operationSmoothing;
        this.operationType = operationType;
        this.translation = translation;
        this.invRotationScale = invRotationScale;
    }

    float eval(float3 p, float d)
    {
        float dShape = evalShape(p);
        return SDF3DPrimitive.evalOperation(operationType, d, dShape, smoothing);
    }

    float evalShape(float3 p)
    {
        p = mul(invRotationScale, p - translation);
        return sdfCapsule(p, halfLength) - blobbing;
    }

    SDF3DPrimitive format()
    {
        SDF3DPrimitive formattedPrimitive;
        formattedPrimitive.shapeType            = SDF3DShapeType::Capsule;
        formattedPrimitive.operationType        = operationType;
        formattedPrimitive.shapeData            = float3(halfLength, 0.0f, 0.0f);
        formattedPrimitive.shapeBlobbing        = blobbing;
        formattedPrimitive.operationSmoothing   = smoothing;
        formattedPrimitive.translation          = translation;
        formattedPrimitive.invRotationScale     = invRotationScale;
        return formattedPrimitive;
    }
};
#endif

END_NAMESPACE_FALCOR

#ifdef HOST_CODE
#undef max
#undef sqrt
#endif
