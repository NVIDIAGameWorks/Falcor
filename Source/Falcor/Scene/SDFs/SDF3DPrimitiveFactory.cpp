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
#include "SDF3DPrimitiveFactory.h"

using namespace Falcor;

SDF3DPrimitive SDF3DPrimitiveFactory::initCommon(SDF3DShapeType shapeType, const float3& shapeData, float blobbing, float operationSmoothing, SDFOperationType operationType, const Transform& transform)
{
    SDF3DPrimitive primitive = {};
    primitive.shapeType = shapeType;
    primitive.shapeData = shapeData;
    primitive.operationType = operationType;
    primitive.shapeBlobbing = blobbing;
    primitive.operationSmoothing = operationSmoothing;
    primitive.translation = transform.getTranslation();
    primitive.invRotationScale = rmcv::inverse(rmcv::mat3(transform.getMatrix()));
    return primitive;
}

AABB SDF3DPrimitiveFactory::computeAABB(const SDF3DPrimitive& primitive)
{
    AABB aabb;
    float rounding = primitive.shapeBlobbing + float(uint(primitive.operationType) >= uint(SDFOperationType::SmoothUnion)) * primitive.operationSmoothing;

    switch (primitive.shapeType)
    {
    case SDF3DShapeType::Sphere:
    {
        float radius = primitive.shapeData.x + rounding;
        aabb.include(float3(radius, radius, radius));
        aabb.include(float3(-radius, radius, radius));
        aabb.include(float3(radius, -radius, radius));
        aabb.include(float3(radius, radius, -radius));
        aabb.include(float3(-radius, -radius, radius));
        aabb.include(float3(radius, -radius, -radius));
        aabb.include(float3(-radius, radius, -radius));
        aabb.include(float3(-radius, -radius, -radius));
    }
    break;
    case  SDF3DShapeType::Ellipsoid:
    case  SDF3DShapeType::Box:
    {
        float3 halfExtents = primitive.shapeData.xyz + float3(rounding);
        aabb.include(float3(halfExtents.x, halfExtents.y, halfExtents.z));
        aabb.include(float3(-halfExtents.x, halfExtents.y, halfExtents.z));
        aabb.include(float3(halfExtents.x, -halfExtents.y, halfExtents.z));
        aabb.include(float3(halfExtents.x, halfExtents.y, -halfExtents.z));
        aabb.include(float3(-halfExtents.x, -halfExtents.y, halfExtents.z));
        aabb.include(float3(halfExtents.x, -halfExtents.y, -halfExtents.z));
        aabb.include(float3(-halfExtents.x, halfExtents.y, -halfExtents.z));
        aabb.include(float3(-halfExtents.x, -halfExtents.y, -halfExtents.z));
    }
    break;
    case  SDF3DShapeType::Torus:
    {
        float smallRadius = rounding;
        float bigRadius = primitive.shapeData.x + rounding;
        aabb.include(float3(bigRadius, smallRadius, bigRadius));
        aabb.include(float3(-bigRadius, smallRadius, bigRadius));
        aabb.include(float3(bigRadius, -smallRadius, bigRadius));
        aabb.include(float3(bigRadius, smallRadius, -bigRadius));
        aabb.include(float3(-bigRadius, -smallRadius, bigRadius));
        aabb.include(float3(bigRadius, -smallRadius, -bigRadius));
        aabb.include(float3(-bigRadius, smallRadius, -bigRadius));
        aabb.include(float3(-bigRadius, -smallRadius, -bigRadius));
    }
    break;
    case  SDF3DShapeType::Cone:
    {
        float tanAngle = primitive.shapeData.x;
        float height = primitive.shapeData.y;
        float radius = tanAngle * height + rounding;
        height += rounding;
        aabb.include(float3(radius, height, radius));
        aabb.include(float3(-radius, height, radius));
        aabb.include(float3(radius, -rounding, radius));
        aabb.include(float3(radius, height, -radius));
        aabb.include(float3(-radius, -rounding, radius));
        aabb.include(float3(radius, -rounding, -radius));
        aabb.include(float3(-radius, height, -radius));
        aabb.include(float3(-radius, -rounding, -radius));
    }
    break;
    case  SDF3DShapeType::Capsule:
    {
        float halfLen = primitive.shapeData.x + rounding;
        aabb.include(float3(rounding, halfLen, rounding));
        aabb.include(float3(-rounding, halfLen, rounding));
        aabb.include(float3(rounding, -halfLen, rounding));
        aabb.include(float3(rounding, halfLen, -rounding));
        aabb.include(float3(-rounding, -halfLen, rounding));
        aabb.include(float3(rounding, -halfLen, -rounding));
        aabb.include(float3(-rounding, halfLen, -rounding));
        aabb.include(float3(-rounding, -halfLen, -rounding));
    }
    break;
    default:
        throw RuntimeError("SDF Primitive has unknown primitive type");
    }

    rmcv::mat4 translate = rmcv::translate(rmcv::identity<rmcv::mat4>(), primitive.translation);
    rmcv::mat4 rotScale = rmcv::inverse(rmcv::transpose(primitive.invRotationScale));
    return aabb.transform(translate * rotScale);
}
