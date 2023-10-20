/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "Utils/Math/MathConstants.slangh"

import Scene.RaytracingInline;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Debug.PixelDebug;
import Rendering.Lights.LightHelpers;
import DiffRendering.DiffSceneIO;

struct RayAD : IDifferentiable
{
    float3 origin;
    float3 direction;
    uint2 pixel;

    [Differentiable]
    [PreferRecompute]
    __init(float3 origin, float3 direction, uint2 pixel)
    {
        this.origin = origin;
        this.direction = direction;
        this.pixel = pixel;
    }
};

// Struct holding info about the differentiable ray-triangle intersection.
struct IntersectionAD : IDifferentiable
{
    GeometryInstanceID instanceID; // Instance ID of the mesh instance that contains the hit triangle.
    uint triangleID;               // Triangle ID of the hit triangle.
    float3 barycentrics;           // Barycentrics of the hit point
    float3 normalW;                // World normal of the hit point
    float3 posW;                   // World position of the hit point
    float hitT;                    // Distance from ray origin to hit point

    [Differentiable]
    [PreferRecompute]
    __init()
    {
        this.instanceID = GeometryInstanceID(0, 0);
        this.triangleID = 0;
        this.barycentrics = float3(0.f);
        this.normalW = float3(0.f);
        this.posW = float3(0.f);
        this.hitT = 0.f;
    }
};

struct SceneQueryAD : IDifferentiable
{
    DiffSceneIO diffSceneIO;
    SceneGradientInfo gradInfo;

    [Differentiable]
    __init(DiffSceneIO _diffSceneIO, SceneGradientInfo _gradInfo)
    {
        diffSceneIO = _diffSceneIO;
        gradInfo = _gradInfo;
    }

    enum DiffIntersectionMode
    {
        AttachToRay = 0,
        AttachToGeometry = 1,
    };

    // Differentiable `traceRayInline` function.
    [Differentiable]
    [PreferRecompute]
    bool traceRayInlineAD(const RayAD rayAD, out IntersectionAD isect, DiffIntersectionMode mode)
    {
        Ray ray = Ray(detach(rayAD.origin), detach(rayAD.direction), 0.f, FLT_MAX);

        float hitT = 0.f;
        SceneRayQuery<0> sceneRayQuery;

        HitInfo hitInfo;
        bool isHit = sceneRayQuery.traceRay(ray, hitInfo, hitT, RAY_FLAG_NONE, 0xff);

        if (!isHit)
        {
            isect = IntersectionAD();
            return false;
        }

        // Convert a non-differentiable HitInfo to a differentiable IntersectionAD.
        // This method recomputes barycentrics and normals using differentiable functions.
        isect = computeIntersectionAD(hitInfo, rayAD, mode);

        return true;
    }

    bool traceVisibilityRay(const Ray ray)
    {
        SceneRayQuery<0> sceneRayQuery;
        return sceneRayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff);
    }

    // Compute differentiable intersection from non-differentiable HitInfo.
    [Differentiable]
    [PreferRecompute]
    IntersectionAD computeIntersectionAD(const HitInfo hitInfo, const RayAD ray, DiffIntersectionMode mode)
    {
        TriangleHit hit = hitInfo.getTriangleHit();

        IntersectionAD isect = IntersectionAD();

        isect.triangleID = hit.primitiveIndex;
        isect.instanceID = hit.instanceID;

        if (mode == DiffIntersectionMode.AttachToRay)
        {
            float3 barycentrics;
            float t;

            isect.posW = loadVertexPosW(hit.instanceID, hit.primitiveIndex, ray, t, barycentrics, isect.normalW);

            isect.hitT = t;
            isect.barycentrics = barycentrics;
        }
        else if (mode == DiffIntersectionMode.AttachToGeometry)
        {
            // Attached to geometry

            // Fix barycentrics.
            isect.barycentrics = no_diff(hit.getBarycentricWeights());
            isect.posW = loadVertexPosW(hit.instanceID, hit.primitiveIndex, isect.barycentrics, isect.normalW);

            // Compute hit T from posW and ray origin.
            isect.hitT = length(isect.posW - ray.origin);
        }

        // Note: We want to pass to vertex positions through shading normals. It hasn't been implemented yet.
        // TODO: Handle shading (vertex) normals correctly.
        isect.normalW = computeShadingNormal(isect.instanceID, isect.triangleID, isect.barycentrics);

        return isect;
    }

    [Differentiable]
    [PreferRecompute]
    float3 computeShadingNormal(GeometryInstanceID instanceID, uint primitiveIndex, float3 barycentrics)
    {
        float3 n[3];
        diffSceneIO.loadVertexNormalsW(gradInfo, instanceID, primitiveIndex, n);

        float3 normal = n[0] * barycentrics.x + n[1] * barycentrics.y + n[2] * barycentrics.z;

        return normal;
    }

    // Compute vertex position in world space using fixed barycentrics.
    [Differentiable]
    [PreferRecompute]
    float3 loadVertexPosW(GeometryInstanceID instanceID, uint triangleID, no_diff float3 barycentrics, out float3 faceNormalW)
    {
        float3 v[3];
        diffSceneIO.loadVertexPositionsW(gradInfo, instanceID, triangleID, v);

        // Compute world position of hit point
        float3 posW = v[0] * barycentrics.x + v[1] * barycentrics.y + v[2] * barycentrics.z;
        faceNormalW = normalize(cross(v[1] - v[0], v[2] - v[0]));

        return posW;
    }

    // Compute vertex position and barycentrics in world space using a differentiable ray.
    [Differentiable]
    [PreferRecompute]
    float3 loadVertexPosW(
        GeometryInstanceID instanceID,
        uint triangleID,
        RayAD ray,
        out float t,
        out float3 barycentrics,
        out float3 faceNormalW
    )
    {
        float3 v[3];
        diffSceneIO.loadVertexPositionsW(gradInfo, instanceID, triangleID, v);

        t = computeRayDistanceToTriangle(v, ray.origin, ray.direction);

        float3 p = ray.origin + ray.direction * t;
        computeBarycentrics(v, p, barycentrics);

        // Compute world position of hit point
        float3 posW = v[0] * barycentrics.x + v[1] * barycentrics.y + v[2] * barycentrics.z;
        faceNormalW = normalize(cross(v[1] - v[0], v[2] - v[0]));

        return posW;
    }

    [Differentiable]
    [PreferRecompute]
    float3 loadCameraPosition() { return diffSceneIO.loadCameraPositionW(gradInfo); }

    [Differentiable]
    [PreferRecompute]
    float2 computeCameraRayScreenPos(float3 direction, float2 frameDim)
    {
        // Compute the normalized ray direction assuming a pinhole camera.
        float3 rayDirN = normalizeSafe(direction);

        // TODO: Use differentiable loaders for the camera to propagate
        // derivatives through the camera properly.
        var camera = gScene.camera;
        float3 cameraW = camera.data.cameraW;
        float3 cameraU = camera.data.cameraU;
        float3 cameraV = camera.data.cameraV;

        float3 rayDirFocalPlane = rayDirN * dot(cameraW, cameraW) / dot(cameraW, rayDirN);

        // Project `rayDirFocalPlane` onto plane by removing `cameraW` component.
        float3 rayDirNc = rayDirFocalPlane - cameraW;
        float2 ndc = float2(dot(rayDirNc, cameraU), dot(rayDirNc, cameraV)) / float2(dot(cameraU, cameraU), dot(cameraV, cameraV));

        float2 p = (ndc + float2(1, -1)) / float2(2, -2);
        float2 screenPos = p * frameDim - float2(0.5, 0.5);
        return screenPos;
    }

    [Differentiable]
    [PreferRecompute]
    float3 computeCameraRayDirection(float2 posScreen, float2 frameDim)
    {
        float2 p = posScreen / float2(frameDim);
        float2 ndc = float2(2, -2) * p + float2(-1, 1);

        // TODO: Use differentiable loaders for the camera frame to propagate gradients through the camera parameters properly.
        var camera = gScene.camera;

        return normalizeSafe(ndc.x * camera.data.cameraU + ndc.y * camera.data.cameraV + camera.data.cameraW);
    }
}
