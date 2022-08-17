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
#include "Utils/Math/MathConstants.slangh"

import Scene.Scene;
import Scene.HitInfo;
import Utils.SDF.SDF3DShapes;
import Scene.SDFs.SDF3DPrimitive;
import Utils.Geometry.IntersectionHelpers;
import Scene.RaytracingInline;
import Utils.Math.AABB;
import Utils.Math.MatrixUtils;
import Marker2DSet;
import SDFEditorTypes;


Texture2D gInputColor;                  ///< Input color (2D GUI will be blended on top).
Texture2D<PackedHitInfo> gVBuffer;      ///< Fullscreen V-buffer for the primary hits.
Texture2D<float2> gLinearZ;             ///< Linear Z constructed from GBuffer.

struct GUIPass
{
    uint2 resolution;
    uint2 mousePos;
    Marker2DSet                         markerSet;
    RWStructuredBuffer<SDFPickingInfo>  pickingData;
    StructuredBuffer<SDFEditingData>    editingPrimitiveData;
    Buffer<uint>                        gridInstanceIDs;
    SDFBBRenderSettings                 bbRenderSettings;
    uint                                gridInstanceCount;
    uint                                ui2DActive;
    SDFGridPlane                        gridPlane;
    SDFGridPlane                        symmetryPlane;
};

ParameterBlock<GUIPass> gGUIPass;

bool iBoundingBox(in float3 ro, in float3 rd, in float3 halfExtent, in float e, out float t)
{
    float2 boxT;
    if (!intersectRayAABB(ro, rd, -halfExtent, halfExtent, boxT))
    {
        return false;
    }

    if (boxT.x > 0.0f)
    {
        t = boxT.x;
        float3 p = ro + rd * t;
        float sd = sdfBoundingBox(p, halfExtent, e);

        if (abs(sd) < 0.001f)
            return true;
    }

    t = boxT.y;
    float3 p = ro + rd * t;
    float sd = sdfBoundingBox(p, halfExtent, e);
    return abs(sd) < 0.001f;
}

bool intersectGridPlane(float3 rayOrigin, float3 rayDir, float3 pointOnPlane, float3 normal, float3 rightVec, out float distanceToIntersection, float gridWidth = 0.1f, float gridScale = 1.0f, float gridSize = 1.0f)
{
    distanceToIntersection = -dot(normal, rayOrigin - pointOnPlane) / dot(normal, rayDir);
    if (distanceToIntersection < 0.0f)
    {
        return false;
    }

    float3 upVec = cross(normal, rightVec);
    float3 p = rayOrigin + distanceToIntersection * rayDir;
    float3 diff = p - pointOnPlane;

    float u = dot(diff, rightVec);
    float v = dot(diff, upVec);
    if (abs(u) > gridSize || abs(v) > gridSize)
    {
        return false;
    }
    u = frac(u * gridScale);
    v = frac(v * gridScale);

    return u < gridWidth || v < gridWidth || u > 1.0f - gridWidth || v > 1.0f - gridWidth;
}

void rayToSDFLocalSpace(uint instanceID, inout Ray cameraRay)
{
    GeometryInstanceID geometryInstanceID;
    geometryInstanceID.index = instanceID;
    const GeometryInstanceData instance = gScene.getGeometryInstance(geometryInstanceID);

    // Load Matrices.
    float4x4 worldMat = gScene.loadWorldMatrix(instance.globalMatrixID);
    float3x3 worldInvTransposeMat = gScene.loadInverseTransposeWorldMatrix(instance.globalMatrixID);

    // Convert ray origin and direction to local space.
    // Multiplying v.M to perform the intended (M^T).v
    cameraRay.origin = mul(cameraRay.origin - extractTranslate(worldMat), worldInvTransposeMat);
    cameraRay.dir = mul(cameraRay.dir, worldInvTransposeMat);
}

float4 traceSDFBoundingBox(Ray cameraRay, uint instanceID, float distanceToHitPoint)
{
    rayToSDFLocalSpace(instanceID, cameraRay);

    float t;
    float4 color = float4(0.0f);
    if (iBoundingBox(cameraRay.origin, cameraRay.dir, float3(0.5f), gGUIPass.bbRenderSettings.edgeThickness, t))
    {
        float alpha = t > distanceToHitPoint ? 0.2f : 1.0f;
        color = (gGUIPass.bbRenderSettings.selectedInstanceID == instanceID) ? float4(0.0f, 1.0f, 0.0f, alpha) : float4(1.0f, 0.0f, 0.0f, alpha);
    }

    return color;
}

bool sdfIntersectPrimitive(float3 rayOrigin, float3 rayDir, float3 bbCenter, float3 bbExtent, SDF3DPrimitive primitive, out float t)
{
    float ilen = 1.f / length(rayDir);
    rayDir *= ilen;

    float2 nearFarT;
    if (!intersectRayAABB(rayOrigin - bbCenter, rayDir, -0.5f * bbExtent, 0.5f * bbExtent, nearFarT))
    {
        return false;
    }

    float tmax = nearFarT.y;
    t = max(nearFarT.x, 0.01f);

    float h = primitive.eval(rayOrigin + t * rayDir);

    for (int i = 0; i < 128; i++)
    {
        t += h;
        float3 pos = rayOrigin + t * rayDir;
        h = primitive.eval(pos);

        if (h <= 0.0f)
            break;

        if (t > tmax)
        {
            return false;
        }
    }

    t *= ilen;
    return true;
}

float4 renderGridPlanes(const Ray cameraRayAtPixel, const HitInfo hit, const float distanceToHitPoint, const float4 inColor)
{
    float4 finalColor = inColor;
    if (gGUIPass.gridPlane.active == 1 || gGUIPass.symmetryPlane.active == 1)
    {
        Ray cameraRayMouse = gScene.camera.computeRayPinhole(gGUIPass.mousePos, gGUIPass.resolution, true);

        if (gGUIPass.gridPlane.active == 1)
        {
            float distanceToIntersection;
            bool hitGridPlane = intersectGridPlane(cameraRayAtPixel.origin, cameraRayAtPixel.dir, gGUIPass.gridPlane.position, gGUIPass.gridPlane.normal, gGUIPass.gridPlane.rightVector, distanceToIntersection, gGUIPass.gridPlane.gridLineWidth, gGUIPass.gridPlane.gridScale, gGUIPass.gridPlane.planeSize);

            // Compute distance from the current pixel's position on the grid plane the the position of the plane where the mouse is.
            float distanceToIntersectionAtMouse = gGUIPass.gridPlane.intersect(cameraRayMouse.origin, cameraRayMouse.dir);
            float3 pointOnPlaneAtMouse = cameraRayMouse.origin + cameraRayMouse.dir * distanceToIntersectionAtMouse;
            float3 pointOnPlane = cameraRayAtPixel.origin + cameraRayAtPixel.dir * distanceToIntersection;
            float len = length(pointOnPlaneAtMouse - pointOnPlane);

            // Blend background with grid (if we hit the grid).
            float4 gridPlaneColor = gGUIPass.gridPlane.color;
            const float alphaFactorForHiddenGrid = 0.05;
            const float factorForHiddenBlob = 0.2;
            float alpha = 0.0f;

            if (hit.getType() == HitType::SDFGrid && abs(distanceToIntersection - distanceToHitPoint) < 0.001f)
            {
                alpha = gGUIPass.gridPlane.color.a;
                gridPlaneColor.rgb = gridPlaneColor.bgr;
            }
            else
            {
                alpha = hitGridPlane ? gGUIPass.gridPlane.color.a : 0.0f;

                if (distanceToIntersection > distanceToHitPoint)   // Is the point on the grid plane through the current pixel behind already rendered geometry?
                {
                    alpha = alpha * alphaFactorForHiddenGrid;
                }
            }

            float4 gridColorPlusBackground = (1 - alpha) * finalColor + alpha * gridPlaneColor;

            // Mix with circular blob showing where the mouse is.
            float radius = gGUIPass.gridPlane.planeSize * 0.1f;
            float falloff = (radius - len) / radius;
            float4 circularBlobColor = (len < radius&& gGUIPass.ui2DActive == 0) ? (float4(1.0) - gGUIPass.gridPlane.color) * falloff : float4(0.0);
            finalColor = gridColorPlusBackground + circularBlobColor * (distanceToIntersection > distanceToHitPoint ? factorForHiddenBlob : 1.0);
        }

        if (gGUIPass.symmetryPlane.active == 1)
        {
            float distanceToIntersection;
            bool hitSymmetryPlane = intersectGridPlane(cameraRayAtPixel.origin, cameraRayAtPixel.dir, gGUIPass.symmetryPlane.position, gGUIPass.symmetryPlane.normal, gGUIPass.symmetryPlane.rightVector, distanceToIntersection, gGUIPass.symmetryPlane.gridLineWidth, gGUIPass.symmetryPlane.gridScale, gGUIPass.symmetryPlane.planeSize);

            // Compute distance from the current pixel's position on the grid plane the the position of the plane where the mouse is.
            float distanceToIntersectionAtMouse = gGUIPass.symmetryPlane.intersect(cameraRayMouse.origin, cameraRayMouse.dir);
            float3 pointOnPlaneAtMouse = cameraRayMouse.origin + cameraRayMouse.dir * distanceToIntersectionAtMouse;
            float3 pointOnPlane = cameraRayAtPixel.origin + cameraRayAtPixel.dir * distanceToIntersection;
            float len = length(pointOnPlaneAtMouse - pointOnPlane);

            float4 symmetryPlaneColor = gGUIPass.symmetryPlane.color;

            if (abs(distanceToIntersection - distanceToHitPoint) > 0.005f)
            {
                symmetryPlaneColor.a = 0.0f;
            }

            finalColor.rgb = lerp(finalColor.rgb, symmetryPlaneColor.rgb, symmetryPlaneColor.a);
        }
    }
    return finalColor;
}

float4 psMain(float2 texC : TEXCOORD) : SV_TARGET
{
    const uint2 pixel = uint2(texC * float2(gGUIPass.resolution));
    const float diagonal = length(gGUIPass.resolution);

    float4 finalColor = gInputColor[pixel];

    Ray cameraRay = gScene.camera.computeRayPinhole(pixel, gGUIPass.resolution, false);

    const HitInfo hit = HitInfo(gVBuffer[pixel]);
    GeometryInstanceID instanceID = GeometryInstanceID(UINT32_MAX, 0);
    HitType hitType = HitType::None;
    float distanceToHitPoint = FLT_MAX;
    if (hit.isValid())
    {
        hitType = hit.getType();
        instanceID = hit.getInstanceID();

        // Compute the distance from the origin of the camera to the target position.
        distanceToHitPoint = gLinearZ[pixel].x / dot(normalize(gScene.camera.data.cameraW), cameraRay.dir);
    }

    if (all(gGUIPass.mousePos == pixel))
    {
        gGUIPass.pickingData[0].distance = distanceToHitPoint;
        gGUIPass.pickingData[0].instanceID = instanceID.index;
        gGUIPass.pickingData[0].hitType = hitType;
    }

    if (gGUIPass.bbRenderSettings.renderMode == uint(SDFBBRenderMode::RenderAll))
    {
        float4 sdfColor;

        for (uint gridIndex = 0; gridIndex < gGUIPass.gridInstanceCount; gridIndex++)
        {
            uint instanceID = gGUIPass.gridInstanceIDs[gridIndex];
            sdfColor = traceSDFBoundingBox(cameraRay, instanceID, distanceToHitPoint);
            finalColor.rgb = lerp(finalColor.rgb, sdfColor.rgb, sdfColor.a);
        }
    }
    else if (gGUIPass.bbRenderSettings.renderMode == uint(SDFBBRenderMode::RenderSelectedOnly))
    {
        float4 sdfColor = traceSDFBoundingBox(cameraRay, gGUIPass.bbRenderSettings.selectedInstanceID, distanceToHitPoint);
        finalColor.rgb = lerp(finalColor.rgb, sdfColor.rgb, sdfColor.a);
    }

    finalColor = renderGridPlanes(cameraRay, hit, distanceToHitPoint, finalColor);

    SDFEditingData editingData = gGUIPass.editingPrimitiveData[0];

    if (editingData.previewEnabled == 1 && editingData.editing == 0 && editingData.instanceID != UINT32_MAX)
    {
        rayToSDFLocalSpace(editingData.instanceID, cameraRay);

        const float hitAlpha = 0.5f;
        const float hiddenHitAlpha = 0.1f;

        float t;
        bool hit = false;

        float3 previewColor = float3(1.0f, 0.0f, 0.0f);
        float alpha = 1.0f;

        if (sdfIntersectPrimitive(cameraRay.origin, cameraRay.dir, editingData.primitiveBB.center(), editingData.primitiveBB.extent(), editingData.primitive, t))
        {
            alpha = t > distanceToHitPoint ? hiddenHitAlpha : hitAlpha;
            hit = true;
        }

        if (gGUIPass.symmetryPlane.active == 1)
        {
            if (sdfIntersectPrimitive(cameraRay.origin, cameraRay.dir, editingData.symmetryPrimitiveBB.center(), editingData.symmetryPrimitiveBB.extent(), editingData.symmetryPrimitive, t))
            {
                if (t <= distanceToHitPoint)
                {
                    alpha = hitAlpha;
                }
                else if (alpha < hitAlpha || alpha == 1.0f)
                {
                    alpha = hiddenHitAlpha;
                }

                hit = true;
            }
        }

        if (editingData.scalingAxis < uint(SDFEditorAxis::Count))
        {
            float3 rayOriginPrimLocal = mul(editingData.primitive.invRotationScale, (cameraRay.origin - editingData.primitive.translation)).xyz;
            float3 rayDirPrimLocal = normalize(mul(editingData.primitive.invRotationScale, cameraRay.dir).xyz);

            float axisPreviewT = 0.0f;
            float2 nearFar;
            switch (editingData.scalingAxis)
            {
            case uint(SDFEditorAxis::X):
            {
                const float3 dimensions = float3(editingData.primitive.shapeData.x, 0.001f, 0.001f);
                intersectRayAABB(rayOriginPrimLocal, rayDirPrimLocal, -0.5f * dimensions, 0.5f * dimensions, nearFar);
                axisPreviewT = nearFar.x;
                break;
            }
            case uint(SDFEditorAxis::Y):
            {
                const float3 dimensions = float3(0.001f, editingData.primitive.shapeData.y, 0.001f);
                intersectRayAABB(rayOriginPrimLocal, rayDirPrimLocal, -0.5f * dimensions, 0.5f * dimensions, nearFar);
                axisPreviewT = nearFar.x;
                break;
            }
            case uint(SDFEditorAxis::Z):
            {
                const float3 dimensions = float3(0.001f, 0.001f, editingData.primitive.shapeData.z);
                intersectRayAABB(rayOriginPrimLocal, rayDirPrimLocal, -0.5f * dimensions, 0.5f * dimensions, nearFar);
                axisPreviewT = nearFar.x;
                break;
            }
            case uint(SDFEditorAxis::All):
            {
                previewColor = float3(0.0f, 1.0f, 0.0f);
                break;
            }
            }
        }

        if (hit)
        {
            finalColor.rgb = lerp(finalColor.rgb, previewColor, alpha);
        }
    }

    // Render all the 2D GUI markers.
    finalColor = gGUIPass.markerSet.render(pixel, finalColor);

    return finalColor;
}
