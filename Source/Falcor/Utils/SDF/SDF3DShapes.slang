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
import Utils.Math.IntervalArithmetic;

/* Functions for computing the signed distance function (SDF) for several different primitives,
* including sphere, ellipsoid, box, torus, cone, capsule, and bounding box.
* For each primitives, there is a "standard" SDF function, e.g, sdfSphere() that returns
* a scalar distance, but also an a version called sdfIntervalSphere() which is the same
* except all computations are evaluated using interval arithmetic, which means it returns
* an interval (float2).
*/

/** Computes signed distance to a sphere centered at the origin.
    \param[in] p The point at which the distance is to be computed.
    \param[in] r The radius of the sphere.
    \return The distance from p to the primitive.
*/
float sdfSphere(float3 p, float r)
{
    return length(p) - r;
}

float2 sdfIntervalSphere(float3 pMin, float3 pMax, float r)
{
    float2 xInterval = float2(pMin.x, pMax.x);
    float2 yInterval = float2(pMin.y, pMax.y);
    float2 zInterval = float2(pMin.z, pMax.z);

    return ivlSub(ivlLength(xInterval, yInterval, zInterval), r);
}

/** Computes signed distance to a ellipsoid centered at the origin.
    \param[in] p The point at which the distance is to be computed.
    \param[in] r The radii (one per xyz) of the ellipsoid.
    \return The distance from p to the primitive.
*/
float sdfEllipsoid(float3 p, float3 r)
{
    float k0 = length(p / r);
    float k1 = length(p / (r * r));
    return k0 * (k0 - 1.0f) / k1;
}

float2 sdfIntervalEllipsoid(float3 pMin, float3 pMax, float3 r)
{
    float2 xInterval = float2(pMin.x, pMax.x);
    float2 yInterval = float2(pMin.y, pMax.y);
    float2 zInterval = float2(pMin.z, pMax.z);

    float3 rSqrd = r * r;
    float2 k0 = ivlLength(ivlDiv(xInterval, r.x), ivlDiv(yInterval, r.y), ivlDiv(zInterval, r.z));
    float2 k1 = ivlLength(ivlDiv(xInterval, rSqrd.x), ivlDiv(yInterval, rSqrd.y), ivlDiv(zInterval, rSqrd.z));
    return ivlDiv(ivlPosMul(k0, ivlSub(k0, 1.0f)), k1);
}

/** Computes signed distance to a box centered at the origin.
    \param[in] p The point at which the distance is to be computed.
    \param[in] b The halflengths of the box.
    \return The distance from p to the primitive.
*/
float sdfBox(float3 p, float3 b)
{
    float3 q = abs(p) - b;
    return length(max(q, 0.0f)) + min(max(max(q.x, q.y), q.z), 0.0f);
}

float2 sdfIntervalBox(float3 pMin, float3 pMax, float3 b)
{
    float2 xAbsIntervalSubB = ivlSub(ivlAbs(float2(pMin.x, pMax.x)), b.x);
    float2 yAbsIntervalSubB = ivlSub(ivlAbs(float2(pMin.y, pMax.y)), b.y);
    float2 zAbsIntervalSubB = ivlSub(ivlAbs(float2(pMin.z, pMax.z)), b.z);

    return ivlAdd(ivlLength(ivlMax(xAbsIntervalSubB, 0.0f), ivlMax(yAbsIntervalSubB, 0.0f), ivlMax(zAbsIntervalSubB, 0.0f)),
            ivlMin(ivlMax(ivlMax(xAbsIntervalSubB, yAbsIntervalSubB), zAbsIntervalSubB), 0.0f));
}

/** Computes signed distance to a torus centered at the origin.
    \param[in] p The point at which the distance is to be computed.
    \param[in] r The radius from the origin to the center of the "tube". Note that the tube has zero radius here. This can be added on top using shapeBlobbing.
    \return The distance from p to the primitive.
*/
float sdfTorus(float3 p, float r)
{
    return length(float2(length(p.xz) - r, p.y));
}

float2 sdfIntervalTorus(float3 pMin, float3 pMax, float r)
{
    float2 xInterval = float2(pMin.x, pMax.x);
    float2 yInterval = float2(pMin.y, pMax.y);
    float2 zInterval = float2(pMin.z, pMax.z);

    return ivlLength(ivlSub(ivlLength(xInterval, zInterval), r), yInterval);
}

/** Computes signed distance to a cone centered at the origin, with its apex point at (0,h,0) and base in the xz-plane (y=0).
    \param[in] p The point at which the distance is to be computed.
    \param[in] tan This is tan of the half angle at the apex. This means that the radius at the base is tan * h.
    \param[in] h The height of the cone.
    \return The distance from p to the primitive.
*/
float sdfCone(float3 p, float tan, float h)
{
    float2 q = h * float2(tan, -1.0f);
    float2 w = float2(length(p.xz), p.y - 0.5f * h);
    float2 a = w - q * saturate(dot(w, q) / dot(q, q));
    float2 b = w - q * float2(saturate(w.x / q.x), 1.0f);
    float k = sign(q.y);
    float d = min(dot(a, a), dot(b, b));
    float s = max(k * (w.x * q.y - w.y * q.x), k * (w.y - q.y));
    return sqrt(d) * sign(s);
}

float2 sdfIntervalCone(float3 pMin, float3 pMax, float tan, float h)
{
    float2 xInterval = float2(pMin.x, pMax.x);
    float2 yInterval = float2(pMin.y, pMax.y) - 0.5f * h;
    float2 zInterval = float2(pMin.z, pMax.z);

    float2 q = h * float2(tan, -1.0);
    float2 wX = ivlLength(xInterval, zInterval);

    float dotQQ = dot(q, q);
    float2 dotWQ = ivlAdd(ivlMul(wX, q.x), ivlMul(yInterval, q.y));
    float2 satDotWQDivDotQQ = ivlSaturate(ivlDiv(dotWQ, dotQQ));
    float2 satWxDivQx = ivlSaturate(ivlDiv(wX, q.x));

    float2 aX = ivlSub(wX, ivlMul(satDotWQDivDotQQ, q.x));
    float2 aY = ivlSub(yInterval, ivlMul(satDotWQDivDotQQ, q.y));

    float2 bX = ivlSub(wX, ivlMul(satWxDivQx, q.x));
    float2 bY = ivlSub(yInterval, q.y);

    float k = sign(q.y);

    float2 d = ivlMin(ivlAdd(ivlSquare(aX), ivlSquare(aY)), ivlAdd(ivlSquare(bX), ivlSquare(bY)));
    float2 s = ivlMax(ivlMul(ivlSub(ivlMul(wX, q.y), ivlMul(yInterval, q.x)), k), ivlMul(ivlSub(yInterval, q.y), k));
    return ivlMul(ivlSqrt(d), sign(s));
}

/** Computes signed distance to a capsule centered at the origin. Located at (0, -hl, 0) to (0, +hl, 0).
    \param[in] p The point at which the distance is to be computed. Note that the capsule has zero radius here. This is added on top using shapeBlobbing.
    \param[in] hl The half length of the capsule..
    \return The distance from p to the primitive.
*/
float sdfCapsule(float3 p, float hl)
{
    p.y -= clamp(p.y, -hl, hl);
    return length(p);
}

float2 sdfIntervalCapsule(float3 pMin, float3 pMax, float hl)
{
    float2 xInterval = float2(pMin.x, pMax.x);
    float2 yInterval = float2(pMin.y, pMax.y);
    float2 zInterval = float2(pMin.z, pMax.z);

    yInterval = ivlSub(yInterval, ivlClamp(yInterval, -hl, hl));
    return ivlLength(xInterval, yInterval, zInterval);
}

/** Computes signed distance to the edges of a bounding centered.
    \param[in] p The point at which the distance is to be computed.
    \param[in] halfBoxSides The half box sides.
    \param[in] edgeThickness The thickness of the edges.
    \return The distance from p to the primitive.
*/
float sdfBoundingBox(float3 point, float3 halfBoxSides, float edgeThickness)
{
    float3 p = abs(point) - halfBoxSides;
    float3 q = abs(p + edgeThickness) - edgeThickness;
    return min(min(
        length(max(float3(p.x, q.y, q.z), 0.0f)) + min(max(p.x, max(q.y, q.z)), 0.0f),
        length(max(float3(q.x, p.y, q.z), 0.0f)) + min(max(q.x, max(p.y, q.z)), 0.0f)),
        length(max(float3(q.x, q.y, p.z), 0.0f)) + min(max(q.x, max(q.y, p.z)), 0.0f));
}

