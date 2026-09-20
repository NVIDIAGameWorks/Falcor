/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
import Utils.Math.Ray;
import Utils.Geometry.GeometryHelpers;
import StaticParams;
import Params;

/** Types of samplable lights.
*/
enum class LightType
{
    EnvMap,
    Emissive,
    Analytic
};

/** Describes a light sample.
*/
struct LightSample
{
    float3  Li;             ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
    float   pdf;            ///< Pdf with respect to solid angle at the shading point.
    float3  origin;         ///< Ray origin for visibility evaluation (offseted to avoid self-intersection).
    float   distance;       ///< Ray distance for visibility evaluation (shortened to avoid self-intersection).
    float3  dir;            ///< Ray direction for visibility evaluation (normalized).
    uint    lightType;      ///< Light type this sample comes from (LightType casted to uint).
#if USE_RESTIRPT
    uint    triangleId;     ///< For reconstructing hitInfo for sampled emissive lights.
    float2  barycentrics;   ///< For reconstructing hitInfo for sampled emissive lights.
    float   geometryTerm;   ///< For computing jacobian in GRIS.
    float3  normal;         ///< Normal for area (including sphere) lights.
#endif
    Ray getVisibilityRay() { return Ray(origin, dir, 0.f, distance); }
};

/** Describes a path vertex.
*/
struct PathVertex
{
    uint index;         ///< Vertex index (0 = camera, 1 = primary hit, 2 = secondary hit, etc.).
    float3 pos;         ///< Vertex position.
    float3 faceNormal;  ///< Geometry normal at the vertex (zero if not on a surface).
    bool frontFacing;   ///< True if path vertex is on the front-facing side (if on a surface).
    bool isCurveHit;

    __init(uint index, float3 pos, float3 faceNormal = float3(0.f), bool frontFacing = true, bool isCurveHit = false)
    {
        this.index = index;
        this.pos = pos;
        this.faceNormal = faceNormal;
        this.frontFacing = frontFacing;
        this.isCurveHit = isCurveHit;
    }

    float3 getRayOrigin(float3 rayDir)
    {
        return isCurveHit ? pos : computeRayOrigin(pos, dot(faceNormal, rayDir) >= 0 ? faceNormal : -faceNormal);
    }

    float3 getOrientedFaceNormal()
    {
        return frontFacing ? faceNormal : -faceNormal;
    }
};

/** Evaluates the currently configured heuristic for multiple importance sampling (MIS).
    \param[in] n0 Number of samples taken from the first sampling strategy.
    \param[in] p0 Pdf for the first sampling strategy.
    \param[in] n1 Number of samples taken from the second sampling strategy.
    \param[in] p1 Pdf for the second sampling strategy.
    \return Weight for the contribution from the first strategy (p0).
*/
float evalMIS(float n0, float p0, float n1, float p1)
{
    switch (MISHeuristic(kMISHeuristic))
    {
    case MISHeuristic::Balance:
    {
        float q0 = n0 * p0;
        float q1 = n1 * p1;
        return q0 / (q0 + q1);
    }
    case MISHeuristic::PowerTwo:
    {
        float q0 = (n0 * p0) * (n0 * p0);
        float q1 = (n1 * p1) * (n1 * p1);
        return q0 / (q0 + q1);
    }
    case MISHeuristic::PowerExp:
    {
        float q0 = pow(n0 * p0, kMISPowerExponent);
        float q1 = pow(n1 * p1, kMISPowerExponent);
        return q0 / (q0 + q1);
    }
    default:
        return 0.f;
    }
}
