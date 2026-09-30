#include "Utils/Math/MathConstants.slangh"

import Utils.Sampling.SampleGenerator;

// Struct for a photon differential 64B.
// Could be packed tighter
struct PhotonDifferential {
    float3 flux;     // Incoming flux
    uint pathLength; // Length of the photon path
    float3 dir;      // Incoming direction
    float normalX;
    float3 dPdx; // Positional x derivative
    float normalY;
    float3 dPdy; // Positional y derivative
    float normalZ;

    __init() {
        this.dir = float3(0);
        normalX = 0.0f;
        normalY = 1.0f;
        normalZ = 0.0f;
        this.flux = float3(0);
        this.pathLength = 0;
        this.dPdx = float3(0);
        this.dPdy = float3(0);
    }

    // Normal helper. Face normal of the photon surface
    property float3 normalW {
        get { return float3(normalX, normalY, normalZ); }
        set {
            normalX = newValue.x;
            normalY = newValue.y;
            normalZ = newValue.z;
        }
    }

    // Helper to check if this is a caustic photon (has differential data)
    // If sqDistance > 0.0f then Distance > 0.0f
    bool isCaustic() { return dot(dPdx, dPdx) > 0.0f || dot(dPdy, dPdy) > 0.0f; }
}

//
// Tracing Ray Differentials by Homan Igehy
//

/*  Update ray differentrial transfer
    \param[in] Current normal
    \param[in] Positional derivative
    \param[in] Directional derivative
    \param[in] World ray direction
    \param[in] Ray T current
*/
void updateTransferRayDifferential(float3 N, inout float3 dPdx, float3 dDdx, float3 D, float t)
{
    float dtdx = -1.0f * dot(dPdx + t * dDdx, N) / dot(D, N);
    dPdx = dPdx + t * dDdx + dtdx * D;
    // dDdx remains the same
}

/*  Update ray differentrial reflection
    \param[in] Current normal
    \param[in] Normal derivative
    \param[in] Directional derivative
    \param[in] World ray direction
*/
void updateReflectRayDifferential(float3 N, float3 dNdx, inout float3 dDdx, float3 D)
{
    // dPdx remains the same
    N = normalize(N);
    float dDNdx = dot(dDdx, N) + dot(D, dNdx);
    dDdx = dDdx - 2 * (dot(D, N) * dNdx + dDNdx * N);
}

/*  Update ray differentrial refraction
    \param[in] Direction
    \param[in] Refacted ray
    \param[in] Normal
    \param[in] Ratio of the incident index of refraction to thetransmitted index of refraction
    \param[in] Normal derivative
    \param[in] Directional derivative
*/
void updateRefractRayDifferential(float3 D, float3 R, float3 N, float eta, float3 dNdx, inout float3 dDdx)
{
    // dPdx remains the same
    N = normalize(N);
    float DN = dot(D, N);
    float RN = dot(R, N);

    // To prevent total internal reflection. But I'm not really sure if it works or not.
    // Should check the AAPS stuff (for util functions)
    //if (RN <= 0.0001f) return;

    float mu = eta * DN - RN;
    float dDNdx = dot(dDdx, N) + dot(D, dNdx);
    float dmudx = (eta - eta * eta * DN / RN) * dDNdx;
    dDdx = eta * dDdx - (mu * dNdx + dmudx * N);
}

// Igehy's Normal Interpolated Triangles
void calculateDNdx(float3 dP1, float3 dP2, float3 dN1, float3 dN2, float3 N, float3 dPdx, out float3 dNdx)
{
    float P11 = dot(dP1, dP1);
    float P12 = dot(dP1, dP2);
    float P22 = dot(dP2, dP2);

    float Q1 = dot(dP1, dPdx);
    float Q2 = dot(dP2, dPdx);

    float delta = P11 * P22 - P12 * P12;
    if (abs(delta) < 1e-8) // Gemini suggested
    {
        dNdx = float3(0.0f);
        return;
    }

    float dudx = (Q1 * P22 - Q2 * P12) / delta;
    float dvdx = (Q2 * P11 - Q1 * P12) / delta;

    float n2 = dot(N, N);
    float n1 = sqrt(n2);
    dNdx = dudx * dN1 + dvdx * dN2;
    dNdx = (n2 * dNdx - dot(N, dNdx) * N) / (n2 * n1);
}

/*
    Photon differentials approach
    Evaluates if a shading point falls inside the photon's differential footprint.
    Returns a kernel weight (0.0 if outside, > 0.0 if inside) and outputs the footprint area.

    \param[in] surface position
    \param[in] photon's position (photon's aabb center or position from reservoir)
    \param[in] photon's normal
    \param[in] photon's positional x derivative 
    \param[in] photon's positional y derivative 
    \param[in/out] footprint area
*/
float evaluateDifferentialWeight(float3 surfacePos, float3 photonPosW, float3 photonNormalW, float3 dPdx, float3 dPdy, out float footprintArea)
{
    float3 delta = surfacePos - photonPosW;

    footprintArea = length(cross(dPdx, dPdy));
    // Safe footprint area that prevents zero area
    //footprintArea = (dot(dPdx, dPdx) + dot(dPdy, dPdy)) * 0.5f;

    // To prevent large footprints
    if (footprintArea < 1e-6f) return 0.0f;
    //footprintArea = max(1e-9f, footprintArea);

    // Project delta onto the photon's surface plane
    float distToPlane = dot(delta, photonNormalW);
    // Cull the point if it is too far from photon's plane
    if (abs(distToPlane) > 0.01f) return 0.0f;
    float3 vPlane = delta - distToPlane * photonNormalW;

    // Dual basis stuff from PhotonDifferential paper
    // Use the inverse basis to find local coordinates (s, t)
    // This transforms the stretched parallelogram into a unit circle context
    float3 invX = cross(dPdy, photonNormalW) / footprintArea;
    float3 invY = cross(photonNormalW, dPdx) / footprintArea;

    float s = dot(vPlane, invX);
    float t = dot(vPlane, invY);

    // Check if the point falls inside the footprint boundary
    float d2 = s * s + t * t;
    if (d2 > 1.0f) return 0.0f;

    // See Epanechnikov Kernel: 2/pi * (1 - d2)
    return M_2_PI * (1.0f - d2);
}

// REAL-TIME RAY TRACED CAUSTICS by Xueqing Yang and Yaobin Ouyang NVIDIA
//float getArea(float3 dPdx, float3 dPdy)
//{
//    float area;
//    if (gAreaType == 0)
//    {
//        area = (dot(dPdx, dPdx) + dot(dPdy, dPdy)) * 0.5;
//    }
//    else if (gAreaType == 1)
//    {
//        area = (length(dPdx) + length(dPdy));
//        area *= area;
//    }
//    else if (gAreaType == 2)
//    {
//        area = max(dot(dPdx, dPdx), dot(dPdy, dPdy));
//    }
//    else
//    {
//        area = length(cross(dPdx, dPdy));
//    }
//    return area;
//}
