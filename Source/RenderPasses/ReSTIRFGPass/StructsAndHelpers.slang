#include "Utils/Math/MathConstants.slangh"

import Scene.RaytracingInline;
import Utils.Sampling.SampleGenerator;
import Utils.Color.ColorHelpers;
import Rendering.Materials.IsotropicGGX;
import Rendering.Materials.Fresnel;

static const float kShadowRayEpsilon = 0.98f;

//Struct for a photon. Could be packed tighter
struct PhotonData {
    float3 flux;        //Incoming flux
    uint pathLenght;    //Length of the photon path
    float3 dir;         //Incoming direction
    uint _pad2;
    float3 normalW;     //Face normal of the photon surface
    uint _pad3;

    __init() {
        this.dir = float3(0);
        this.normalW = float3(0,1,0);
        this.flux = float3(0);
        this.pathLenght = 0;
        this._pad2 = 0;
        this._pad3 = 0;
    }
}

enum class PhotonType // : uint32_t
{
    Global = 0,
    Caustic = 1
};

//Data for the FG sample 64B
struct FGSampleData {
    float3 posW;        //Position
    float normalX;
    float3 radianceOut; //Radiance
    float normalY;
    float3 sampleView; //Incoming View
    float normalZ;

    __init() {
        posW = float3(0);
        normalX = 0;
        normalY = 1;
        normalZ = 0;
        radianceOut = float3(0);
        sampleView = float3(0);
    }

    //Normal helper
    property float3 normal {
        get { return float3(normalX, normalY, normalZ); }
        set {
            normalX = newValue.x;
            normalY = newValue.y;
            normalZ = newValue.z;
        }
    }
}

//Reservoir for a Final Gather Sample 112B
struct ReservoirFG {
    PackedHitInfo sampleSurface;    // HitSample 16B
    float3 thp;                     // Throughput used at the end
    float targetFunction;           // Store target function
    FGSampleData sampleData;        // Data of the sample 64B
    float wSum;                     // W and wSum
    uint c;                         // Confidence weight
    float jacobianDeterminant;      // Jacobian determinant to the current sample
    uint pathLength;                // Path Length

    __init() {
        sampleSurface = {};
        thp = float3(1);
        targetFunction = 0;
        sampleData = FGSampleData();
        wSum = 0;
        c = 1;
        pathLength = 0;
        jacobianDeterminant = -1.0;
    }

    //Valid check
    bool isValid() {
        return jacobianDeterminant >= 0;
    }
}

// Reservoir for a CausticSample 112B
struct ReservoirCaustic {
    PackedHitInfo sampleSurface;    // HitSample 16B
    float3 thp;                     // Throughput used at the end
    float targetFunction;           // Store target function
    float3 sampleSurfaceView;       // Incoming Camera View
    float photonDirX;               // (Incoming) photon direction (x)
    float3 photonPosW;              // Position
    float photonDirY;               // (Incoming) photon direction (y)
    float3 photonRadiance;          // Radiance
    float photonDirZ;               // (Incoming) photon direction (z)
    float wSum;                     // W and wSum
    uint c;                         // Confidence weight
    uint pathLength;                // Path Length
    uint _pad;

    __init() {
        sampleSurface = {};
        thp = float3(1);
        targetFunction = 0;
        sampleSurfaceView = float3(0);
        photonDirX = 0;
        photonPosW = float3(0); 
        photonDirY = 0;
        photonRadiance = float3(0);
        photonDirZ = 0;
        wSum = 0;
        c = 0;
        pathLength = 0;
        _pad = 0;
    }

    // Photon direction helper
    property float3 photonDir {
        get { return float3(photonDirX, photonDirY, photonDirZ); }
        set {
            photonDirX = newValue.x;
            photonDirY = newValue.y;
            photonDirZ = newValue.z;
        }
    }

    // Valid check
    bool isValid() {
        return any(photonRadiance > 0);
    }
}

/*  Load shading data from hit and direction
    \param[in] Triangle Hit
    \param[in] incoming world direction (ray direction)
    \param[in] Level of Detail sample mode (LOD 0 for all casess)
    \return Shading data of the triangle hit
*/
ShadingData loadShadingData(const HitInfo hit, const float3 rayDir, const ITextureSampler lod)
{
    const TriangleHit triangleHit = hit.getTriangleHit();
    VertexData v = gScene.getVertexData(triangleHit);
    uint materialID = gScene.getMaterialID(triangleHit.instanceID);
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir/*, lod*/);
    
    return sd;
}

/*  2D pixel index to 1D reservoir index
    \param[in] Pixel coordinate
    \param[in] Pixel resolution on X axis
    \return Linear index
*/
uint index2Dto1D(uint2 index, uint frameDimX) {
    return index.x + index.y * frameDimX;
}

/*  Visibility test with shadow ray. The ray is dipatched from the surface to the light source
    \param[in] Shading data from the surface the visibility should be tested
    \param[in] Direction to the light source
    \param[in] Distance to the light source
*/
bool dispatchShadowRay(ShadingData sd, float3 toLight, float distanceToLight) {
    //Use ray query, <1> means alpha test is enabled
    SceneRayQuery<1> rayQuery;
    // A epsilon is applied to the distance to avoid self-intersections with emissive triangle light sources
    distanceToLight *= kShadowRayEpsilon;
    const Ray ray = Ray(sd.computeRayOrigin(), toLight, 0.f, distanceToLight); 
    return rayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff);
}

/*  Evaluates a Final Gather sample after resampling and multiplies with BSDF
    Jacobian and visibility ray are not needed, as it was already handled in the resampling process
    \param[in] Current Reservoir
    \param[in] Corresponding shading data
    \param[in] Corresponding Material instance
    \param[in/out] Random Number Generator
    \return Radiance for the reservoir
*/
float3 evaluateFinalGatherSample(ReservoirFG r, ShadingData sd, IMaterialInstance mi, inout SampleGenerator sg) {
    float3 Li = float3(0);
    float3 toLight = r.sampleData.posW - sd.posW;
    float dist = length(toLight);
    toLight /= dist;
    Li = r.sampleData.radianceOut;
    Li *= mi.eval(sd, toLight, sg);

    return Li;
}

/*  Target Function for final gather reservoirs
    \param[in] Shading data
    \param[in] Properties of the BSDF
    \param[in] Direction to Light/Sample
    \return One-channel Target Function that approximates the BSDF
*/
float evalApproximateBRDF(ShadingData sd, BSDFProperties bsdfProperties, float3 toLight) {
    const float3 N = bsdfProperties.guideNormal;
    const float3 H = normalize(sd.V + toLight);
    const float NdotV = saturate(dot(N, sd.V));
    const float NdotL = saturate(dot(N, toLight));
    const float NdotH = saturate(dot(N, H));
    const float LdotH = saturate(dot(toLight, H));
    const float diffuse = luminance(bsdfProperties.diffuseReflectionAlbedo);
    const float specular = luminance(bsdfProperties.specularReflectionAlbedo);

    // BRDF terms for target function (DGF Microfaset BRDF)
    float D = evalNdfGGX(bsdfProperties.roughness * bsdfProperties.roughness, NdotH);
    float G = evalMaskingSmithGGXSeparable(bsdfProperties.roughness, NdotL, NdotV);

    // float F = specular < 1e-8f ? 0.f :  evalFresnelSchlick(specular, 1.f, LdotH);     //Right way
    float F = specular < 1e-8f ? 0.f : evalFresnelSchlick(specular, 1.f, LdotH) / specular; // Not "right" but less noisier

    // eval brdf terms
    float diffuseBRDF = NdotL * M_1_PI;
    float specularBRDF = D * G * F / (4.f * NdotV);

    // Mix diffuse and specular BRDF
    float diffuseProb = 0;
    if(diffuse > 0 || specular > 0)
        diffuseProb = diffuse / (diffuse + specular);
    float brdf = max(0.f, lerp(specularBRDF, diffuseBRDF, diffuseProb));

    return max(0.f, brdf);
}

/*  Target function for radiance estimate resampling (caustics)
    \param[in] Shading data
    \param[in] Properties of the BSDF
    \param[in] Direction to Light/Sample
    \return One-channel Target Function that approximates the BSDF
*/
float evalApproximateBRDFPhoton(ShadingData sd, BSDFProperties bsdfProperties, float3 toLight) {
    const float3 N = bsdfProperties.guideNormal;
    const float3 H = normalize(sd.V + toLight);
    const float NdotV = saturate(dot(N, sd.V));
    const float NdotL = saturate(dot(N, toLight));
    const float NdotH = saturate(dot(N, H));
    const float LdotH = saturate(dot(toLight, H));
    const float diffuse = luminance(bsdfProperties.diffuseReflectionAlbedo);
    const float specular = luminance(bsdfProperties.specularReflectionAlbedo);

    if (NdotL < 0)
        return 0;

    // BRDF terms for target function (DGF Microfaset BRDF)
    float D = evalNdfGGX(bsdfProperties.roughness * bsdfProperties.roughness, NdotH);
    float G = evalMaskingSmithGGXSeparable(bsdfProperties.roughness, NdotL, NdotV);

    // float F = specular < 1e-8f ? 0.f :  evalFresnelSchlick(specular, 1.f, LdotH);     //Right way
    float F = specular < 1e-8f ? 0.f : evalFresnelSchlick(specular, 1.f, LdotH) / specular; // Not "right" but less noisier

    // eval brdf terms
    float diffuseBRDF = M_1_PI;
    float specularBRDF = D * G * F / (4.f * NdotV * NdotL); //Take NdotL out of the specular term

    // Mix diffuse and specular BRDF
    float diffuseProb = 0;
    if (diffuse > 0 || specular > 0)
        diffuseProb = diffuse / (diffuse + specular);
    float brdf = max(0.f, lerp(specularBRDF, diffuseBRDF, diffuseProb));

    return max(0.f, brdf);
}

/*  Unbiased target function (p_hat) with Jacobian and Visibility test
    \param[in] Current Reservoir
    \param[in] Corresponding Shading data
    \param[in] Corresponding Properties of the BSDF
    \param[in] If true, sample is assumed visible (due to previous visibility test) and the visibility test is skipped
    \return One-channel Target Function that approximates the BSDF
*/
float evaluateFinalGatherSampleApproximateUnbiased(ReservoirFG r, ShadingData sd, BSDFProperties bsdfProperties, bool assumeVisible = false)
{
    float Li = 0.0;
    float3 toSurface = sd.posW - r.sampleData.posW;
    float dist = length(toSurface);
    toSurface /= dist;
    float jacobianDeterminant = dot(r.sampleData.normal, toSurface) / (dist * dist);
    jacobianDeterminant /= r.jacobianDeterminant;

    if (jacobianDeterminant < 0 || isnan(jacobianDeterminant) || isinf(jacobianDeterminant))
        return 0.0;

    Li = luminance(r.sampleData.radianceOut) * jacobianDeterminant;
    if (!assumeVisible)
        if (!dispatchShadowRay(sd, -toSurface, dist))
            return 0.0;

    Li *= evalApproximateBRDF(sd, bsdfProperties, -toSurface);

    return Li;
}
