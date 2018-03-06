import HostDeviceData;

#ifndef INTERPOLATION_MODE
#define INTERPOLATION_MODE linear
#endif

struct VertexOut
{
    INTERPOLATION_MODE float3 normalW    : NORMAL;
    INTERPOLATION_MODE float3 bitangentW : BITANGENT;
    INTERPOLATION_MODE float2 texC       : TEXCRD;
    INTERPOLATION_MODE float3 posW       : POSW;
    INTERPOLATION_MODE float3 colorV     : COLOR;
    INTERPOLATION_MODE float4 prevPosH   : PREVPOSH;
    INTERPOLATION_MODE float2 lightmapC  : LIGHTMAPUV;
    float4 posH : SV_POSITION;
#ifdef _SINGLE_PASS_STEREO
    INTERPOLATION_MODE float4 rightEyePosS : NV_X_RIGHT;
    uint4 viewportMask : NV_VIEWPORT_MASK;
    uint renderTargetIndex : SV_RenderTargetArrayIndex;
#endif
};

/** Shading result struct
*/
struct ShadingResult
{
    float3 diffuseBrdf;             // The result of the diffuse BRDF
    float3 specularBrdf;            // The result of the specular BRDF
    float3 diffuse;                 // The diffuse component of the result
    float3 specular;                // The specular component of the result
    float4 color;                   // The final color. Alpha holds the opacity valua
};

ShadingResult initShadingResult()
{
    ShadingResult sr;
    sr.diffuse = 0;
    sr.color.rgb = 0;
    sr.color.a = 1;
    sr.specular = 0;
    sr.diffuseBrdf = 0;
    sr.specularBrdf = 0;
    return sr;
}

struct SurfaceGeometry
{
    float3    posW;         ///< Shading hit position in world space
    float3    V;            ///< Direction to the eye at shading hit
    float3    N;            ///< Shading normal at shading hit
    float3    T;            ///< Shading tangent at shading hit
    float3    B;            ///< Shading bitangent at shading hit
    float2    uv;           ///< Texture mapping coordinates
    float     NdotV;        // Unclamped, can be negative
};

SurfaceGeometry initSurfaceGeometry()
{
    SurfaceGeometry rs;
    rs.posW = 0;
    rs.V = 0;
    rs.N = 0;
    rs.T = 0;
    rs.B = 0;
    rs.uv = 0;
    rs.NdotV = 0;
    return rs;
}

struct LightSample
{
    float3 diffuse;   // The light intensity at the surface location used for the diffuse term
    float3 specular;  // The light intensity at the surface location used for the specular term. For light probes, the diffuse and specular components are different
    float3 L;         // The direction from the surface to the light source
    float3 posW;      // The world-space position of the light 
    float NdotH;      // Unclamped, can be negative
    float NdotL;      // Unclamped, can be negative
    float LdotH;      // Unclamped, can be negative
    float distance;   // Distance from the light-source to the surface
};

interface IBxDF
{
    ShadingResult eval(SurfaceGeometry geom, LightSample light, float shadowFactor);
    ShadingResult evalAmbient(float3 ambientLight);
    ShadingResult evalProbe(SurfaceGeometry geom, LightProbeData lightProbe);
    float3 getIntrinsicLighting();
    float getOpacity();
}

struct ShadingPoint<TBxDF:IBxDF>
{
    SurfaceGeometry geom;
    TBxDF bxdf;
};

interface IMaterial
{
    associatedtype BxDF : IBxDF;
    ShadingPoint<BxDF> evalShadingPoint(VertexOut v, float3 camPosW);
}

interface ILight
{
    ShadingResult evalLighting<TBxDF:IBxDF>(ShadingPoint<TBxDF> p, float shadowFactor);
}

ShadingResult evalMaterial<TBxDF : IBxDF, TLight : ILight>(ShadingPoint<TBxDF> p, TLight light, float shadowFactor)
{
    return light.evalLighting(p, shadowFactor);
}

ShadingPoint<TMaterial.BxDF> prepareShadingData<TMaterial : IMaterial>(VertexOut v, TMaterial mat, float3 camPosW)
{
    return mat.evalShadingPoint(v, camPosW);
}