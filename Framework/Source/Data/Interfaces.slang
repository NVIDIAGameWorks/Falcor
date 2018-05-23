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
    float shadowsDepthC : SYS_DEPTH;
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

struct GBufferElement
{
    float3 position;
    float3 normal;
    float3 baseColor;
    float3 specular;
    float roughness;
    float3 intrinsicLighting;
};

GBufferElement initGBufferElem()
{
    GBufferElement rs;
    rs.position = 0;
    rs.normal = 0;
    rs.baseColor = 0;
    rs.specular = 0;
    rs.roughness = 0;
    rs.intrinsicLighting = 0;
    return rs;
}

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
    float     NdotV;        // Unclamped, can be negative
    float4    posH;
    float3    V;            ///< Direction to the eye at shading hit
    float3    N;            ///< Shading normal at shading hit
    float3    T;            ///< Shading tangent at shading hit
    float3    B;            ///< Shading bitangent at shading hit
    float2    uv;           ///< Texture mapping coordinates
    float     shadowDepthC;
    void packGBuffer(inout GBufferElement elem)
    {
        elem.position = posW;
        elem.normal = N;
    }
};

SurfaceGeometry initSurfaceGeometry()
{
    SurfaceGeometry rs;
    rs.posW = 0;
    rs.posH = 0;
    rs.V = 0;
    rs.N = 0;
    rs.T = 0;
    rs.B = 0;
    rs.uv = 0;
    rs.NdotV = 0;
    rs.shadowDepthC = 0;
    return rs;
}

SurfaceGeometry surfaceGeometryLoadFromGBuffer(GBufferElement elem, CameraData cam)
{
    SurfaceGeometry rs;
    rs.posW = elem.position;
    rs.posH = mul(float4(rs.posW, 1.0), cam.viewProjMat);
    rs.V = normalize(cam.posW - rs.posW);
    rs.N = elem.normal;
    rs.T = 0;
    rs.B = 0;
    rs.uv = 0;
    rs.NdotV = dot(rs.N, rs.V);
    rs.shadowDepthC = 0;
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

struct GlobalLightParams
{
    SamplerState probeSampler;
};

interface IBxDF
{
    ShadingResult eval(SurfaceGeometry geom, LightSample light, float shadowFactor);
    ShadingResult evalAmbient(float3 ambientLight);
    ShadingResult evalProbe(SurfaceGeometry geom, LightProbeData lightProbe, GlobalLightParams gParams);
    float3 getIntrinsicLighting();
    float getOpacity();
    float getRoughness();
    float3 getSpecular();
    void packGBuffer(inout GBufferElement);
}

interface IAlphaTest
{
    void run(float alpha, float threshold, float3 posW);
}

struct ShadingPoint<TBxDF:IBxDF>
{
    SurfaceGeometry geom;
    TBxDF bxdf;
    
};

// slang bug: this function cannot be member of ShadingPoint<TBxDF>
GBufferElement packGBuffer<TBxDF:IBxDF>(ShadingPoint<TBxDF> shadingPoint)
{
    GBufferElement rs = initGBufferElem();
    shadingPoint.bxdf.packGBuffer(rs);
    shadingPoint.geom.packGBuffer(rs);
    return rs;
}

interface IMaterial
{
    associatedtype BxDF : IBxDF;
    ShadingPoint<BxDF> evalShadingPoint(VertexOut v, float3 camPosW);
}

interface ILightFilter
{
    bool filter<TLight:ILight>(TLight light);
};

interface ILightIntegrator
{
    associatedtype AggregateResult;
    AggregateResult processLight<TLight:ILight>(AggregateResult prev, TLight light);
};

interface ILightCollection
{
    int getLightCount();
    int filter<TLightFilter:ILightFilter>(int i0, TLightFilter filter, RWStructuredBuffer<int> indexRange, int indexStart);
    TLightIntegrator.AggregateResult integrateRanged<TLightIntegrator:ILightIntegrator>(TLightIntegrator integrator, TLightIntegrator.AggregateResult initVal, int i0, StructuredBuffer<int> range, inout int indexStart, int indexEnd);
    TLightIntegrator.AggregateResult integrateAll<TLightIntegrator:ILightIntegrator>(TLightIntegrator integrator, TLightIntegrator.AggregateResult initVal);
};

interface ILight
{
    ShadingResult evalLighting<TBxDF:IBxDF>(ShadingPoint<TBxDF> p, GlobalLightParams globalLightParams);
    float3 getLightDir(float3 p);
}

struct LightEnv<TLight : ILightCollection>
{
    TLight lights;
    GlobalLightParams params;
};

struct LightIntegrator<TBxDF : IBxDF> : ILightIntegrator
{
    ShadingPoint<TBxDF> p;
    GlobalLightParams globalLightParams;

    typedef ShadingResult AggregateResult;
    
    ShadingResult processLight<TLight:ILight>(ShadingResult prev, TLight light)
    {
        ShadingResult rs = light.evalLighting(p, globalLightParams);
        rs.diffuse += prev.diffuse;
        rs.specular += prev.specular;
        rs.color += prev.color;
        rs.color.a = prev.color.a;
        return rs;
    }
};

ShadingResult evalMaterial<TBxDF : IBxDF, TLight : ILightCollection>(ShadingPoint<TBxDF> p, LightEnv<TLight> lEnv)
{
    LightIntegrator<TBxDF> integrator;
    integrator.p = p;
    integrator.globalLightParams = lEnv.params;
    return lEnv.lights.integrateAll(integrator, initShadingResult());
}

// deprecated
ShadingResult evalMaterial<TBxDF : IBxDF, TLight : ILightCollection>(ShadingPoint<TBxDF> p, float shadowFactor, LightEnv<TLight> lEnv)
{
    LightIntegrator<TBxDF> integrator;
    integrator.p = p;
    integrator.globalLightParams = lEnv.params;
    return lEnv.lights.integrateAll(integrator, initShadingResult());
}

ShadingPoint<TMaterial.BxDF> prepareShadingData<TMaterial : IMaterial>(VertexOut v, TMaterial mat, float3 camPosW)
{
    return mat.evalShadingPoint(v, camPosW);
}
