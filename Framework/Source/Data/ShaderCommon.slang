__exported import ShaderCommonImpl;

import Interfaces;
import Lights;
import Material;



type_param TLightCollection : ILight;

typedef ShadingPoint<TMaterial.BxDF> ShadingData;

#ifdef _VERTEX_BLENDING
float4x4 getBlendedBoneMat(float4 weights, uint4 ids)
{
    float4x4 boneMat = gBoneMat[ids.x] * weights.x;
    boneMat += gBoneMat[ids.y] * weights.y;
    boneMat += gBoneMat[ids.z] * weights.z;
    boneMat += gBoneMat[ids.w] * weights.w;

    return boneMat;
}

float3x3 getBlendedInvTransposeBoneMat(float4 weights, uint4 ids)
{
    float3x3 mat = (float3x3)gInvTransposeBoneMat[ids.x] * weights.x;
    mat += (float3x3)gInvTransposeBoneMat[ids.y] * weights.y;
    mat += (float3x3)gInvTransposeBoneMat[ids.z] * weights.z;
    mat += (float3x3)gInvTransposeBoneMat[ids.w] * weights.w;

    return mat;
}
#endif


float2 calcMotionVector(float2 pixelCrd, float4 prevPosH, float2 renderTargetDim)
{
    float2 prevCrd = prevPosH.xy / prevPosH.w;
#ifdef FALCOR_VK
    prevCrd *= float2(0.5, 0.5);
#else
    prevCrd *= float2(0.5, -0.5);
#endif
    prevCrd += 0.5f;
    float2 normalizedCrd = pixelCrd / renderTargetDim;
    return prevCrd - normalizedCrd;
}

ParameterBlock<TLightEnvironment> gLightEnv;