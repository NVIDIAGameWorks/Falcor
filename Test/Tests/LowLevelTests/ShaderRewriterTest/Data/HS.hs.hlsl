




cbuffer cbR0  : register ( b0 )
{ 
float4 cbR0_sv0;
float4 cbR0_sv1;


}; 

cbuffer cbR1 
{ 
float4 cbR1_sv0;


}; 

cbuffer cbR2 
{ 
float4 cbR2_sv0;
float4 cbR2_sv1;


}; 

cbuffer cbR3  : register ( b1 )
{ 
float4 cbR3_sv0;


}; 

RWStructuredBuffer<float4> rwSBR1;



RWTexture2D<float4> erTex : register ( u7 );

RWTexture2D<float4> rwTextureR0;

RWTexture2D<float4> rwTextureR1;


RWByteAddressBuffer  rwRBR1;



struct HS_OUT 
{ 
float4 output0 : OUTPUT0; 
float4 output1 : OUTPUT1; 
}; 



struct VS_CONTROL_POINT_OUTPUT
{
    float3 vPosition : WORLDPOS;
    float2 vUV       : TEXCOORD0;
    float3 vTangent  : TANGENT;
};

struct HS_CONSTANT_DATA_OUTPUT
{
    float Edges[4]        : SV_TessFactor;
    float Inside[2]       : SV_InsideTessFactor;
    
    float3 vTangent[4]    : TANGENT;
    float2 vUV[4]         : TEXCOORD;
    float3 vTanUCorner[4] : TANUCORNER;
    float3 vTanVCorner[4] : TANVCORNER;
    float4 vCWts          : TANWEIGHTS;
};

#define MAX_POINTS 32

HS_CONSTANT_DATA_OUTPUT patchConstantFunction(InputPatch<VS_CONTROL_POINT_OUTPUT, MAX_POINTS> ip, uint PatchID : SV_PrimitiveID )
{	
    HS_CONSTANT_DATA_OUTPUT hscOut;
    hscOut.Edges[0] = 0.0;
    hscOut.Edges[1] = 0.0;
    hscOut.Edges[2] = 0.0;
    hscOut.Edges[3] = 0.0;

    hscOut.Inside[0] = 0.0;
    hscOut.Inside[1] = 0.0;
    
    
    hscOut.vTangent[0] = float3(0.0, 0.0, 0.0);
    hscOut.vTangent[1] = float3(0.0, 0.0, 0.0);
    hscOut.vTangent[2] = float3(0.0, 0.0, 0.0);
    hscOut.vTangent[3] = float3(0.0, 0.0, 0.0);
    hscOut.vUV[0] = float2(0.0, 0.0);
    hscOut.vUV[1] = float2(0.0, 0.0);
    hscOut.vUV[2] = float2(0.0, 0.0);
    hscOut.vUV[3] = float2(0.0, 0.0);
    hscOut.vTanUCorner[0] = float3(0.0, 0.0, 0.0);
    hscOut.vTanUCorner[1] = float3(0.0, 0.0, 0.0);
    hscOut.vTanUCorner[2] = float3(0.0, 0.0, 0.0);
    hscOut.vTanUCorner[3] = float3(0.0, 0.0, 0.0);

    hscOut.vTanVCorner[0] = float3(0.0, 0.0, 0.0);
    hscOut.vTanVCorner[1] = float3(0.0, 0.0, 0.0);
    hscOut.vTanVCorner[2] = float3(0.0, 0.0, 0.0);
    hscOut.vTanVCorner[3] = float3(0.0, 0.0, 0.0);
    hscOut.vCWts = float4(0.0, 0.0, 0.0, 0.0);
    
    return hscOut;
}

[domain("quad")]
[partitioning("integer")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(16)]
[patchconstantfunc("patchConstantFunction")]
HS_OUT main(InputPatch<VS_CONTROL_POINT_OUTPUT, MAX_POINTS> ip, uint i : SV_OutputControlPointID, uint PatchID : SV_PrimitiveID )
{ 

HS_OUT hsOut; 

hsOut.output0 = cbR1_sv0 + cbR2_sv1 + erTex.Load(0) + rwTextureR0.Load(0) + rwTextureR1.Load(0) + float4(0.05, 0.05, 0.05, 0.05);
hsOut.output1 = cbR1_sv0 + cbR2_sv1 + erTex.Load(0) + rwTextureR0.Load(0) + rwTextureR1.Load(0) + float4(0.05, 0.05, 0.05, 0.05);


return hsOut; 
} 

