
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





cbuffer cbR0 
{ 
float4 cbR0_sv0;
float4 cbR0_sv1;


}; 

cbuffer cbR2 
{ 
float4 cbR2_sv0;
float4 cbR2_sv1;


}; 

RWStructuredBuffer<float4> erRWUAV : register ( u7 );



Texture2D<float4> textureR0;

RWTexture2D<float4> rwTextureR0 : register ( u10 );

RWTexture2D<float4> rwTextureR2 : register ( u12 );


RWByteAddressBuffer  rwRBR0;

RWByteAddressBuffer  rwRBR1;



struct DS_OUT 
{ 
float4 output0 : DS_OUTPUT0; 
float4 output1 : DS_OUTPUT1; 
}; 


[domain("quad")] 
DS_OUT main( HS_CONSTANT_DATA_OUTPUT input,  float2 UV : SV_DomainLocation, const OutputPatch<DS_OUT, 16> bezpatch) 
{ 
 
DS_OUT dsOut; 


dsOut.output0 = cbR2_sv1 + textureR0.Load(int3(0, 0, 0)) + rwTextureR0.Load(0) + rwTextureR2.Load(0) + float4(0.05, 0.05, 0.05, 0.05);
dsOut.output1 = cbR2_sv1 + textureR0.Load(int3(0, 0, 0)) + rwTextureR0.Load(0) + rwTextureR2.Load(0) + float4(0.05, 0.05, 0.05, 0.05);


return dsOut; 

} 
 
