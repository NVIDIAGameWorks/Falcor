

struct VS_OUT 
{ 
}; 


struct PS_OUT 
{ 
float4 output0 : SV_TARGET0; 
float4 output1 : SV_TARGET1; 
float4 output2 : SV_TARGET2; 
float4 output3 : SV_TARGET3; 
float4 output4 : SV_TARGET4; 
float4 output5 : SV_TARGET5; 
float4 output6 : SV_TARGET6; 
float4 output7 : SV_TARGET7; 
}; 





cbuffer cbR0 
{ 
float4 cbR0_sv0;
float4 cbR0_sv1;


}; 

cbuffer cbR1 
{ 
float4 cbR1_sv0;
float4 cbR1_sv1;
float4 cbR1_sv2;


}; 

cbuffer cbR2 
{ 
float4 cbR2_sv0;


}; 

cbuffer cbR3 
{ 
float4 cbR3_sv0;
float4 cbR3_sv1;


}; 

StructuredBuffer<float4> sbR0 : register ( t0 );

StructuredBuffer<float4> sbR1 : register ( t1 );

StructuredBuffer<float4> sbR2 : register ( t2 );

StructuredBuffer<float4> sbR3;

RWStructuredBuffer<float4> rwSBR0;

RWStructuredBuffer<float4> rwSBR1;

RWStructuredBuffer<float4> rwSBR2 : register ( u8 );



Texture2D<float4> textureR0;

Texture2D<float4> textureR1;


ByteAddressBuffer  rbR0;

ByteAddressBuffer  rbR2;

RWByteAddressBuffer  rwRBR0;

RWByteAddressBuffer  rwRBR1;


PS_OUT main(VS_OUT vOut) 
{ 

PS_OUT psOut; 

psOut.output0 = cbR3_sv0 + textureR0.Load(int3(0, 0, 0)) + textureR1.Load(int3(0, 0, 0)) + sbR0[0] + sbR2[0] + sbR3[0] + rwSBR0[0] + rbR0.Load4(0) + rbR2.Load4(0) + float4(0.05, 0.05, 0.05, 0.05);
psOut.output1 = cbR3_sv0 + textureR0.Load(int3(0, 0, 0)) + textureR1.Load(int3(0, 0, 0)) + sbR0[0] + sbR2[0] + sbR3[0] + rwSBR0[0] + rbR0.Load4(0) + rbR2.Load4(0) + float4(0.05, 0.05, 0.05, 0.05);
psOut.output2 = cbR3_sv0 + textureR0.Load(int3(0, 0, 0)) + textureR1.Load(int3(0, 0, 0)) + sbR0[0] + sbR2[0] + sbR3[0] + rwSBR0[0] + rbR0.Load4(0) + rbR2.Load4(0) + float4(0.05, 0.05, 0.05, 0.05);
psOut.output3 = cbR3_sv0 + textureR0.Load(int3(0, 0, 0)) + textureR1.Load(int3(0, 0, 0)) + sbR0[0] + sbR2[0] + sbR3[0] + rwSBR0[0] + rbR0.Load4(0) + rbR2.Load4(0) + float4(0.05, 0.05, 0.05, 0.05);
psOut.output4 = cbR3_sv0 + textureR0.Load(int3(0, 0, 0)) + textureR1.Load(int3(0, 0, 0)) + sbR0[0] + sbR2[0] + sbR3[0] + rwSBR0[0] + rbR0.Load4(0) + rbR2.Load4(0) + float4(0.05, 0.05, 0.05, 0.05);
psOut.output5 = cbR3_sv0 + textureR0.Load(int3(0, 0, 0)) + textureR1.Load(int3(0, 0, 0)) + sbR0[0] + sbR2[0] + sbR3[0] + rwSBR0[0] + rbR0.Load4(0) + rbR2.Load4(0) + float4(0.05, 0.05, 0.05, 0.05);
psOut.output6 = cbR3_sv0 + textureR0.Load(int3(0, 0, 0)) + textureR1.Load(int3(0, 0, 0)) + sbR0[0] + sbR2[0] + sbR3[0] + rwSBR0[0] + rbR0.Load4(0) + rbR2.Load4(0) + float4(0.05, 0.05, 0.05, 0.05);
psOut.output7 = cbR3_sv0 + textureR0.Load(int3(0, 0, 0)) + textureR1.Load(int3(0, 0, 0)) + sbR0[0] + sbR2[0] + sbR3[0] + rwSBR0[0] + rbR0.Load4(0) + rbR2.Load4(0) + float4(0.05, 0.05, 0.05, 0.05);


return psOut; 
} 
