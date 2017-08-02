

struct VS_OUT 
{ 
float4 output0 : OUTPUT0; 
float4 output1 : OUTPUT1; 
float4 output2 : OUTPUT2; 
float4 output3 : OUTPUT3; 
float4 output4 : OUTPUT4; 
float4 output5 : OUTPUT5; 
float4 output6 : OUTPUT6; 
float4 output7 : OUTPUT7; 
}; 






cbuffer cbR1 
{ 
float4 cbR1_sv0;
float4 cbR1_sv1;
float4 cbR1_sv2;


}; 

cbuffer cbR3 
{ 
float4 cbR3_sv0;
float4 cbR3_sv1;


}; 

StructuredBuffer<float4> sbR0;

StructuredBuffer<float4> sbR1;

StructuredBuffer<float4> sbR2;

StructuredBuffer<float4> sbR3;

StructuredBuffer<float4> sbR4;

RWStructuredBuffer<float4> rwSBR0 : register ( u8 );

RWStructuredBuffer<float4> rwSBR1;

RWStructuredBuffer<float4> rwSBR2;



Texture2D<float4> textureR1;


ByteAddressBuffer  rbR1;

ByteAddressBuffer  rbR2;

ByteAddressBuffer  rbR3;

RWByteAddressBuffer  rwRBR0;

RWByteAddressBuffer  rwRBR1;


VS_OUT main(float4 pos : POSITION) 
{ 

VS_OUT vsOut; 

vsOut.output0 = cbR3_sv0 + textureR1.Load(int3(0, 0, 0)) + sbR0[0] + sbR2[0] + sbR3[0] + sbR4[0] + rwSBR0[0] + rbR2.Load4(0) + rbR3.Load4(0) + float4(0.05, 0.05, 0.05, 0.05);
vsOut.output1 = cbR3_sv0 + textureR1.Load(int3(0, 0, 0)) + sbR0[0] + sbR2[0] + sbR3[0] + sbR4[0] + rwSBR0[0] + rbR2.Load4(0) + rbR3.Load4(0) + float4(0.05, 0.05, 0.05, 0.05);
vsOut.output2 = cbR3_sv0 + textureR1.Load(int3(0, 0, 0)) + sbR0[0] + sbR2[0] + sbR3[0] + sbR4[0] + rwSBR0[0] + rbR2.Load4(0) + rbR3.Load4(0) + float4(0.05, 0.05, 0.05, 0.05);
vsOut.output3 = cbR3_sv0 + textureR1.Load(int3(0, 0, 0)) + sbR0[0] + sbR2[0] + sbR3[0] + sbR4[0] + rwSBR0[0] + rbR2.Load4(0) + rbR3.Load4(0) + float4(0.05, 0.05, 0.05, 0.05);
vsOut.output4 = cbR3_sv0 + textureR1.Load(int3(0, 0, 0)) + sbR0[0] + sbR2[0] + sbR3[0] + sbR4[0] + rwSBR0[0] + rbR2.Load4(0) + rbR3.Load4(0) + float4(0.05, 0.05, 0.05, 0.05);
vsOut.output5 = cbR3_sv0 + textureR1.Load(int3(0, 0, 0)) + sbR0[0] + sbR2[0] + sbR3[0] + sbR4[0] + rwSBR0[0] + rbR2.Load4(0) + rbR3.Load4(0) + float4(0.05, 0.05, 0.05, 0.05);
vsOut.output6 = cbR3_sv0 + textureR1.Load(int3(0, 0, 0)) + sbR0[0] + sbR2[0] + sbR3[0] + sbR4[0] + rwSBR0[0] + rbR2.Load4(0) + rbR3.Load4(0) + float4(0.05, 0.05, 0.05, 0.05);
vsOut.output7 = cbR3_sv0 + textureR1.Load(int3(0, 0, 0)) + sbR0[0] + sbR2[0] + sbR3[0] + sbR4[0] + rwSBR0[0] + rbR2.Load4(0) + rbR3.Load4(0) + float4(0.05, 0.05, 0.05, 0.05);


return vsOut; 
} 
