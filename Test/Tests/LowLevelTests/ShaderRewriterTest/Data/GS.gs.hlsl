

struct GS_IN 
{ 
float4 input0 : INPUT0; 
}; 



struct GS_OUT 
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






cbuffer cbR0 
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

RWStructuredBuffer<float4> erRWUAV : register ( u7 );

RWStructuredBuffer<float4> rwSBR0 : register ( u8 );

RWStructuredBuffer<float4> rwSBR1;



Texture2D<float4> textureR1;

RWTexture2D<float4> rwTextureR1 : register ( u11 );

RWTexture2D<float4> rwTextureR2;

RWTexture2D<float4> rwTextureR3;


ByteAddressBuffer  rbR0;

RWByteAddressBuffer  rwRBR0;

RWByteAddressBuffer  rwRBR1;



[maxvertexcount(3)] 
 
void main(triangle GS_IN input[3], inout TriangleStream<GS_OUT> triStream) 
{ 
GS_OUT gsOut; 

gsOut.output0 = cbR1_sv0 + cbR2_sv1 + textureR1.Load(int3(0, 0, 0)) + rwTextureR1.Load(0) + rwTextureR2.Load(0) + rwTextureR3.Load(0) + rwSBR0[0] + rbR0.Load4(0) + float4(0.05, 0.05, 0.05, 0.05);
gsOut.output1 = cbR1_sv0 + cbR2_sv1 + textureR1.Load(int3(0, 0, 0)) + rwTextureR1.Load(0) + rwTextureR2.Load(0) + rwTextureR3.Load(0) + rwSBR0[0] + rbR0.Load4(0) + float4(0.05, 0.05, 0.05, 0.05);
gsOut.output2 = cbR1_sv0 + cbR2_sv1 + textureR1.Load(int3(0, 0, 0)) + rwTextureR1.Load(0) + rwTextureR2.Load(0) + rwTextureR3.Load(0) + rwSBR0[0] + rbR0.Load4(0) + float4(0.05, 0.05, 0.05, 0.05);
gsOut.output3 = cbR1_sv0 + cbR2_sv1 + textureR1.Load(int3(0, 0, 0)) + rwTextureR1.Load(0) + rwTextureR2.Load(0) + rwTextureR3.Load(0) + rwSBR0[0] + rbR0.Load4(0) + float4(0.05, 0.05, 0.05, 0.05);
gsOut.output4 = cbR1_sv0 + cbR2_sv1 + textureR1.Load(int3(0, 0, 0)) + rwTextureR1.Load(0) + rwTextureR2.Load(0) + rwTextureR3.Load(0) + rwSBR0[0] + rbR0.Load4(0) + float4(0.05, 0.05, 0.05, 0.05);
gsOut.output5 = cbR1_sv0 + cbR2_sv1 + textureR1.Load(int3(0, 0, 0)) + rwTextureR1.Load(0) + rwTextureR2.Load(0) + rwTextureR3.Load(0) + rwSBR0[0] + rbR0.Load4(0) + float4(0.05, 0.05, 0.05, 0.05);
gsOut.output6 = cbR1_sv0 + cbR2_sv1 + textureR1.Load(int3(0, 0, 0)) + rwTextureR1.Load(0) + rwTextureR2.Load(0) + rwTextureR3.Load(0) + rwSBR0[0] + rbR0.Load4(0) + float4(0.05, 0.05, 0.05, 0.05);
gsOut.output7 = cbR1_sv0 + cbR2_sv1 + textureR1.Load(int3(0, 0, 0)) + rwTextureR1.Load(0) + rwTextureR2.Load(0) + rwTextureR3.Load(0) + rwSBR0[0] + rbR0.Load4(0) + float4(0.05, 0.05, 0.05, 0.05);


} 
