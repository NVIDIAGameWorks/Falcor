

struct PS_IN 
{ 
float4 input0 : OUTPUT0; 
float4 input1 : OUTPUT1; 
float4 input2 : OUTPUT2; 
float4 input3 : OUTPUT3; 
float4 input4 : OUTPUT4; 
float4 input5 : OUTPUT5; 
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
}; 






cbuffer cbR0 
{ 
float4 cbR0_sv0;


}; 

cbuffer cbR3  : register ( b2 )
{ 
float4 cbR3_sv0;
float4 cbR3_sv1;
float4 cbR3_sv2;


}; 

StructuredBuffer<float4> sbR0 : register ( t0 );

RWStructuredBuffer<float4> rwSBR0 : register ( u7 );

RWStructuredBuffer<float4> rwSBR1;



Texture2D<float4> textureR0 : register ( t2 );

Texture2D<float4> textureR1;

RWTexture2D<float4> rwTextureR0 : register ( u9 );

RWTexture2D<float4> rwTextureR2;

RWTexture2D<float4> rwTextureR3;


ByteAddressBuffer  rbR0 : register ( t1 );

RWByteAddressBuffer  rwRBR1 : register ( u8 );


PS_OUT main(PS_IN psIn) 
{ 

PS_OUT psOut; 

psOut.output0 = cbR0_sv0 + cbR3_sv2 + textureR0.Load(int3(0, 0, 0)) + textureR1.Load(int3(0, 0, 0)) + rwTextureR0.Load(0) + rwTextureR2.Load(0) + rwTextureR3.Load(0) + sbR0[0] + rwSBR1[0] + rbR0.Load4(0) + psIn.input0 + psIn.input1 + psIn.input2 + psIn.input3 + psIn.input4 + psIn.input5 + float4(0.05, 0.05, 0.05, 0.05);
psOut.output1 = cbR0_sv0 + cbR3_sv2 + textureR0.Load(int3(0, 0, 0)) + textureR1.Load(int3(0, 0, 0)) + rwTextureR0.Load(0) + rwTextureR2.Load(0) + rwTextureR3.Load(0) + sbR0[0] + rwSBR1[0] + rbR0.Load4(0) + psIn.input0 + psIn.input1 + psIn.input2 + psIn.input3 + psIn.input4 + psIn.input5 + float4(0.05, 0.05, 0.05, 0.05);
psOut.output2 = cbR0_sv0 + cbR3_sv2 + textureR0.Load(int3(0, 0, 0)) + textureR1.Load(int3(0, 0, 0)) + rwTextureR0.Load(0) + rwTextureR2.Load(0) + rwTextureR3.Load(0) + sbR0[0] + rwSBR1[0] + rbR0.Load4(0) + psIn.input0 + psIn.input1 + psIn.input2 + psIn.input3 + psIn.input4 + psIn.input5 + float4(0.05, 0.05, 0.05, 0.05);
psOut.output3 = cbR0_sv0 + cbR3_sv2 + textureR0.Load(int3(0, 0, 0)) + textureR1.Load(int3(0, 0, 0)) + rwTextureR0.Load(0) + rwTextureR2.Load(0) + rwTextureR3.Load(0) + sbR0[0] + rwSBR1[0] + rbR0.Load4(0) + psIn.input0 + psIn.input1 + psIn.input2 + psIn.input3 + psIn.input4 + psIn.input5 + float4(0.05, 0.05, 0.05, 0.05);
psOut.output4 = cbR0_sv0 + cbR3_sv2 + textureR0.Load(int3(0, 0, 0)) + textureR1.Load(int3(0, 0, 0)) + rwTextureR0.Load(0) + rwTextureR2.Load(0) + rwTextureR3.Load(0) + sbR0[0] + rwSBR1[0] + rbR0.Load4(0) + psIn.input0 + psIn.input1 + psIn.input2 + psIn.input3 + psIn.input4 + psIn.input5 + float4(0.05, 0.05, 0.05, 0.05);
psOut.output5 = cbR0_sv0 + cbR3_sv2 + textureR0.Load(int3(0, 0, 0)) + textureR1.Load(int3(0, 0, 0)) + rwTextureR0.Load(0) + rwTextureR2.Load(0) + rwTextureR3.Load(0) + sbR0[0] + rwSBR1[0] + rbR0.Load4(0) + psIn.input0 + psIn.input1 + psIn.input2 + psIn.input3 + psIn.input4 + psIn.input5 + float4(0.05, 0.05, 0.05, 0.05);
psOut.output6 = cbR0_sv0 + cbR3_sv2 + textureR0.Load(int3(0, 0, 0)) + textureR1.Load(int3(0, 0, 0)) + rwTextureR0.Load(0) + rwTextureR2.Load(0) + rwTextureR3.Load(0) + sbR0[0] + rwSBR1[0] + rbR0.Load4(0) + psIn.input0 + psIn.input1 + psIn.input2 + psIn.input3 + psIn.input4 + psIn.input5 + float4(0.05, 0.05, 0.05, 0.05);


return psOut; 
} 
