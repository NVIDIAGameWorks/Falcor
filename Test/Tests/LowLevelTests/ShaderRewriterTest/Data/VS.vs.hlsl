

struct VS_OUT 
{ 
float4 output0 : OUTPUT0; 
float4 output1 : OUTPUT1; 
}; 







cbuffer cbR0 
{ 
float4 cbR0_sv0;


}; 

cbuffer cbR1  : register ( b0 )
{ 
float4 cbR1_sv0;
float4 cbR1_sv1;
float4 cbR1_sv2;


}; 

cbuffer cbR2  : register ( b1 )
{ 
float4 cbR2_sv0;


}; 

cbuffer cbR3 
{ 
float4 cbR3_sv0;
float4 cbR3_sv1;
float4 cbR3_sv2;


}; 

StructuredBuffer<float4> sbR0;



RWTexture2D<float4> rwTextureR1 : register ( u10 );

RWTexture2D<float4> rwTextureR2;


ByteAddressBuffer  rbR0;

RWByteAddressBuffer  rwRBR0;

RWByteAddressBuffer  rwRBR1;


VS_OUT main(float4 pos : POSITION) 
{ 

VS_OUT vsOut; 

vsOut.output0 = cbR0_sv0 + cbR1_sv1 + cbR2_sv0 + cbR3_sv2 + rwTextureR1.Load(0) + rwTextureR2.Load(0) + sbR0[0] + rbR0.Load4(0) + float4(0.05, 0.05, 0.05, 0.05);
vsOut.output1 = cbR0_sv0 + cbR1_sv1 + cbR2_sv0 + cbR3_sv2 + rwTextureR1.Load(0) + rwTextureR2.Load(0) + sbR0[0] + rbR0.Load4(0) + float4(0.05, 0.05, 0.05, 0.05);


return vsOut; 
} 
