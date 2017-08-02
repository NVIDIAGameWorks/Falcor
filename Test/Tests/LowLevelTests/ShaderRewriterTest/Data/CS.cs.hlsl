




cbuffer cbR0  : register ( b0 )
{ 
float4 cbR0_sv0;


}; 

cbuffer cbR1 
{ 
float4 cbR1_sv0;
float4 cbR1_sv1;
float4 cbR1_sv2;


}; 

cbuffer cbR2  : register ( b1 )
{ 
float4 cbR2_sv0;
float4 cbR2_sv1;


}; 

cbuffer cbR3 
{ 
float4 cbR3_sv0;
float4 cbR3_sv1;


}; 

StructuredBuffer<float4> sbR0;

RWStructuredBuffer<float4> rwSBR0 : register ( u0 );

RWStructuredBuffer<float4> rwSBR1;



Texture2D<float4> textureR0 : register ( t0 );

Texture2D<float4> textureR1 : register ( t1 );

RWTexture2D<float4> rwTextureR0 : register ( u1 );

RWTexture2D<float4> rwTextureR1;

RWTexture2D<float4> rwTextureR2;

RWTexture2D<float4> rwTextureR3 : register ( u2 );


ByteAddressBuffer  rbR0;

RWByteAddressBuffer  rwRBR0;

RWByteAddressBuffer  rwRBR1;



[numthreads(4, 4, 4)] 
void main() 
{ 
} 
