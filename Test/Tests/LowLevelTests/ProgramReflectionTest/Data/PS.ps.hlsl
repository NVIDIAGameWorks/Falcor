
// Struct Declare Code. 
struct PS_IN 
{  
};  
// Struct Declare Code. 
struct PS_OUT 
{  
float4 ps_output : SV_TARGET0; 
};  

// Constant Buffers Declare Code. Count = 5 
cbuffer cb0  : register ( b0 )  
{ 
}; 
 
cbuffer cb1  
{ 
}; 
 
cbuffer cb2  
{ 
}; 
 
cbuffer cb3  
{ 
}; 
 
cbuffer cb4  : register ( b4 , space0 )  
{ 
}; 
 
// Structured Buffers Resources Declare Code. Count = 8 
// Struct Declare Code. 
struct sbRStruct0 
{  
    float4 sbR0 ; 
    float4 sbR1 ; 
};  
 
StructuredBuffer<sbRStruct0> sbR0[3] ; 
 
 // Struct Declare Code. 
struct sbRStruct1 
{  
    float4 sbR0 ; 
    float4 sbR1 ; 
};  
 
StructuredBuffer<sbRStruct1> sbR1 : register ( t10 , space0 )  ; 
 
 // Struct Declare Code. 
struct sbRStruct2 
{  
    float4 sbR0 ; 
    float4 sbR1 ; 
};  
 
StructuredBuffer<sbRStruct2> sbR2[4] ; 
 
 // Struct Declare Code. 
struct sbRStruct3 
{  
    float4 sbR0 ; 
    float4 sbR1 ; 
};  
 
StructuredBuffer<sbRStruct3> sbR3[2] : register ( t16 , space0 )  ; 
 
  
 // Texture Resources Declare Code. Count = 3 
Texture2D<float4> textureR0 [7] : register ( t0 , space0 ) ; 
Texture2D<float4> textureR1 ; 
Texture2D<float4> textureR2 [2]; 
 
 // Sampler Resources Declare Code. Count = 8 
SamplerState samplerRArray0[3] : register ( s0 , space0 ) ; 
SamplerState samplerRArray1 : register ( s3 ) ; 
SamplerState samplerRArray2 : register ( s4 ) ; 
SamplerState samplerRArray3 : register ( s0 , space1 ) ; 
SamplerState samplerRArray4[3] : register ( s1 , space1 ) ; 
SamplerState samplerRArray5 : register ( s4 , space1 ) ; 
SamplerState samplerRArray6[3] : register ( s0 , space2 ) ; 
SamplerState samplerRArray7[2] : register ( s3 , space2 ) ; 
 
 
// Pixel Shader Code. 
PS_OUT main ( PS_IN ps_in ) 
{ 
    PS_OUT ps_out; 
 
    float4 base_output;  
 
 
 
    base_output = float4(0.0, 0.0, 0.0, 0.0)  ; 
 
 
    ps_out.ps_output = base_output; 
 
 
    return ps_out; 
 
} 

