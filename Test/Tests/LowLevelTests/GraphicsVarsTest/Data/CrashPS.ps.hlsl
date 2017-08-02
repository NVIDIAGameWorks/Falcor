
// Struct Declare Code. 
struct PS_IN 
{  
float4 ps_input0 : VS_OUTPUT0; 
};  
// Struct Declare Code. 
struct PS_OUT 
{  
float4 ps_output0 : SV_TARGET0; 
};  

// Constant Buffers Declare Code. Count = 1 
cbuffer CB0  : register ( b5 )  
{ 
    float4 color_value0 ; 
 
}; 
 
// Shader Resources Declare Code. Count = 0 

// Pixel Shader Code. 
PS_OUT main ( PS_IN ps_in ) 
{ 
    PS_OUT ps_out; 
 
    float4 color_val0;  
 
 
    color_val0 = color_value0; 
 
 
    color_val0 = color_value0 + float4(0.0, 0.0, 0.0, 0.0); 
 
 
    ps_out.ps_output0 = color_val0; 
 
 
    return ps_out; 
 
} 

