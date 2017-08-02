
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
cbuffer cb0  
{ 
}; 
 
cbuffer cb1  
{ 
}; 
 
cbuffer cb2  
{ 
}; 
 
cbuffer cb3  : register ( b3 )  
{ 
}; 
 
cbuffer cb4  
{ 
}; 
 
// Shader Resources Declare Code. Count = 0 

// Pixel Shader Code. 
PS_OUT main ( PS_IN ps_in ) 
{ 
    PS_OUT ps_out; 
 
    float4 base_output;  
 
 
 
    base_output = float4(0.0, 0.0, 0.0, 0.0)  ; 
 
 
    ps_out.ps_output = base_output; 
 
 
    return ps_out; 
 
} 

