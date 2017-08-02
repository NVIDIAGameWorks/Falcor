
// Struct Declare Code. 
struct VS_IN 
{  
float4 vs_position_input : POSITION; 
};  
// Struct Declare Code. 
struct VS_OUT 
{  
float4 vs_output0 : SV_POSITION; 
};  
 
// Constant Buffers Declare Code. Count = 0 
// Shader Resources Declare Code. Count = 0 
 
//   Vertex Shader Code. 
VS_OUT main ( VS_IN vs_in ) 
{ 
    VS_OUT vs_out; 
    float4 vs_pos_value;  
 
    vs_pos_value = vs_in.vs_position_input; 
 
 
 
 
    vs_out.vs_output0 = vs_pos_value; 
 
 
    return vs_out; 
} 

