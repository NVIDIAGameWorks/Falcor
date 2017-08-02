
// Struct Declare Code. 
struct VS_IN 
{  
float4 val_position_input : POSITION; 
};  
// Struct Declare Code. 
struct VS_OUT 
{  
float4 vs_output : SV_POSITION; 
};  
 
// Constant Buffers Declare Code. Count = 0 
// Shader Resources Declare Code. Count = 0 
 
//   Vertex Shader Code. 
VS_OUT main ( VS_IN vs_in ) 
{ 
    VS_OUT vs_out; 
    float4 position_value;  
 
    position_value = vs_in.val_position_input; 
 
 
 
 
    vs_out.vs_output = position_value; 
 
 
    return vs_out; 
} 

