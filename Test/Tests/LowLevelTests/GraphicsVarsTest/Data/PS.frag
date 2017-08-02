#version 450 
 
#define VULKAN 100 
 
// Constant Buffers Declare Code. Count = 5 
uniform cb0 { 
}; 
 
uniform cb1 { 
}; 
 
uniform cb2 { 
}; 
 
layout(binding = 3) uniform cb3 { 
}; 
 
uniform cb4 { 
}; 
 
// Shader Resources Declare Code. Count = 0 
// Pixel Shader Code. 
void main() 
{ 
    float4 base_output;  
 
 
 
    base_output = float4(0.0, 0.0, 0.0, 0.0)  ; 
 
 
    ps_out.ps_output = base_output; 
 
 
} 

