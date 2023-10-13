import Scene.Raster;

cbuffer PerFrameCB {
    float4 gColor;
};

VSOut vsMain(VSIn vIn){
    return defaultVS(vIn);
}

float4 psMain() : SV_TARGET{
    return float4(0, 0, 0, 1);
}
