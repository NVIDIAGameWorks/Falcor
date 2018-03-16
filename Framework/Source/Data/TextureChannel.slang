#include "HostDeviceSharedMacros.h"
import Interfaces;
/** Load data from a texture
*/
float4 sampleTexture(Texture2D t, SamplerState s, float2 uv, float4 factor, uint mode)
{
    if(mode == ChannelTypeUnused) return 0;
    if(mode == ChannelTypeConst) return factor;
    // else mode == ChannelTypeTexture
    return t.Sample(s, uv);
}

