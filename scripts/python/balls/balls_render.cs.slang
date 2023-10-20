import Utils.Math.HashUtils;

struct Ball {
    float2 pos;
    float2 vel;
};

StructuredBuffer<Ball> g_balls;
uniform uint g_ball_count;
uniform float g_ball_radius;
RWTexture2D<float4> g_output;
uniform uint g_resolution;

float3 pseudocolor(uint value)
{
    uint h = jenkinsHash(value);
    return (uint3(h, h >> 8, h >> 16) & 0xff) / 255.f;
}

[numthreads(16, 16, 1)]
void main(uint3 thread_id: SV_DispatchThreadID)
{
    uint2 pixel = thread_id.xy;
    if (any(pixel >= g_resolution))
        return;

    float3 color = float3(0.f);

    float2 pos = float2(pixel) / float(g_resolution) * 2.f - 1.f;

    for (uint idx = 0; idx < g_ball_count; ++idx)
    {
        Ball ball = g_balls[idx];
        float d = distance(ball.pos, pos);
        if (d < g_ball_radius)
        {
            color += pseudocolor(idx) * smoothstep(g_ball_radius, 0.8f * g_ball_radius, d);
        }
    }

    g_output[pixel] = float4(color, 1.f);
}
