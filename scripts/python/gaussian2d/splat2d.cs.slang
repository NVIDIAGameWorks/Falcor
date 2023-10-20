#include "Utils/NVAPI.slangh"

static const float kSkipDistance = 0.1;

struct Blob : IDifferentiable {
    float2 pos;
    float2 scale;
    float3 color;
    float rotation;
};

[BackwardDifferentiable]
float2x2 inverse(float2x2 M)
{
    float2x2 inv;
    float invdet = 1.0f / determinant(M);
    inv[0][0] = M[1][1] * invdet;
    inv[1][1] = M[0][0] * invdet;
    inv[0][1] = -M[0][1] * invdet;
    inv[1][0] = -M[1][0] * invdet;
    return inv;
}

[BackwardDifferentiable]
float2x2 rotation_matrix(float angle)
{
    float c = cos(angle);
    float s = sin(angle);
    return float2x2(c, -s, s, c);
}

[BackwardDifferentiable]
float2x2 scale_matrix(float2 scale)
{
    return float2x2(scale.x, 0, 0, scale.y);
}

[BackwardDifferentiable]
float gaussian(float2 x, float2x2 sigma)
{
    float2x2 sigma_inv = inverse(sigma);
    return exp(-0.5 * dot(x, mul(sigma_inv, x)));
}

[BackwardDifferentiable]
float3 eval(no_diff float2 pos, Blob blob) {
    float2x2 R = rotation_matrix(blob.rotation);
    float2x2 S = scale_matrix(max(blob.scale, 0.001));
    float2x2 sigma = mul(mul(R, S), mul(transpose(S), transpose(R)));
    return gaussian(pos - blob.pos, sigma) * blob.color;
}

struct Forward {
    StructuredBuffer<Blob> blobs;
    uint blob_count;
    RWStructuredBuffer<float3> image;
    uint2 resolution;

    void forward(uint2 idx)
    {
        if (any(idx >= resolution))
            return;

        float2 pos = float2(idx) / resolution;
        float3 color = float3(0);
        for (uint i = 0; i < blob_count; ++i) {
            // if (kSkipDistance > 0 && distance(pos, blobs[i].pos) > kSkipDistance)
            //     continue;
            color += eval(pos, blobs[i]);
        }
        image[idx.y * resolution.x + idx.x] = color;
    }
}

struct Backward {
    StructuredBuffer<Blob> blobs;
    RWByteAddressBuffer grad_blobs;
    uint blob_count;
    StructuredBuffer<float3> grad_image;
    uint2 resolution;

    void write_grad(uint offset, float value)
    {
        float sum = WaveActiveSum(value);
        if (WaveIsFirstLane())
            grad_blobs.InterlockedAddF32(offset, sum);
    }

    void backward(uint2 idx)
    {
        if (any(idx >= resolution))
            return;

        no_diff float2 pos = float2(idx) / resolution;

        for (uint i = 0; i < blob_count; ++i) {
            var d_blob = diffPair(blobs[i]);
            float3 d_image = grad_image[idx.y * resolution.x + idx.x];
            if (kSkipDistance > 0 && distance(pos, blobs[i].pos) > kSkipDistance)
                continue;
            bwd_diff(eval)(pos, d_blob, d_image);
            write_grad(i * 32, d_blob.d.pos.x);
            write_grad(i * 32 + 4, d_blob.d.pos.y);
            write_grad(i * 32 + 8, d_blob.d.scale.x);
            write_grad(i * 32 + 12, d_blob.d.scale.y);
            write_grad(i * 32 + 16, d_blob.d.color.r);
            write_grad(i * 32 + 20, d_blob.d.color.g);
            write_grad(i * 32 + 24, d_blob.d.color.b);
            write_grad(i * 32 + 28, d_blob.d.rotation);
        }
    }
}

ParameterBlock<Forward> forward;
ParameterBlock<Backward> backward;

[numthreads(16, 16, 1)]
void forward_main(uint3 tid: SV_DispatchThreadID)
{
    forward.forward(tid.xy);
}

[numthreads(16, 16, 1)]
void backward_main(uint3 tid: SV_DispatchThreadID)
{
    backward.backward(tid.xy);
}
