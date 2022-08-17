/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

// We expect tight packing and matrices stored in row-major layout.
// This struct should be 100B holding 32 scalar values (fp16/32).
struct Test1
{
    float2x3 a;         // 24B
    float3x4 b;         // 48B
    float16_t3x2 c;     // 12B
    float16_t2x4 d;     // 16B
};

struct Test2
{
#if LAYOUT == 1
    float16_t4 a, b, c, d;
    float16_t2 getFloat16_2(uint i)
    {
        switch (i)
        {
        case 0: return a.xy;
        case 1: return a.zw;
        case 2: return b.xy;
        case 3: return b.zw;
        case 4: return c.xy;
        case 5: return c.zw;
        case 6: return d.xy;
        case 7: return d.zw;
        }
        return {};
    }

#elif LAYOUT == 2
    float16_t4x4 m;
    float16_t2 getFloat16_2(uint i)
    {
        switch (i)
        {
        case 0: return float16_t2(m[0][0], m[0][1]);
        case 1: return float16_t2(m[0][2], m[0][3]);
        case 2: return float16_t2(m[1][0], m[1][1]);
        case 3: return float16_t2(m[1][2], m[1][3]);
        case 4: return float16_t2(m[2][0], m[2][1]);
        case 5: return float16_t2(m[2][2], m[2][3]);
        case 6: return float16_t2(m[3][0], m[3][1]);
        case 7: return float16_t2(m[3][2], m[3][3]);
        }
        return {};
    }

#elif LAYOUT == 3
    float16_t4x4 m;
    float16_t2 getFloat16_2(uint i)
    {
        switch (i)
        {
        case 0: return m[0].xy;
        case 1: return m[0].zw;
        case 2: return m[1].xy;
        case 3: return m[1].zw;
        case 4: return m[2].xy;
        case 5: return m[2].zw;
        case 6: return m[3].xy;
        case 7: return m[3].zw;
        }
        return {};
    }

#else
    float16_t2 getFloat16_2(uint i) { return {}; }
#endif
};

RWStructuredBuffer<float> result;

StructuredBuffer<Test1> data1;
StructuredBuffer<Test2> data2;

[numthreads(1, 1, 1)]
void testStructuredBufferMatrixLoad1()
{
    Test1 t = data1[0];

    uint idx = 0;

    result[idx++] = t.a[0][0];
    result[idx++] = t.a[0][1];
    result[idx++] = t.a[0][2];
    result[idx++] = t.a[1][0];
    result[idx++] = t.a[1][1];
    result[idx++] = t.a[1][2];

    result[idx++] = t.b[0][0];
    result[idx++] = t.b[0][1];
    result[idx++] = t.b[0][2];
    result[idx++] = t.b[0][3];
    result[idx++] = t.b[1][0];
    result[idx++] = t.b[1][1];
    result[idx++] = t.b[1][2];
    result[idx++] = t.b[1][3];
    result[idx++] = t.b[2][0];
    result[idx++] = t.b[2][1];
    result[idx++] = t.b[2][2];
    result[idx++] = t.b[2][3];

    result[idx++] = t.c[0][0];
    result[idx++] = t.c[0][1];
    result[idx++] = t.c[1][0];
    result[idx++] = t.c[1][1];
    result[idx++] = t.c[2][0];
    result[idx++] = t.c[2][1];

    result[idx++] = t.d[0][0];
    result[idx++] = t.d[0][1];
    result[idx++] = t.d[0][2];
    result[idx++] = t.d[0][3];
    result[idx++] = t.d[1][0];
    result[idx++] = t.d[1][1];
    result[idx++] = t.d[1][2];
    result[idx++] = t.d[1][3];
}

[numthreads(1, 1, 1)]
void testStructuredBufferMatrixLoad2()
{
    Test2 t = data2[0];

    uint idx = 0;

    for (int i = 0; i < 8; i++)
    {
        float16_t2 v = t.getFloat16_2(i);
        result[idx++] = v.x;
        result[idx++] = v.y;
    }
}
