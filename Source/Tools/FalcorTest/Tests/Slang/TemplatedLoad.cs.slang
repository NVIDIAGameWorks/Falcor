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
RWStructuredBuffer<uint16_t> result;
ByteAddressBuffer data;

[numthreads(1, 1, 1)]
void testTemplatedScalarLoad16()
{
    for (int i = 0; i < 10; i++)
    {
        result[i] = data.Load<uint16_t>(i * 2);
    }
    for (int i = 10; i < 20; i++)
    {
        result[i] = asuint16(data.Load<float16_t>(i * 2));
    }
}

[numthreads(1, 1, 1)]
void testTemplatedVectorLoad16()
{
    uint idx = 0;

    // Test templated load of uint16_t types with 1-4 components.
    uint16_t1 u1 = data.Load<uint16_t1>(idx); idx += 2;
    uint16_t2 u2 = data.Load<uint16_t2>(idx); idx += 4;
    uint16_t3 u3 = data.Load<uint16_t3>(idx); idx += 6;
    uint16_t4 u4 = data.Load<uint16_t4>(idx); idx += 8;

    // Test templated load of float16_t types with 1-4 components.
    float16_t1 f1 = data.Load<float16_t1>(idx); idx += 2;
    float16_t2 f2 = data.Load<float16_t2>(idx); idx += 4;
    float16_t3 f3 = data.Load<float16_t3>(idx); idx += 6;
    float16_t4 f4 = data.Load<float16_t4>(idx); idx += 8;

    idx = 0;

    // Write results.
    result[idx++] = u1.x;
    result[idx++] = u2.x;
    result[idx++] = u2.y;
    result[idx++] = u3.x;
    result[idx++] = u3.y;
    result[idx++] = u3.z;
    result[idx++] = u4.x;
    result[idx++] = u4.y;
    result[idx++] = u4.z;
    result[idx++] = u4.w;

    result[idx++] = asuint16(f1.x);
    result[idx++] = asuint16(f2.x);
    result[idx++] = asuint16(f2.y);
    result[idx++] = asuint16(f3.x);
    result[idx++] = asuint16(f3.y);
    result[idx++] = asuint16(f3.z);
    result[idx++] = asuint16(f4.x);
    result[idx++] = asuint16(f4.y);
    result[idx++] = asuint16(f4.z);
    result[idx++] = asuint16(f4.w);
}

[numthreads(1, 1, 1)]
void testTemplatedMatrixLoad16_2x4()
{
    float16_t2x4 m = data.Load<float16_t2x4>(0);

    uint idx = 0;

    // Write results.
    result[idx++] = asuint16(m[0][0]);
    result[idx++] = asuint16(m[0][1]);
    result[idx++] = asuint16(m[0][2]);
    result[idx++] = asuint16(m[0][3]);
    result[idx++] = asuint16(m[1][0]);
    result[idx++] = asuint16(m[1][1]);
    result[idx++] = asuint16(m[1][2]);
    result[idx++] = asuint16(m[1][3]);
}


[numthreads(1, 1, 1)]
void testTemplatedMatrixLoad16_4x3()
{
    float16_t4x3 m = data.Load<float16_t4x3>(0);

    uint idx = 0;

    // Write results.
    result[idx++] = asuint16(m[0][0]);
    result[idx++] = asuint16(m[0][1]);
    result[idx++] = asuint16(m[0][2]);
    result[idx++] = asuint16(m[1][0]);
    result[idx++] = asuint16(m[1][1]);
    result[idx++] = asuint16(m[1][2]);
    result[idx++] = asuint16(m[2][0]);
    result[idx++] = asuint16(m[2][1]);
    result[idx++] = asuint16(m[2][2]);
    result[idx++] = asuint16(m[3][0]);
    result[idx++] = asuint16(m[3][1]);
    result[idx++] = asuint16(m[3][2]);
}
