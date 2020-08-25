/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import SlangShared;

RWStructuredBuffer<uint> result;

[numthreads(1, 1, 1)]
void testEnum()
{
    // Store the enum values cast to uint.
    result[0] = (uint) Type1::A;
    result[1] = (uint) Type1::B;
    result[2] = (uint) Type1::C;
    result[3] = (uint) Type1::D;

    result[4] = (uint) Type2::A;
    result[5] = (uint) Type2::B;
    result[6] = (uint) Type2::C;
    result[7] = (uint) Type2::D;

    result[8] = (uint) Type3::A;
    result[9] = (uint) Type3::B;
    result[10] = (uint) Type3::C;
    result[11] = (uint) Type3::D;
}

[numthreads(1, 1, 1)]
void testScalarTypes()
{
    int i = 0;

    // 16/32/64-bit floating-point
    {
        float16_t a = 1 / 3.f;
        result[i++] = asuint((float)a);
    }
    {
        float32_t a = 1 / 5.f;
        result[i++] = asuint(a);
    }
    {
        // NOTE: Must use 'L' suffix for 64-bit literals, otherwise interpreted as 32-bit.
        float64_t a = 1 / 7.0L;
        uint lo, hi;
        asuint(a, lo, hi);
        result[i++] = lo;
        result[i++] = hi;
    }

    // 16/32/64-bit integer
    {
        int16_t a = 30000;      // 0x00007530 -> 0x7530 (30000) in 16-bit.
        int16_t b = -200000;    // 0xfffcf2c0 -> Truncated to 0xf2c0 (-3392) in 16-bit.
        result[i++] = asuint((int)a);
        result[i++] = asuint((int)b);
    }
    {
        int32_t a = 291123;
        int32_t b = -2000000000;
        result[i++] = asuint(a);
        result[i++] = asuint(b);
    }
    {
        // NOTE: Must use 'LL' suffix for signed 64-bit literals, otherwise interpreted as 32-bit.
        int64_t a = 0x12345678aabbccddLL;
        uint lo = a, hi = a >> 32;
        result[i++] = lo;
        result[i++] = hi;
    }

    // 16/32/64-bit unsigned integer
    {
        uint16_t a = 59123;     // 0x0000e6f3 -> Truncated to 0xe6f3 (59123)
        uint16_t b = -10;       // 0xfffffff6 -> Truncated to 0xfff6 (65526)
        result[i++] = (uint)a;
        result[i++] = (uint)b;
    }
    {
        uint32_t a = 0xfedc1234;
        uint32_t b = -129;
        result[i++] = a;
        result[i++] = b;
    }
    {
        // NOTE: Must use 'ULL' suffix for unsigned 64-bit literals, otherwise interpreted as 32-bit.
        uint64_t a = 0x12345678aabbccddULL;
        uint lo = a, hi = a >> 32;
        result[i++] = lo;
        result[i++] = hi;
    }
}

struct A
{
    uint a;
    float b;
};

struct Foo
{
    int a;
    uint b;
    float c;
    bool d;
    float2x2 e;
    A f;
};

[numthreads(1, 1, 1)]
void testDefaultInitializers()
{
    int i = 0;

    // Scalar types (i=0)
    {
        int a = 1, b = 2, c = {};
        a = {};
        b = c;
        result[i++] = a;
        result[i++] = b;
        result[i++] = c;
    }
    {
        uint a = 1, b = 2, c = {};
        a = {};
        b = c;
        result[i++] = a;
        result[i++] = b;
        result[i++] = c;
    }
    {
        float a = 1, b = 2, c = {};
        a = {};
        b = c;
        result[i++] = asuint(a);
        result[i++] = asuint(b);
        result[i++] = asuint(c);
    }
#if 0
    // Fails on SM5.1 probably due to fxc bug with asuint.
    {
        double a = 1, b = 2, c = {};
        a = {};
        b = c;
        asuint(a, result[i], result[i + 1]); i += 2;
        asuint(b, result[i], result[i + 1]); i += 2;
        asuint(c, result[i], result[i + 1]); i += 2;
    }
#endif

    // Vector/matrix types (i=9)
    {
        float4 a = float4(1, 2, 3, 4), b = {};
        a = {};
        result[i++] = asuint(a.x);
        result[i++] = asuint(a.y);
        result[i++] = asuint(a.z);
        result[i++] = asuint(a.w);
        result[i++] = asuint(b.x);
        result[i++] = asuint(b.y);
        result[i++] = asuint(b.z);
        result[i++] = asuint(b.w);
    }
    {
        float2x2 a = float2x2(1, 2, 3, 4), b = {};
        a = {};
        result[i++] = asuint(a[0][0]);
        result[i++] = asuint(a[0][1]);
        result[i++] = asuint(a[1][0]);
        result[i++] = asuint(a[1][1]);
        result[i++] = asuint(b[0][0]);
        result[i++] = asuint(b[0][1]);
        result[i++] = asuint(b[1][0]);
        result[i++] = asuint(b[1][1]);
    }

    // Struct (i=25)
    {
        A myA;
        myA.a = 1;
        myA.b = 2;
        Foo foo;
        foo.a = 1;
        foo.b = 2;
        foo.c = 3;
        foo.d = true;
        foo.e = float2x2(1, 2, 3, 4);
        foo.f = myA;

        foo = {};

        result[i++] = asuint(foo.a);
        result[i++] = asuint(foo.b);
        result[i++] = asuint(foo.c);
        result[i++] = asuint(foo.d);
        result[i++] = asuint(foo.e[0][0]);
        result[i++] = asuint(foo.e[0][1]);
        result[i++] = asuint(foo.e[1][0]);
        result[i++] = asuint(foo.e[1][1]);
        result[i++] = asuint(foo.f.a);
        result[i++] = asuint(foo.f.b);
    }

    // All done (i=35)
}
