/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
RWStructuredBuffer<uint> result;

#ifdef USE_INT
typedef int Type
#else
typedef uint Type
#endif

#ifdef TEST_A

    //
    // Test generic interface.
    //

    interface ITestA<let N : Type> {
    static const Type scale;
    static void eval(Type val[N], out Type res[N]);
};

struct TestA<let N : Type> : ITestA<N>
{
    static const Type scale = 3 * N;
    static void eval(Type val[N], out Type res[N])
    {
        for (uint i = 0; i < N; i++)
            res[i] = scale * val[i];
    }
};

[numthreads(32, 1, 1)]
void testGenericsInterface(uint3 threadID: SV_DispatchThreadID)
{
    const uint i = threadID.x;
    if (i >= 32)
        return;

    Type val[4] = { i, i + 1, i + 2, i + 3 };
    Type res[4];

    TestA<4> test;
    test.eval(val, res);

    result[i * 4 + 0] = res[0];
    result[i * 4 + 1] = res[1];
    result[i * 4 + 2] = res[2];
    result[i * 4 + 3] = res[3];
}

#elif defined(TEST_B)

    //
    // Test generic function in interface.
    //

    interface ITestB {
    static void eval<let N : Type>(Type val[N], out Type res[N]);
};

struct TestB : ITestB
{
    static void eval<let N : Type>(Type val[N], out Type res[N])
    {
        static const Type scale = 3 * N;
        for (uint i = 0; i < N; i++)
            res[i] = scale * val[i];
    }
};

[numthreads(32, 1, 1)]
void testGenericsFunction(uint3 threadID: SV_DispatchThreadID)
{
    const uint i = threadID.x;
    if (i >= 32)
        return;

    Type val[4] = { i, i + 1, i + 2, i + 3 };
    Type res[4];

    TestB test;
    test.eval<4>(val, res);

    result[i * 4 + 0] = res[0];
    result[i * 4 + 1] = res[1];
    result[i * 4 + 2] = res[2];
    result[i * 4 + 3] = res[3];
}

#endif
