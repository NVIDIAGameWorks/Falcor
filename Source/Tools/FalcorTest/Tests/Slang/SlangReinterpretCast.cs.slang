/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#pragma once
#include "Utils/HostDeviceShared.slangh"

 /** This tests uses Slang's 'reinterpret' call to cast a fixed-size struct
     between different struct types containing a mix of 16/32-bit basic types.
     Note all 4B types are 4B aligned to avoid issues due to packing rules.
 */

BEGIN_NAMESPACE_FALCOR

struct Blob
{
    uint4 data[3];
};

struct A
{
    uint a;
    float b;
    float16_t c;
    int16_t d;
    int2 e;
    float3 f;
    uint g;
    float16_t3 h;
    uint16_t i;
    float16_t2 j;
};

struct B
{
    float a[12];
};

struct C
{
    uint4 a;
    float16_t4 b;
    float d[4];
    float16_t4 c;
};

struct D
{
    int16_t a;
    int16_t b;
    uint3 c;
    float d;
    float4 e;
    float3 r;
};

struct E
{
    float16_t4 a[6];
};

struct F
{
    int a;   
    float16_t4 b[5];
    uint16_t c;
    float16_t d;
};

#ifndef HOST_CODE
StructuredBuffer<Blob> data;

RWStructuredBuffer<A> resultA;
RWStructuredBuffer<B> resultB;
RWStructuredBuffer<C> resultC;
RWStructuredBuffer<D> resultD;
RWStructuredBuffer<E> resultE;
RWStructuredBuffer<F> resultF;

[numthreads(32, 1, 1)]
void main(uint3 threadID : SV_DispatchThreadID)
{
    const uint i = threadID.x;

    Blob blob = data[i];

    B b = reinterpret<B, Blob>(blob);
    C c = reinterpret<C, B>(b);
    D d = reinterpret<D, C>(c);
    E e = reinterpret<E, D>(d);
    F f = reinterpret<F, E>(e);
    A a = reinterpret<A, F>(f);

    resultA[i] = a;
    resultB[i] = b;
    resultC[i] = c;
    resultD[i] = d;
    resultE[i] = e;
    resultF[i] = f;
}
#endif

END_NAMESPACE_FALCOR
