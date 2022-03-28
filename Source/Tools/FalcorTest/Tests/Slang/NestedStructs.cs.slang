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
RWStructuredBuffer<uint> result;

struct S1
{
    float2 a;
    uint b;
};

struct S2
{
    bool3 a;
    S1 s1; // 3 values
    float b;
    uint2 c;
};

struct S3
{
    uint a;
    bool b;
    S2 s2; // 9 values
    float3 c;
    S1 s1; // 3 values
};

cbuffer CB
{
    float a;
    S3 s3; // 17 values
    S2 s2; // 9 values
}

[numthreads(1, 1, 1)]
void main()
{
    result[0] = asuint(a);
    result[1] = s3.a;
    result[2] = s3.b ? 1 : 0;
    result[3] = s3.s2.a.x ? 1 : 0;
    result[4] = s3.s2.a.y ? 1 : 0;
    result[5] = s3.s2.a.z ? 1 : 0;
    result[6] = asuint(s3.s2.s1.a.x);
    result[7] = asuint(s3.s2.s1.a.y);
    result[8] = s3.s2.s1.b;
    result[9] = asuint(s3.s2.b);
    result[10] = s3.s2.c.x;
    result[11] = s3.s2.c.y;
    result[12] = asuint(s3.c.x);
    result[13] = asuint(s3.c.y);
    result[14] = asuint(s3.c.z);
    result[15] = asuint(s3.s1.a.x);
    result[16] = asuint(s3.s1.a.y);
    result[17] = s3.s1.b;
    result[18] = s2.a.x ? 1 : 0;
    result[19] = s2.a.y ? 1 : 0;
    result[20] = s2.a.z ? 1 : 0;
    result[21] = asuint(s2.s1.a.x);
    result[22] = asuint(s2.s1.a.y);
    result[23] = s2.s1.b;
    result[24] = asuint(s2.b);
    result[25] = s2.c.x;
    result[26] = s2.c.y;
}
