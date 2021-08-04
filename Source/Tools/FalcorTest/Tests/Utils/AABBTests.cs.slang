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
import Utils.Math.AABB;

StructuredBuffer<float3> testData;
RWStructuredBuffer<float3> result;

cbuffer CB
{
    uint n;     // Number of elements in testData
};

[numthreads(1, 1, 1)]
void testAABB(uint3 threadId : SV_DispatchThreadID)
{
    const uint idx = threadId.x;
    uint i = 0;

    // Test 0: set()
    {
        AABB b;
        b.set(testData[0]);
        result[i++] = b.minPoint;
        result[i++] = b.maxPoint;

        b.set(testData[1], testData[2]);
        result[i++] = b.minPoint;
        result[i++] = b.maxPoint;
    }

    // Test 1: valid()
    {
        AABB b;
        b.set(testData[0]);
        result[i++] = b.valid() ? 1 : 0; // expect 1 (single point)
        b.set(testData[0], testData[0] + 0.5f);
        result[i++] = b.valid() ? 1 : 0; // expect 1 (extent is 0.5)
        b.set(testData[0], testData[0] + float3(1.f, 1.f, -0.5f));
        result[i++] = b.valid() ? 1 : 0; // expect 0 (z extent is negative)

        result[i++] = b.minPoint;
        result[i++] = b.maxPoint;
    }

    // Test 2: invalidate()
    {
        AABB b;
        b.set(testData[0], testData[0] + 0.5f);
        b.invalidate();
        result[i++] = b.valid() ? 1 : 0; // expect 0
        result[i++] = b.minPoint;
        result[i++] = b.maxPoint;
    }

    // Test3: include()
    {
        AABB b0;
        b0.invalidate();
        b0.include(testData[0]);
        b0.include(testData[1]);
        result[i++] = b0.minPoint;
        result[i++] = b0.maxPoint;

        AABB b1;
        b1.invalidate();
        b1.include(testData[2]);
        b1.include(testData[3]);
        result[i++] = b1.minPoint;
        result[i++] = b1.maxPoint;

        b0.include(b1);
        result[i++] = b0.minPoint;
        result[i++] = b0.maxPoint;
    }

    // Test 4: contains()
    {
        AABB b;
        b.invalidate();
        result[i++] = b.contains(float3(0)) ? 1 : 0;   // expect 0
        result[i++] = b.contains(testData[0]) ? 1 : 0; // expect 0

        b.set(testData[0]);
        result[i++] = b.contains(testData[0]) ? 1 : 0; // expect 1
        result[i++] = b.contains(testData[1]) ? 1 : 0; // expect 0

        b.invalidate();
        for (uint i = 0; i < n; i++) b.include(testData[i]);
        result[i++] = b.contains(float3( 1.00f, 1.00f, 1.00f)) ? 1 : 0; // expect 1
        result[i++] = b.contains(float3(-3.50f, 1.00f, 1.00f)) ? 1 : 0; // expect 1
        result[i++] = b.contains(float3(-3.50f, 2.75f, 4.50f)) ? 1 : 0; // expect 1
        result[i++] = b.contains(float3(-3.50f, 2.76f, 4.50f)) ? 1 : 0; // expect 0
        result[i++] = b.contains(float3(-9.50f, 8.00f, 0.00f)) ? 1 : 0; // expect 0
    }

    // Test 5: center(), extent(), area(), volume()
    {
        AABB b;
        b.set(testData[0]);
        result[i++] = b.center();
        result[i++] = b.extent();
        result[i++] = b.area();
        result[i++] = b.volume();
        result[i++] = b.radius();

        b.include(testData[0] - float3(0.f, 1.f, 0.f));
        result[i++] = b.center();
        result[i++] = b.extent();
        result[i++] = b.area();
        result[i++] = b.volume();
        result[i++] = b.radius();

        b.invalidate();
        for (uint i = 0; i < n; i++) b.include(testData[i]);
        result[i++] = b.center();
        result[i++] = b.extent();
        result[i++] = b.area();
        result[i++] = b.volume();
        result[i++] = b.radius();
    }

    // Test 6: intersects()
    {
        AABB a, b;
        a.set(testData[0]);     // {  1.00f,  2.50f, -0.50f }
        a.include(testData[1]); // { -3.50f, -0.00f, -1.25f }

        b.set(float3(-5.0f, -5.0f, -5.0f), float3(5.0f, 5.0f, 5.0f));
        result[i].x = a.intersects(b);
        result[i].y = b.intersects(a); // Expect true (box fully inside larger box)
        i++;

        b.set(float3(-4.0f, 2.5f, -0.5f), float3(-3.5f, 3.0f, 0.0f));
        result[i].x = a.intersects(b);
        result[i].y = b.intersects(a); // Expect true (box touches corner of other box)
        i++;

        b.set(float3(1.0f, 1.0f, -1.0f), float3(2.0f, 1.5f, 0.0f));
        result[i].x = a.intersects(b);
        result[i].y = b.intersects(a); // Expect true (box touches side of other box)
        i++;

        b.set(float3(1.0f, -5.0f, -5.0f), float3(1.0f, 5.0f, 5.0f));
        result[i].x = a.intersects(b);
        result[i].y = b.intersects(a); // Expect true (box touches side of other thin box)
        i++;

        b.set(float3(2.0f, -1.0f, -1.0f), float3(100.0f, 100.0f, 100.0f));
        result[i].x = a.intersects(b);
        result[i].y = b.intersects(a); // Expect false
        i++;

        b.set(float3(0.0f, 1.0f, -0.25f), float3(1.0f, 2.0f, 0.0f));
        result[i].x = a.intersects(b);
        result[i].y = b.intersects(a); // Expect false
        i++;
    }

    // Test 7: minDistance(point)
    {
        AABB a;
        a.set(testData[0]);     // {  1.00f,  2.50f, -0.50f }
        a.include(testData[1]); // { -3.50f, -0.00f, -1.25f }

        result[i++] = a.minDistance(float3( 0.0f, 1.0f, -1.0f)); // Expect 0.0 (point fully inside)
        result[i++] = a.minDistance(float3( 1.0f, 1.0f, -1.0f)); // Expect 0.0 (point on side)
        result[i++] = a.minDistance(float3(-3.5f, 0.0f, -0.5f)); // Expect 0.0 (point on corner)
        result[i++] = a.minDistance(float3( 3.0f, 0.0f, -0.5f)); // Expect 2.0 (along axis from corner)
        result[i++] = a.minDistance(float3( 3.5f, 1.0f, -1.0f)); // Expect 2.5 (along axis from side)
        result[i++] = a.minDistance(float3( 5.0f,-3.0f, -0.5f)); // Expect 5.0 (diagonal from corner)
        result[i++] = a.minDistance(float3( 5.0f,-3.0f, -1.0f)); // Expect 5.0 (diagonal from side)
        result[i++] = a.minDistance(float3( 5.0f,-3.0f, 11.5f)); // Expect 13.0 (diagonal from corner)
    }

    // Test 8: minDistance(box)
    {
        AABB a, b;
        a.set(testData[0]);     // {  1.00f,  2.50f, -0.50f }
        a.include(testData[1]); // { -3.50f, -0.00f, -1.25f }

        // For these expect 0.0 (boxes that touch or intersect).
        b.set(float3(-5.0f,-5.0f, -5.0f), float3( 5.0f, 5.0f, 5.0f));
        result[i++] = a.minDistance(b);
        b.set(float3(-4.0f, 2.5f, -0.5f), float3(-3.5f, 3.0f, 0.0f));
        result[i++] = a.minDistance(b);
        b.set(float3( 1.0f, 1.0f, -1.0f), float3( 2.0f, 1.5f, 0.0f));
        result[i++] = a.minDistance(b);
        b.set(float3( 1.0f,-5.0f, -5.0f), float3( 1.0f, 5.0f, 5.0f));
        result[i++] = a.minDistance(b);

        b.set(float3( 2.0f, 0.5f, -1.0f), float3( 2.5f, 1.5f,-0.75f));
        result[i++] = a.minDistance(b); // Expect 1.0
        b.set(float3(-7.5f, 6.5f, -2.0f), float3(-6.5f, 7.5f, 5.0f));
        result[i++] = a.minDistance(b); // Expect 5.0
        b.set(float3(-7.5f, 6.5f, -0.5f), float3(-6.5f, 7.5f, 5.0f));
        result[i++] = a.minDistance(b); // Expect 5.0
        b.set(float3(-7.5f, 6.5f, 11.5f), float3(-6.5f, 7.5f, 12.5f));
        result[i++] = a.minDistance(b); // Expect 13.0
    }
}
