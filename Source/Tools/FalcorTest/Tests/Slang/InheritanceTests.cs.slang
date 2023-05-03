/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
StructuredBuffer<int> testType;
StructuredBuffer<int2> testValue;
StructuredBuffer<float3> data;
RWStructuredBuffer<int> resultsInt;
RWStructuredBuffer<float2> resultsFloat;

[anyValueSize(64)]
interface ITestInterface
{
    /// Returns a value stored in the class, won't have a default implementation
    int getStoredValue();
    /// This binary op will have a default implementation, returning the lhs
    float getBinaryOp(float lhs, float rhs);
    /// This binary op will have a default implementation, returning the v
    float getUnaryOp(float v);
};

/// Implements the default implementations, does *NOT* implement the interface.
struct TestInterfaceBase
{
    int value0; ///< This tests if members of base (value0) are in memory before members of inherited class (value1)
    float getBinaryOp(float lhs, float rhs) { return lhs; }
    float getUnaryOp(float v) { return v; }
};

/// Implements all three explicitly
struct TestV0SubNeg : TestInterfaceBase, ITestInterface
{
    int value1;
    int getStoredValue() { return (value0 - value1); }
    float getBinaryOp(float lhs, float rhs) { return lhs - rhs; }
    float getUnaryOp(float v) { return -v; }
}

/// Implements only the mandatory, leaving the other two at defaults
struct TestV1DefDef : TestInterfaceBase, ITestInterface
{
    int value1;
    int getStoredValue() { return (value0 - value1) + 1; }
}

/// Implements mandatory and unary
struct TestV2DefNeg : TestInterfaceBase, ITestInterface
{
    int value1;
    int getStoredValue() { return (value0 - value1) + 2; }
    float getUnaryOp(float v) { return -v; }
}

/// Implements mandatory and binary
struct TestV3SumDef : TestInterfaceBase, ITestInterface
{
    int value1;
    int getStoredValue() { return (value0 - value1) + 3; }
    float getBinaryOp(float lhs, float rhs) { return lhs + rhs; }
}

#ifdef COMPILE_WITH_ERROR
/// Implements binary, unary, but not mandatory, should fail to compile
struct TestDefSumNeg : TestInterfaceBase, ITestInterface
{
    float getBinaryOp(float lhs, float rhs) { return lhs + rhs; }
    float getUnaryOp(float v) { return -v; }
}
#endif

ITestInterface createTestClassManual(int type, int2 value)
{
    switch (type)
    {
    case 0:
    {
        TestV0SubNeg result;
        result.value0 = value[0];
        result.value1 = value[1];
        return result;
    }
    case 1:
    {
        TestV1DefDef result;
        result.value0 = value[0];
        result.value1 = value[1];
        return result;
    }
    case 2:
    {
        TestV2DefNeg result;
        result.value0 = value[0];
        result.value1 = value[1];
        return result;
    }
    case 3:
    {
        TestV3SumDef result;
        result.value0 = value[0];
        result.value1 = value[1];
        return result;
    }
#ifdef COMPILE_WITH_ERROR
    case 4:
    {
        TestDefSumNeg result;
        return result;
    }
#endif
    default:
    {
        TestV3SumDef result;
        result.value0 = -65536;
        result.value1 = -65536;
        return result;
    }
    }
}

ITestInterface createTestClassConformance(int type, int2 value)
{
    return createDynamicObject<ITestInterface, int2>(type, value);
}

[numthreads(16, 1, 1)]
void testInheritanceManual(uint3 threadID: SV_DispatchThreadID)
{
    uint i = threadID.x;
    if (i >= NUM_TESTS)
        return;

    ITestInterface test = createTestClassManual(testType[i], testValue[i]);
    resultsInt[i] = test.getStoredValue();
    resultsFloat[i][0] = test.getBinaryOp(data[i][0], data[i][1]);
    resultsFloat[i][1] = test.getUnaryOp(data[i][2]);
}

[numthreads(16, 1, 1)]
void testInheritanceConformance(uint3 threadID: SV_DispatchThreadID)
{
    uint i = threadID.x;
    if (i >= NUM_TESTS)
        return;

    ITestInterface test = createTestClassConformance(testType[i], testValue[i]);
    resultsInt[i] = test.getStoredValue();
    resultsFloat[i][0] = test.getBinaryOp(data[i][0], data[i][1]);
    resultsFloat[i][1] = test.getUnaryOp(data[i][2]);
}
