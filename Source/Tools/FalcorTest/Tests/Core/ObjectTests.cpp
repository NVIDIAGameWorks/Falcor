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
#include "Testing/UnitTest.h"
#include "Core/Object.h"

namespace Falcor
{

class DummyObject : public Object
{
    FALCOR_OBJECT(DummyObject)
public:
    DummyObject() { getCount()++; }
    ~DummyObject() { getCount()--; }

    static uint32_t& getCount()
    {
        static uint32_t sCount = 0;
        return sCount;
    }
};

CPU_TEST(Object_ref)
{
    ASSERT_EQ(DummyObject::getCount(), 0);

    ref<DummyObject> r1;
    ref<DummyObject> r2;

    EXPECT_TRUE(r1 == r1);
    EXPECT_TRUE(r1 == r2);
    EXPECT_TRUE(r1 == nullptr);
    EXPECT_FALSE(r1 != r1);
    EXPECT_FALSE(r1 != r2);
    EXPECT_FALSE(r1 != nullptr);
    EXPECT_FALSE(bool(r1));
    EXPECT(r1.get() == nullptr);

    r1 = make_ref<DummyObject>();
    EXPECT_EQ(DummyObject::getCount(), 1);
    EXPECT_EQ(r1->refCount(), 1);

    EXPECT_TRUE(r1 == r1);
    EXPECT_FALSE(r1 == r2);
    EXPECT_FALSE(r1 == nullptr);
    EXPECT_FALSE(r1 != r1);
    EXPECT_TRUE(r1 != r2);
    EXPECT_TRUE(r1 != nullptr);
    EXPECT_TRUE(bool(r1));
    EXPECT(r1.get() != nullptr);

    r2 = r1;
    EXPECT_EQ(DummyObject::getCount(), 1);
    EXPECT_EQ(r1->refCount(), 2);
    EXPECT_TRUE(r1 == r2);
    EXPECT_FALSE(r1 != r2);

    r2 = nullptr;
    EXPECT_EQ(DummyObject::getCount(), 1);
    EXPECT_EQ(r1->refCount(), 1);

    r1 = nullptr;
    EXPECT_EQ(DummyObject::getCount(), 0);
}

class DummyBuffer;

class DummyDevice : public Object
{
    FALCOR_OBJECT(DummyDevice)
public:
    ref<DummyBuffer> buffer;

    DummyDevice() { getCount()++; }
    ~DummyDevice() { getCount()--; }

    static uint32_t& getCount()
    {
        static uint32_t sCount = 0;
        return sCount;
    }
};

class DummyBuffer : public Object
{
    FALCOR_OBJECT(DummyBuffer)
public:
    BreakableReference<DummyDevice> device;

    DummyBuffer(ref<DummyDevice> device) : device(std::move(device)) { getCount()++; }
    ~DummyBuffer() { getCount()--; }

    static uint32_t& getCount()
    {
        static uint32_t sCount = 0;
        return sCount;
    }
};

CPU_TEST(Object_BreakableReference)
{
    ASSERT_EQ(DummyDevice::getCount(), 0);
    ASSERT_EQ(DummyBuffer::getCount(), 0);

    {
        ref<DummyDevice> device = make_ref<DummyDevice>();

        // Create a buffer that has a reference to the device -> cyclic reference
        device->buffer = make_ref<DummyBuffer>(device);

        EXPECT_EQ(DummyDevice::getCount(), 1);
        EXPECT_EQ(DummyBuffer::getCount(), 1);

        DummyBuffer* bufferPtr = device->buffer.get();

        // Release the device
        device = nullptr;

        // Device is not released as there is still a reference from the buffer
        EXPECT_EQ(DummyDevice::getCount(), 1);
        EXPECT_EQ(DummyBuffer::getCount(), 1);

        // Break the cycle
        bufferPtr->device.breakStrongReference();

        EXPECT_EQ(DummyDevice::getCount(), 0);
        EXPECT_EQ(DummyBuffer::getCount(), 0);
    }

    {
        ref<DummyDevice> device = make_ref<DummyDevice>();

        // Create a buffer that has a reference to the device -> cyclic reference
        device->buffer = make_ref<DummyBuffer>(device);
        // Immediately break the cycle
        device->buffer->device.breakStrongReference();

        EXPECT_EQ(DummyDevice::getCount(), 1);
        EXPECT_EQ(DummyBuffer::getCount(), 1);

        // Release the device
        device = nullptr;

        // Device is released as there is no strong reference from the buffer
        EXPECT_EQ(DummyDevice::getCount(), 0);
        EXPECT_EQ(DummyBuffer::getCount(), 0);
    }
}

} // namespace Falcor
