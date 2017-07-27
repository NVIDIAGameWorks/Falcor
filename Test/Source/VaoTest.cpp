/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/
#include "VaoTest.h"

void VaoTest::addTests()
{
    addTestToList<TestSimpleCreate>();
    addTestToList<TestIndexedCreate>();
    addTestToList<TestMultiBufferCreate>();
    addTestToList<TestLayout>();
}

testing_func(VaoTest, TestSimpleCreate)
{
    //Create vertex buffer
    const uint32_t bufferSize = 9u;
    float bufferData[bufferSize] = { 0.f, 0.25f, 0.5f, 0.75f, 1.f, 1.25f, 1.5f, 2.f, 2.25f };
    Buffer::SharedPtr pBuffer = Buffer::create(bufferSize, Resource::BindFlags::Vertex, Buffer::CpuAccess::Read, bufferData);
    Vao::BufferVec bufferVec;
    bufferVec.push_back(pBuffer);

    //create vertex layout
    std::string name = VERTEX_POSITION_NAME;
    ResourceFormat format = ResourceFormat::RGB32Float;
    uint32_t shaderLoc = VERTEX_POSITION_LOC;
    VertexBufferLayout::SharedPtr pBufferLayout = VertexBufferLayout::create();
    pBufferLayout->addElement(name, 0u, format, bufferSize, shaderLoc);
    VertexLayout::SharedPtr pLayout = VertexLayout::create();
    pLayout->addBufferLayout(0u, pBufferLayout);

    //Craete and check vao
    Vao::Topology topology = Vao::Topology::LineStrip;
    Vao::SharedPtr pVao = Vao::create(topology, pLayout, bufferVec);
    //Some of this should be helper function-ized as other tests need it
    if ((pVao->getVertexBuffer(0u)->getBindFlags() & Resource::BindFlags::Vertex) == Resource::BindFlags::None ||
        pVao->getVertexBuffersCount() != 1 ||
        pVao->getIndexBuffer() != nullptr ||
        pVao->getPrimitiveTopology() != topology ||
        pVao->getVertexLayout()->getBufferCount() != 1 ||
        pVao->getVertexLayout()->getBufferLayout(0u)->getElementName(0u) != name ||
        pVao->getVertexLayout()->getBufferLayout(0u)->getElementFormat(0u) != format ||
        pVao->getVertexLayout()->getBufferLayout(0u)->getElementArraySize(0u) != bufferSize ||
        pVao->getVertexLayout()->getBufferLayout(0u)->getElementShaderLocation(0u) != shaderLoc)
    {
        return test_fail("Vao's properties do not match the properties that were used to create it");
    }

    return test_pass();
}

testing_func(VaoTest, TestIndexedCreate)
{
    //Create vertex buffer
    const uint32_t bufferSize = 9u;
    float bufferData[bufferSize] = { 0.f, 0.25f, 0.5f, 0.75f, 1.f, 1.25f, 1.5f, 2.f, 2.25f };
    Buffer::SharedPtr pBuffer = Buffer::create(bufferSize, Resource::BindFlags::Vertex, Buffer::CpuAccess::Read, bufferData);
    Vao::BufferVec bufferVec;
    bufferVec.push_back(pBuffer);

    //create veretx layout 
    std::string name = VERTEX_POSITION_NAME;
    ResourceFormat format = ResourceFormat::RGB32Float;
    uint32_t shaderLoc = VERTEX_POSITION_LOC;
    VertexBufferLayout::SharedPtr pBufferLayout = VertexBufferLayout::create();
    pBufferLayout->addElement(name, 0u, format, bufferSize, shaderLoc);
    VertexLayout::SharedPtr pLayout = VertexLayout::create();
    pLayout->addBufferLayout(0u, pBufferLayout);

    //create index buffer
    const uint32_t indexBufferSize = 12;
    uint32_t indexBufferData[indexBufferSize] = { 0, 1, 2, 2, 3, 4, 5, 6, 7, 6, 7, 8 };
    Buffer::SharedPtr pIndexBuffer = Buffer::create(indexBufferSize, Resource::BindFlags::Index, Buffer::CpuAccess::Read, indexBufferData);

    //Create and test vao
    Vao::Topology topology = Vao::Topology::TriangleStrip;
    Vao::SharedPtr pVao = Vao::create(topology, pLayout, bufferVec, pIndexBuffer, ResourceFormat::R32Uint);
    if ((pVao->getVertexBuffer(0u)->getBindFlags() & Resource::BindFlags::Vertex) == Resource::BindFlags::None ||
        pVao->getVertexBuffersCount() != 1 ||
        pVao->getIndexBuffer()->getSize() != indexBufferSize ||
        (pVao->getIndexBuffer()->getBindFlags() & Resource::BindFlags::Index) == Resource::BindFlags::None ||
        pVao->getPrimitiveTopology() != topology ||
        pVao->getVertexLayout()->getBufferCount() != 1 ||
        pVao->getVertexLayout()->getBufferLayout(0u)->getElementName(0u) != name ||
        pVao->getVertexLayout()->getBufferLayout(0u)->getElementFormat(0u) != format ||
        pVao->getVertexLayout()->getBufferLayout(0u)->getElementArraySize(0u) != bufferSize ||
        pVao->getVertexLayout()->getBufferLayout(0u)->getElementShaderLocation(0u) != shaderLoc)
    {
        return test_fail("Vao's properties do not match the properties that were used to create it");
    }

    return test_pass();
}

testing_func(VaoTest, TestMultiBufferCreate)
{
    //Create buffers
    const uint32_t bufferSize = 9u;
    float bufferData[bufferSize] = { 0.f, 0.25f, 0.5f, 0.75f, 1.f, 1.25f, 1.5f, 2.f, 2.25f };
    const uint32_t numBuffers = 10u;
    Vao::BufferVec bufferVec;
    for (uint32_t i = 0; i < numBuffers; ++i)
    {
        Buffer::SharedPtr pBuffer = Buffer::create(bufferSize, Resource::BindFlags::Vertex, Buffer::CpuAccess::Read, bufferData);
        bufferVec.push_back(pBuffer);
    }

    //Create Layout
    std::string name = VERTEX_POSITION_NAME;
    ResourceFormat format = ResourceFormat::RGB32Float;
    uint32_t shaderLoc = VERTEX_POSITION_LOC;
    VertexLayout::SharedPtr pLayout = VertexLayout::create();
    for (uint32_t i = 0; i < numBuffers; ++i)
    {
        VertexBufferLayout::SharedPtr pBufferLayout = VertexBufferLayout::create();
        pBufferLayout->addElement(name, 0u, format, bufferSize, shaderLoc);
        pLayout->addBufferLayout(i, pBufferLayout);
    }

    //Create VAO and check properties
    Vao::Topology topology = Vao::Topology::LineList;
    Vao::SharedPtr pVao = Vao::create(topology, pLayout, bufferVec);
    //check 'global' properties
    if (pVao->getVertexBuffersCount() != numBuffers || pVao->getIndexBuffer() != nullptr || 
        pVao->getVertexLayout()->getBufferCount() != numBuffers || pVao->getPrimitiveTopology() != topology)
    {
        return test_fail("Buffer count or index buffer of VAO is incorrect");
    }

    //check per buffer properties
    for (uint32_t i = 0; i < numBuffers; ++i)
    {
        if ((pVao->getVertexBuffer(i)->getBindFlags() & Resource::BindFlags::Vertex) == Resource::BindFlags::None ||
            pVao->getVertexLayout()->getBufferLayout(i)->getElementName(0u) != name ||
            pVao->getVertexLayout()->getBufferLayout(i)->getElementFormat(0u) != format ||
            pVao->getVertexLayout()->getBufferLayout(i)->getElementArraySize(0u) != bufferSize ||
            pVao->getVertexLayout()->getBufferLayout(i)->getElementShaderLocation(0u) != shaderLoc)
        {
            return test_fail("Vao's buffer properties do not match the properties that were used to create it");
        }
    }

    return test_pass();
}

testing_func(VaoTest, TestLayout)
{    
    //Create vertex buffer
    const uint32_t bufferSize = 15u;
    float bufferData[bufferSize] = { 0.f, 0.25f, 0.5f, 0.75f, 1.f, 1.25f, 1.5f, 2.f, 2.25f, 2.5f, 2.75f, 3.f, 3.25f, 3.5f, 3.75f };
    Buffer::SharedPtr pBuffer = Buffer::create(bufferSize, Resource::BindFlags::Vertex, Buffer::CpuAccess::Read, bufferData);
    Vao::BufferVec bufferVec;
    bufferVec.push_back(pBuffer);
    
    //Create Layout
    const uint32_t numElements = 4;
    std::string names[numElements] = { VERTEX_POSITION_NAME, VERTEX_NORMAL_NAME, VERTEX_BITANGENT_NAME, VERTEX_TEXCOORD_NAME };
    ResourceFormat formats[numElements] = { ResourceFormat::RGBA32Float, ResourceFormat::RGB32Float, ResourceFormat::RGB32Float, ResourceFormat::RG32Float };
    uint32_t shaderLocs[numElements] = { VERTEX_POSITION_LOC, VERTEX_NORMAL_LOC, VERTEX_BITANGENT_LOC, VERTEX_TEXCOORD_LOC };
    VertexLayout::SharedPtr pLayout = VertexLayout::create();
    VertexBufferLayout::SharedPtr pBufferLayout = VertexBufferLayout::create();
    uint32_t offset = 0u;
    for (uint32_t i = 0; i < numElements; ++i)
    {
        pBufferLayout->addElement(names[i], offset, formats[i], bufferSize, shaderLocs[i]);
        offset += getFormatBytesPerBlock(formats[i]);
    }
    pLayout->addBufferLayout(0u, pBufferLayout);

    //Create VAO and check properties
    Vao::Topology topology = Vao::Topology::TriangleList;
    Vao::SharedPtr pVao = Vao::create(topology, pLayout, bufferVec);
    //check 'global' properties
    if (pVao->getVertexBuffersCount() != 1 || pVao->getIndexBuffer() != nullptr || pVao->getVertexLayout()->getBufferCount() != 1 || 
        (pVao->getVertexBuffer(0u)->getBindFlags() & Resource::BindFlags::Vertex) == Resource::BindFlags::None || 
        pVao->getPrimitiveTopology() != topology)
    {
        return test_fail("Buffer count, index buffer, or buffer bind flag of VAO is incorrect");
    }

    //check buffer element properties 
    for (uint32_t i = 0; i < numElements; ++i)
    {
        if (pVao->getVertexLayout()->getBufferLayout(0u)->getElementName(i) != names[i] ||
            pVao->getVertexLayout()->getBufferLayout(0u)->getElementFormat(i) != formats[i] ||
            pVao->getVertexLayout()->getBufferLayout(0u)->getElementArraySize(i) != bufferSize ||
            pVao->getVertexLayout()->getBufferLayout(0u)->getElementShaderLocation(i) != shaderLocs[i])
        {
            return test_fail("VAO's Buffer layout is incorrect");
        }
    }

    return test_pass();
}

int main()
{
    VaoTest voat;
    voat.init(true);
    voat.run();
    return 0;
}