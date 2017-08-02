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
#include "TestHelper.h"
#include "API/VertexLayout.h"

namespace Falcor
{
    namespace TestHelper
    {

#ifdef FALCOR_VK
#define ADJUST_Y(a) (-(a))
#else
#define ADJUST_Y(a) a
#endif

        //  Return a horizontal line of 4 points along the middle.
        Vao::SharedPtr getBasicPointsVao()
        {

            static const vec2 kVertices[] =
            {
                glm::vec2(0.0, ADJUST_Y(0.0)),
            };

            const uint32_t vbSize = (uint32_t)(sizeof(glm::vec2)*arraysize(kVertices));
            Buffer::SharedPtr pVB = Buffer::create(vbSize, Buffer::BindFlags::Vertex, Buffer::CpuAccess::Write, (void*)kVertices);

            VertexLayout::SharedPtr pLayout = VertexLayout::create();
            VertexBufferLayout::SharedPtr pBufLayout = VertexBufferLayout::create();
            pBufLayout->addElement("POSITION", 0, ResourceFormat::RG32Float, 1, 0);
            pLayout->addBufferLayout(0, pBufLayout);

            Vao::BufferVec buffers{ pVB };
            return Vao::create(buffers, pLayout, nullptr, ResourceFormat::Unknown, Vao::Topology::PointList);

        }


        //  Two horizontal lines.
        Vao::SharedPtr getBasicLinesListVao()
        {

            static const vec2 kVertices[] =
            {
                glm::vec2(-1.0, ADJUST_Y(-0.5)),
                glm::vec2(1.0, ADJUST_Y(-0.5)),
                glm::vec2(-1.0, ADJUST_Y(0.5)),
                glm::vec2(1.0, ADJUST_Y(0.5))
            };

            const uint32_t vbSize = (uint32_t)(sizeof(glm::vec2)*arraysize(kVertices));
            Buffer::SharedPtr pVB = Buffer::create(vbSize, Buffer::BindFlags::Vertex, Buffer::CpuAccess::Write, (void*)kVertices);

            VertexLayout::SharedPtr pLayout = VertexLayout::create();
            VertexBufferLayout::SharedPtr pBufLayout = VertexBufferLayout::create();
            pBufLayout->addElement("POSITION", 0, ResourceFormat::RG32Float, 1, 0);
            pLayout->addBufferLayout(0, pBufLayout);

            Vao::BufferVec buffers{ pVB };
            return Vao::create(buffers, pLayout, nullptr, ResourceFormat::Unknown, Vao::Topology::LineList);

        }

        //  Return a horizontal line along the middle.
        Vao::SharedPtr getBasicLineStripVAO()
        {

            static const vec2 kVertices[] =
            {
                glm::vec2(-1.0, ADJUST_Y(0.0)),
                glm::vec2(0.0, ADJUST_Y(0.0)),
                glm::vec2(1.0, ADJUST_Y(0.0))
            };

            const uint32_t vbSize = (uint32_t)(sizeof(glm::vec2)*arraysize(kVertices));
            Buffer::SharedPtr pVB = Buffer::create(vbSize, Buffer::BindFlags::Vertex, Buffer::CpuAccess::Write, (void*)kVertices);

            VertexLayout::SharedPtr pLayout = VertexLayout::create();
            VertexBufferLayout::SharedPtr pBufLayout = VertexBufferLayout::create();
            pBufLayout->addElement("POSITION", 0, ResourceFormat::RG32Float, 1, 0);
            pLayout->addBufferLayout(0, pBufLayout);

            Vao::BufferVec buffers{ pVB };
            return Vao::create(buffers, pLayout, nullptr, ResourceFormat::Unknown, Vao::Topology::LineStrip);

        }

        //  
        Vao::SharedPtr getBasicTriangleListVao()
        {
            static const vec2 kVertices[] =
            {
                glm::vec2(1.0, ADJUST_Y(-0.8)),
                glm::vec2(0.0, ADJUST_Y(0.8)),
                glm::vec2(-1.0, ADJUST_Y(-0.8))
            };

            const uint32_t vbSize = (uint32_t)(sizeof(glm::vec2)*arraysize(kVertices));
            Buffer::SharedPtr pVB = Buffer::create(vbSize, Buffer::BindFlags::Vertex, Buffer::CpuAccess::Write, (void*)kVertices);

            VertexLayout::SharedPtr pLayout = VertexLayout::create();
            VertexBufferLayout::SharedPtr pBufLayout = VertexBufferLayout::create();
            pBufLayout->addElement("POSITION", 0, ResourceFormat::RG32Float, 1, 0);
            pLayout->addBufferLayout(0, pBufLayout);

            uint32_t indexBufferArray[] = { 0, 1, 2 };
            const uint32_t ibSize = (uint32_t)(sizeof(uint32_t) * arraysize(indexBufferArray));
            Buffer::SharedPtr pIB = Buffer::create(ibSize, Buffer::BindFlags::Index, Buffer::CpuAccess::Write, (void*)indexBufferArray);

            Vao::BufferVec buffers{ pVB };
            return Vao::create(buffers, pLayout, pIB, ResourceFormat::R32Uint, Vao::Topology::TriangleList);

        }



        //  Return the Fullscreen Quad Vao.
        Vao::SharedPtr getFullscreenQuadVao()
        {

            static const vec2 kVertices[] =
            {
                glm::vec2(-1, ADJUST_Y(1)),
                glm::vec2(-1, ADJUST_Y(-1)),
                glm::vec2(1, ADJUST_Y(1)),
                glm::vec2(1, ADJUST_Y(-1)) 
            };


            const uint32_t vbSize = (uint32_t)(sizeof(glm::vec2)*arraysize(kVertices));
            Buffer::SharedPtr pVB = Buffer::create(vbSize, Buffer::BindFlags::Vertex, Buffer::CpuAccess::Write, (void*)kVertices);

            VertexLayout::SharedPtr pLayout = VertexLayout::create();
            VertexBufferLayout::SharedPtr pBufLayout = VertexBufferLayout::create();
            pBufLayout->addElement("POSITION", 0, ResourceFormat::RG32Float, 1, 0);
            pLayout->addBufferLayout(0, pBufLayout);

            Vao::BufferVec buffers{ pVB };
            return Vao::create(buffers, pLayout, nullptr, ResourceFormat::Unknown, Vao::Topology::TriangleStrip);
        }


        //  Return the Reversed Fullscreend Quad Vao.
        Vao::SharedPtr getReversedFullscreenQuadVao()
        {

            static const vec2 kVertices[] =
            {
                glm::vec2(-1, ADJUST_Y(1)),
                glm::vec2(1, ADJUST_Y(1)),
                glm::vec2(-1, ADJUST_Y(-1)),
                glm::vec2(1, ADJUST_Y(-1))
            };


            const uint32_t vbSize = (uint32_t)(sizeof(glm::vec2)*arraysize(kVertices));
            Buffer::SharedPtr pVB = Buffer::create(vbSize, Buffer::BindFlags::Vertex, Buffer::CpuAccess::Write, (void*)kVertices);

            VertexLayout::SharedPtr pLayout = VertexLayout::create();
            VertexBufferLayout::SharedPtr pBufLayout = VertexBufferLayout::create();
            pBufLayout->addElement("POSITION", 0, ResourceFormat::RG32Float, 1, 0);
            pLayout->addBufferLayout(0, pBufLayout);

            Vao::BufferVec buffers{ pVB };
            return Vao::create(buffers, pLayout, nullptr, ResourceFormat::Unknown, Vao::Topology::TriangleStrip);
        }





        GraphicsState::SharedPtr getOnePixelState(RenderContext* pCtx)
        {
            GraphicsProgram::SharedPtr program = GraphicsProgram::createFromFile("BlendTest.vs.hlsl", "BlendTest.ps.hlsl");
            Vao::SharedPtr vao = TestHelper::getFullscreenQuadVao();
            Fbo::Desc fboDesc;
            fboDesc.setColorTarget(0, ResourceFormat::RGBA32Float);
            Fbo::SharedPtr testFbo = FboHelper::create2D(1, 1, fboDesc);

            GraphicsState::SharedPtr pState = GraphicsState::create();
            pState->setProgram(program);
            pState->setVao(vao);
            pState->setFbo(testFbo);
            return pState;
        }


        //  Create a Texture.
        Texture::SharedPtr createRGBA32FRWTexture(uint32_t newWidth, uint32_t newHeight)
        {
            //  Return the Texture 2D.
            return Texture::create2D(newWidth, newHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
        }

        float randFloatZeroToOne()
        {
            return static_cast<float>(rand()) / RAND_MAX;
        }

        vec4 randVec4ZeroToOne()
        {
            return vec4(randFloatZeroToOne(), randFloatZeroToOne(), randFloatZeroToOne(), randFloatZeroToOne());
        }

        bool nearCompare(const float lhs, const float rhs)
        {
            const float ep = 0.0001f;
            return abs(lhs - rhs) < ep;
        }

        bool nearVec4(const vec4& lhs, const vec4& rhs)
        {
            return nearCompare(lhs.x, rhs.x) && nearCompare(lhs.y, rhs.y) && nearCompare(lhs.z, rhs.z) && nearCompare(lhs.w, rhs.w);
        }
    }

#undef  ADJUST_Y

}
