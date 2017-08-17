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
        Vao::SharedPtr getFullscreenQuadVao()
        {
            static const glm::vec2 kVertices[] =
            {
                glm::vec2(-1,  1),
                glm::vec2(-1, -1),
                glm::vec2(1,  1),
                glm::vec2(1, -1),
            };

            const uint32_t vbSize = (uint32_t)(sizeof(glm::vec2)*arraysize(kVertices));
            Buffer::SharedPtr pVB = Buffer::create(vbSize, Buffer::BindFlags::Vertex, Buffer::CpuAccess::Write, (void*)kVertices);

            VertexLayout::SharedPtr pLayout = VertexLayout::create();
            VertexBufferLayout::SharedPtr pBufLayout = VertexBufferLayout::create();
            pBufLayout->addElement("POSITION", 0, ResourceFormat::RG32Float, 1, 0);
            pLayout->addBufferLayout(0, pBufLayout);

            Vao::BufferVec buffers{ pVB };
            return Vao::create(Vao::Topology::TriangleStrip, pLayout, buffers);
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
}
