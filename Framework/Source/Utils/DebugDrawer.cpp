/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
#include "Framework.h"
#include "Utils/DebugDrawer.h"
#include "API/RenderContext.h"
#include "Graphics/Camera/Camera.h"
#include "Utils/AABB.h"
#include <array>
#include "glm/gtc/constants.hpp"

namespace Falcor
{
    DebugDrawer::SharedPtr DebugDrawer::create(uint32_t maxVertices)
    {
        return SharedPtr(new DebugDrawer(maxVertices));
    }

    void DebugDrawer::addLine(const glm::vec3& a, const glm::vec3& b)
    {
        if (mVertexData.capacity() - mVertexData.size() >= 2)
        {
            mVertexData.push_back({a, mCurrentColor});
            mVertexData.push_back({b, mCurrentColor});
            mDirty = true;
        }
    }

    void DebugDrawer::addQuad(const Quad& quad)
    {
        addLine(quad[0], quad[1]);
        addLine(quad[1], quad[2]);
        addLine(quad[2], quad[3]);
        addLine(quad[3], quad[0]);
    }

    void DebugDrawer::addBoundingBox(const BoundingBox& aabb)
    {
        glm::vec3 min = aabb.center - aabb.extent;
        glm::vec3 max = aabb.center + aabb.extent;

        Quad bottomFace = { min, glm::vec3(max.x, min.y, min.z), glm::vec3(max.x, min.y, max.z), glm::vec3(min.x, min.y, max.z) };
        addQuad(bottomFace);

        Quad topFace = { glm::vec3(min.x, max.y, min.z), glm::vec3(max.x, max.y, min.z), max, glm::vec3(min.x, max.y, max.z) };
        addQuad(topFace);

        addLine(bottomFace[0], topFace[0]);
        addLine(bottomFace[1], topFace[1]);
        addLine(bottomFace[2], topFace[2]);
        addLine(bottomFace[3], topFace[3]);
    }

    DebugDrawer::Quad buildQuad(const glm::vec3& center, const glm::vec3& up, const glm::vec3& right)
    {
        // Length of each quad side
        static const float size = 0.08f;

        // Half widths based on size constant
        glm::vec3 upOffset = glm::normalize(up) * size / 2.0f;
        glm::vec3 rightOffset = glm::normalize(right) * size / 2.0f;

        // CCW from top left
        DebugDrawer::Quad quad;
        quad[0] = center + upOffset - rightOffset; // Top left
        quad[1] = center - upOffset - rightOffset; // Bottom left
        quad[2] = center - upOffset + rightOffset; // Bottom right
        quad[3] = center + upOffset + rightOffset; // Top right
        return quad;
    }

    // Generates a quad centered at currFrame's position facing nextFrame's position
    DebugDrawer::Quad createQuadForFrame(const ObjectPath::Frame& currFrame, const ObjectPath::Frame& nextFrame)
    {
        glm::vec3 forward = nextFrame.position - currFrame.position;
        glm::vec3 right = glm::cross(forward, currFrame.up);
        glm::vec3 up = glm::cross(right, forward);

        return buildQuad(currFrame.position, up, right);
    }

    // Generates a quad centered at currFrame's position oriented halfway between direction to prevFrame and direction to nextFrame
    DebugDrawer::Quad createQuadForFrame(const ObjectPath::Frame& prevFrame, const ObjectPath::Frame& currFrame, const ObjectPath::Frame& nextFrame)
    {
        glm::vec3 lastToCurrFoward = currFrame.position - prevFrame.position;
        glm::vec3 lastToCurrRight = glm::normalize(glm::cross(lastToCurrFoward, prevFrame.up));
        glm::vec3 lastToCurrUp = glm::normalize(glm::cross(lastToCurrRight, lastToCurrFoward));

        glm::vec3 currToNextFoward = nextFrame.position - currFrame.position;

        // If curr and next are the same, use the direction from prev to curr
        if (glm::length(currToNextFoward) < 0.001f)
        {
            currToNextFoward = lastToCurrFoward;
        }

        glm::vec3 currToNextRight = glm::normalize(glm::cross(currToNextFoward, currFrame.up));
        glm::vec3 currToNextUp = glm::normalize(glm::cross(currToNextRight, currToNextFoward));

        // Half vector between two direction normals
        glm::vec3 midUp = (lastToCurrUp + currToNextUp) / 2.0f;
        glm::vec3 midRight = (lastToCurrRight + currToNextRight) / 2.0f;

        return buildQuad(currFrame.position, midUp, midRight);
    }

    void DebugDrawer::addPath(const ObjectPath::SharedPtr& pPath)
    {
        // If a path has one or no keyframes, there's no path to draw
        if (pPath->getKeyFrameCount() <= 1)
        {
            return;
        }

        const float step = 1.0f / (float)kPathDetail;
        const float epsilon = 1.0e-6f; // A bit more than glm::epsilon

        ObjectPath::Frame prevFrame;
        pPath->getFrameAt(0, 0.0f, prevFrame);

        ObjectPath::Frame currFrame;
        pPath->getFrameAt(0, step, currFrame);

        Quad lastQuad = createQuadForFrame(prevFrame, currFrame);
        Quad currQuad;

        // Draw quad to cap path beginning
        addQuad(lastQuad);

        const float maxFrameIndex = (float)(pPath->getKeyFrameCount() - 1);

        // Add epsilon so loop's <= works properly
        const float pathEnd = maxFrameIndex + epsilon;

        for (float frame = step; frame <= pathEnd; frame += step)
        {
            // Loop can overshoot the max index
            // Clamp frame to right below max index so interpolation on the path will work
            frame = std::min(frame, maxFrameIndex - epsilon);

            uint32_t frameID = (uint32_t)(glm::floor(frame));
            float t = frame - (float)frameID;

            ObjectPath::Frame nextFrame;
            pPath->getFrameAt(frameID, t + step, nextFrame);
            currQuad = createQuadForFrame(prevFrame, currFrame, nextFrame);

            // Draw current quad
            addQuad(currQuad);

            // Connect last quad to current
            addLine(lastQuad[0], currQuad[0]);
            addLine(lastQuad[1], currQuad[1]);
            addLine(lastQuad[2], currQuad[2]);
            addLine(lastQuad[3], currQuad[3]);

            prevFrame = currFrame;
            lastQuad = currQuad;
            currFrame = nextFrame;
        }
    }

    void DebugDrawer::render(RenderContext* pContext, Camera *pCamera)
    {
        ConstantBuffer* pCB = pContext->getGraphicsVars()->getConstantBuffer("InternalPerFrameCB").get();
        if (pCB != nullptr)
        {
            pCamera->setIntoConstantBuffer(pCB, 0);
        }

        uploadBuffer();
        pContext->getGraphicsState()->setVao(mpVao);

        pContext->draw((uint32_t)mVertexData.size(), 0);
    }

    void DebugDrawer::uploadBuffer()
    {
        if (mDirty)
        {
            auto pVertexBuffer = mpVao->getVertexBuffer(0);
            pVertexBuffer->updateData(mVertexData.data(), 0, sizeof(LineVertex) * mVertexData.size());
            mDirty = false;
        }
    }

    DebugDrawer::DebugDrawer(uint32_t maxVertices)
    {
        Buffer::SharedPtr pVertexBuffer = Buffer::create(sizeof(LineVertex) * maxVertices, Resource::BindFlags::Vertex, Buffer::CpuAccess::Write, nullptr);

        VertexBufferLayout::SharedPtr pBufferLayout = VertexBufferLayout::create();
        pBufferLayout->addElement("POSITION", 0, ResourceFormat::RGB32Float, 1, 0);
        pBufferLayout->addElement("COLOR", sizeof(glm::vec3), ResourceFormat::RGB32Float, 1, 1);

        VertexLayout::SharedPtr pVertexLayout = VertexLayout::create();
        pVertexLayout->addBufferLayout(0, pBufferLayout);

        mpVao = Vao::create(Vao::Topology::LineList, pVertexLayout, { pVertexBuffer });

        mVertexData.resize(maxVertices);
    }

}
