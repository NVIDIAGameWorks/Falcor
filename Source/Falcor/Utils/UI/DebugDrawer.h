/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "Core/API/VAO.h"

namespace Falcor
{
    struct BoundingBox;
    class RenderContext;
    class Camera;
    class GraphicsState;
    class GraphicsVars;

    /** Utility class to assist in drawing debug geometry
    */
    class dlldecl DebugDrawer
    {
    public:

        using SharedPtr = std::shared_ptr<DebugDrawer>;
        using SharedConstPtr = std::shared_ptr<const DebugDrawer>;

        static const uint32_t kMaxVertices = 10000;     ///< Default max number of vertices per DebugDrawer instance
        static const uint32_t kPathDetail = 10;         ///< Segments between keyframes

        using Quad = std::array<float3, 4>;

        /** Create a new object for drawing debug geometry.
            \param[in] maxVertices Maximum number of vertices that will be drawn.
            \return New object, or throws an exception if creation failed.
        */
        static SharedPtr create(uint32_t maxVertices = kMaxVertices);

        /** Sets the color for following geometry
        */
        void setColor(const float3& color) { mCurrentColor = color; }

        /** Adds a line segment
        */
        void addLine(const float3& a, const float3& b);

        /** Adds a quad described by four corner points
        */
        void addQuad(const Quad& quad);

        /** Adds a world space AABB
        */
        void addBoundingBox(const BoundingBox& aabb);

        /** Renders the contents of the debug drawer
        */
        void render(RenderContext* pContext, GraphicsState* pState, GraphicsVars* pVars, Camera *pCamera);

        /** Get how many vertices are currently pushed
        */
        uint32_t getVertexCount() const { return (uint32_t)mVertexData.size(); }

        /** Get the Vao of vertex data
        */
        const Vao::SharedPtr& getVao() const { return mpVao; };

        /** Clears vertices
        */
        void clear() { mVertexData.clear(); mDirty = true; }

    private:

        void uploadBuffer();

        DebugDrawer(uint32_t maxVertices);

        float3 mCurrentColor;

        struct LineVertex
        {
            float3 position;
            float3 color;
        };

        Vao::SharedPtr mpVao;
        std::vector<LineVertex> mVertexData;
        bool mDirty = true;
    };
}
