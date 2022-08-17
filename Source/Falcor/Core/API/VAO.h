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
#pragma once
#include "VertexLayout.h"
#include "Buffer.h"
#include "Core/Macros.h"
#include "Core/Assert.h"
#include <memory>
#include <vector>

namespace Falcor
{
    /** Abstracts vertex array objects. A VAO must at least specify a primitive topology. You may additionally specify a number of vertex buffer layouts
        corresponding to the number of vertex buffers to be bound. The number of vertex buffers to be bound must match the number described in the layout.
    */
    class FALCOR_API Vao
    {
    public:
        using SharedPtr = std::shared_ptr<Vao>;
        using WeakPtr = std::weak_ptr<Vao>;
        using SharedConstPtr = std::shared_ptr<const Vao>;
        ~Vao() = default;

        /** Primitive topology
        */
        enum class Topology
        {
            Undefined,
            PointList,
            LineList,
            LineStrip,
            TriangleList,
            TriangleStrip
        };

        struct ElementDesc
        {
            static const uint32_t kInvalidIndex = -1;
            uint32_t vbIndex = kInvalidIndex;
            uint32_t elementIndex = kInvalidIndex;
        };

        using BufferVec = std::vector<Buffer::SharedPtr>;

        /** Create a new vertex array object.
            \param primTopology The primitive topology.
            \param pLayout The vertex layout description. Can be nullptr.
            \param pVBs Array of pointers to vertex buffers. Number of buffers must match with pLayout.
            \param pIB Pointer to the index buffer. Can be nullptr, in which case no index buffer will be bound.
            \param ibFormat The resource format of the index buffer. Can be either R16Uint or R32Uint.
            \return New object, or throws an exception on error.
        */
        static SharedPtr create(Topology primTopology, const VertexLayout::SharedPtr& pLayout = nullptr, const BufferVec& pVBs = BufferVec(), const Buffer::SharedPtr& pIB = nullptr, ResourceFormat ibFormat = ResourceFormat::Unknown);

        /** Get the API handle
        */
        const VaoHandle& getApiHandle() const;

        /** Get the vertex buffer count
        */
        uint32_t getVertexBuffersCount() const { return (uint32_t)mpVBs.size(); }

        /** Get a vertex buffer
        */
        const Buffer::SharedPtr& getVertexBuffer(uint32_t index) const { FALCOR_ASSERT(index < (uint32_t)mpVBs.size()); return mpVBs[index]; }

        /** Get a vertex buffer layout
        */
        const VertexLayout::SharedPtr& getVertexLayout() const { return mpVertexLayout; }

        /** Return the vertex buffer index and the element index by its location.
            If the element is not found, returns the default ElementDesc
        */
        ElementDesc getElementIndexByLocation(uint32_t elementLocation) const;

        /** Get the index buffer
        */
        const Buffer::SharedPtr& getIndexBuffer() const { return mpIB; }

        /** Get the index buffer format
        */
        ResourceFormat getIndexBufferFormat() const { return mIbFormat; }

        /** Get the primitive topology
        */
        Topology getPrimitiveTopology() const { return mTopology; }

    protected:
        friend class RenderContext;

    private:
        Vao(const BufferVec& pVBs, const VertexLayout::SharedPtr& pLayout, const Buffer::SharedPtr& pIB, ResourceFormat ibFormat, Topology primTopology);

        VaoHandle mApiHandle;
        VertexLayout::SharedPtr mpVertexLayout;
        BufferVec mpVBs;
        Buffer::SharedPtr mpIB;
        void* mpPrivateData = nullptr;
        ResourceFormat mIbFormat;
        Topology mTopology;
    };
}
