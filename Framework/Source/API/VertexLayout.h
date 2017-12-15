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
#pragma once
#include <vector>
#include "Formats.h"
#include "Data/VertexAttrib.h"
#include "Graphics/Program/Program.h"

namespace Falcor
{
    class Program;

    /** Describes the layout of a vertex buffer that will be bound to a render operation as part of a VAO.
    */
    class VertexBufferLayout : public std::enable_shared_from_this<VertexBufferLayout>
    {
    public:
        using SharedPtr = std::shared_ptr<VertexBufferLayout>;
        using SharedConstPtr = std::shared_ptr<const VertexBufferLayout>;

        enum class InputClass
        {
            PerVertexData,      ///< Buffer elements will represent per-vertex data
            PerInstanceData     ///< Buffer elements will represent per-instance data
        };
        
        static SharedPtr create()
        {
            return SharedPtr(new VertexBufferLayout());
        }

        /** Add a new element to the layout.
            \param name The semantic name of the element. In OpenGL this is just a descriptive field. In DX, this is the semantic name used to match the element with the shader input signature.
            \param offset Offset in bytes of the element from the start of the vertex.
            \param format The format of each channel in the element.
            \param arraySize The array size of the input element. Must be at least 1.
            \param shaderLocation The attribute binding location in the shader.
        */
        void addElement(const std::string& name, uint32_t offset, ResourceFormat format, uint32_t arraySize, uint32_t shaderLocation)
        {
            Element Elem;
            Elem.offset = offset;
            Elem.format = format;
            Elem.shaderLocation = shaderLocation;
            Elem.name = name;
            Elem.arraySize = arraySize;
            mElements.push_back(Elem);
            mVertexStride += getFormatBytesPerBlock(Elem.format) * Elem.arraySize;
        }

        /** Return the element offset pointed to by Index
        */
        uint32_t getElementOffset(uint32_t index) const
        {
            return mElements[index].offset;
        }

        /** Return the element format pointed to by Index
        */
        ResourceFormat getElementFormat(uint32_t index) const
        {
            return mElements[index].format;
        }

        /** Return the semantic name of the element
        */
        const std::string& getElementName(uint32_t index) const
        {
            return mElements[index].name;
        }

        /** Return the array size the element
        */
        const uint32_t getElementArraySize(uint32_t index) const
        {
            return mElements[index].arraySize;
        }

        /** Return the element shader binding location pointed to by Index
        */
        uint32_t getElementShaderLocation(uint32_t index) const
        {
            return mElements[index].shaderLocation;
        }

        /** Return the number of elements in the object
        */
        uint32_t getElementCount() const
        {
            return (uint32_t)mElements.size();
        }
        
        /** Return the total stride of all elements in bytes
        */
        uint32_t getStride() const { return mVertexStride; }

        /** Return the input classification
        */
        InputClass getInputClass() const
        {
            return mClass;
        }

        /** Returns the per-instance data step rate
        */
        uint32_t getInstanceStepRate() const
        {
            return mInstanceStepRate;
        }

        /** Set the input class and the data step rate
            \param inputClass Specifies is this layout object holds per-vertex or per-instance data
            \param instanceStepRate For per-instance data, specifies how many instance to draw using the same per-instance data. If this is zero, it behaves as if the class is PerVertexData
        */
        void setInputClass(InputClass inputClass, uint32_t stepRate) { mClass = inputClass; mInstanceStepRate = stepRate; }

        static const uint32_t kInvalidShaderLocation = uint32_t(-1);
    private:
        VertexBufferLayout() = default;

        struct Element
        {
            uint32_t offset = 0;
            ResourceFormat format = ResourceFormat::Unknown;
            uint32_t shaderLocation = kInvalidShaderLocation;
            std::string name;
            uint32_t arraySize;
            uint32_t vbIndex;
        };

        std::vector<Element> mElements;
        InputClass mClass = InputClass::PerVertexData;
        uint32_t mInstanceStepRate = 0;
        uint32_t mVertexStride = 0;
    };

    /** Container to hold layouts for every vertex layout that will be bound at once to a VAO.
    */
    class VertexLayout : public std::enable_shared_from_this<VertexLayout>
    {
    public:
        using SharedPtr = std::shared_ptr<VertexLayout>;
        using SharedConstPtr = std::shared_ptr<const VertexLayout>;

        static SharedPtr create() { return SharedPtr(new VertexLayout()); }

        /** Add a layout description for a buffer.
        */
        void addBufferLayout(uint32_t index, VertexBufferLayout::SharedConstPtr pLayout)
        {
            if (mpBufferLayouts.size() <= index)
            {
                mpBufferLayouts.resize(index + 1);
            }
            mpBufferLayouts[index] = pLayout;
        }

        /** Get a buffer layout.
        */
        const VertexBufferLayout::SharedConstPtr& getBufferLayout(size_t index) const
        {
            return mpBufferLayouts[index];
        }

        /** Get how many buffer descriptions there are.
        */
        size_t getBufferCount() const { return mpBufferLayouts.size(); }

        /** Add defines to notify a shader program about vertex what input properties have associated buffer data
        */
        void addVertexAttribDclToProg(Program* pProg) const
        {
            pProg->removeDefine("HAS_NORMAL");
            pProg->removeDefine("HAS_BITANGENT");
            pProg->removeDefine("HAS_TEXCRD");
            pProg->removeDefine("HAS_COLORS");
            pProg->removeDefine("HAS_LIGHTMAP_UV");

            for (const auto& l : mpBufferLayouts)
            {
                if(l)
                {
                    for (uint32_t i = 0; i < l->getElementCount(); i++)
                    {
                        if (l->getElementShaderLocation(i) == VERTEX_NORMAL_LOC)
                        {
                            pProg->addDefine("HAS_NORMAL");
                        }
                        if (l->getElementShaderLocation(i) == VERTEX_BITANGENT_LOC)
                        {
                            pProg->addDefine("HAS_BITANGENT");
                        }
                        if (l->getElementShaderLocation(i) == VERTEX_TEXCOORD_LOC)
                        {
                            pProg->addDefine("HAS_TEXCRD");
                        }
                        if (l->getElementShaderLocation(i) == VERTEX_LIGHTMAP_UV_LOC)
                        {
                            pProg->addDefine("HAS_LIGHTMAP_UV");
                        }
                        if (l->getElementShaderLocation(i) == VERTEX_DIFFUSE_COLOR_LOC)
                        {
                            pProg->addDefine("HAS_COLORS");
                        }
                    }
                }
            }
        }

    private:
        VertexLayout() { mpBufferLayouts.reserve(16); }
        std::vector<VertexBufferLayout::SharedConstPtr> mpBufferLayouts;
    };
}