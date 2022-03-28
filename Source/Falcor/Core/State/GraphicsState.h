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
#include "Core/API/GraphicsStateObject.h"
#include "StateGraph.h"
#include "Core/API/FBO.h"
#include "Core/Program/GraphicsProgram.h"
#include "Core/Program/ProgramVars.h"

namespace Falcor
{
    class GraphicsVars;
    class Vao;
    class GraphicsProgram;

    /** Pipeline state.
        This class contains the entire state required by a single draw-call. It's not an immutable object - you can change it dynamically during rendering.
        The recommended way to use it is to create multiple PipelineState objects (ideally, a single object per render-pass)
    */
    class FALCOR_API GraphicsState
    {
    public:
        using SharedPtr = std::shared_ptr<GraphicsState>;
        using SharedConstPtr = std::shared_ptr<const GraphicsState>;
        ~GraphicsState();

        /** Defines the region to render to.
        */
        struct Viewport
        {
            Viewport() = default;
            Viewport(float x, float y, float w, float h, float minZ, float maxZ)
                : originX(x), originY(y), width(w), height(h), minDepth(minZ), maxDepth(maxZ) {}

            float originX = 0;      ///< Top left X position
            float originY = 0;      ///< Top left Y position
            float width = 1.0f;     ///< Viewport width.
            float height = 1.0f;    ///< Viewport height.
            float minDepth = 0;     ///< Minimum depth (0-1)
            float maxDepth = 1;     ///< Maximum depth (0-1)
        };

        /** Defines a region to clip render results to.
        */
        struct Scissor
        {
            Scissor() = default;
            Scissor(int32_t l, int32_t t, int32_t r, int32_t b)
                : left(l), top(t), right(r), bottom(b) {}

            int32_t left = 0;
            int32_t top = 0;
            int32_t right = 0;
            int32_t bottom = 0;
        };

        /** Create a new state object.
            \return A new object, or an exception is thrown if creation failed.
        */
        static SharedPtr create() { return SharedPtr(new GraphicsState()); }

        /** Copy constructor. Useful if you need to make minor changes to an already existing object
        */
        SharedPtr operator=(const SharedPtr& other);

        /** Get current FBO.
        */
        Fbo::SharedPtr getFbo() const { return mpFbo; }

        /** Set an FBO. This function doesn't store the current FBO state.
            \param[in] pFbo An FBO object. If nullptr is used, will detach the current FBO
            \param[in] setVp0Sc0 If true, will set viewport 0 and scissor 0 to match the FBO dimensions
        */
        GraphicsState& setFbo(const Fbo::SharedPtr& pFbo, bool setVp0Sc0 = true);

        /** Set a new FBO and store the current FBO into a stack. Useful for multi-pass effects.
            \param[in] pFbo - a new FBO object. If nullptr is used, will bind an empty framebuffer object
            \param[in] setVp0Sc0 If true, viewport 0 and scissor 0 will be set to match the FBO dimensions
        */
        void pushFbo(const Fbo::SharedPtr& pFbo, bool setVp0Sc0 = true);

        /** Restore the last FBO pushed into the FBO stack. If the stack is empty, an error will be logged.
            \param[in] setVp0Sc0 If true, viewport 0 and scissor 0 will be set to match the FBO dimensions
        */
        void popFbo(bool setVp0Sc0 = true);

        /** Set a new vertex array object. By default, no VAO is bound.
            \param[in] pVao The Vao object to bind. If this is nullptr, will unbind the current VAO.
        */
        GraphicsState& setVao(const Vao::SharedConstPtr& pVao);

        /** Get the currently bound VAO.
        */
        Vao::SharedConstPtr getVao() const { return mpVao; }

        /** Set the stencil reference value.
        */
        GraphicsState& setStencilRef(uint8_t refValue) { mStencilRef = refValue; return *this; }

        /** Get the current stencil reference value.
        */
        uint8_t getStencilRef() const { return mStencilRef; }

        /** Set a viewport.
            \param[in] index Viewport index
            \param[in] vp Viewport to set
            \param[in] setScissors If true, corresponding scissor will be set to the same dimensions
        */
        void setViewport(uint32_t index, const Viewport& vp, bool setScissors = true);

        /** Get a viewport.
            \param[in] index Viewport index
        */
        const Viewport& getViewport(uint32_t index) const { return mViewports[index]; }

        /** Get the array of all the current viewports.
        */
        const std::vector<Viewport>& getViewports() const { return mViewports; }

        /** Push the current viewport and sets a new one
            \param[in] index Viewport index
            \param[in] vp Viewport to set
            \param[in] setScissors If true, corresponding scissor will be set to the same dimensions
        */
        void pushViewport(uint32_t index, const Viewport& vp, bool setScissors = true);

        /** Pops the last viewport from the stack and sets it
            \param[in] index Viewport index
            \param[in] setScissors If true, corresponding scissor will be set to the same dimensions
        */
        void popViewport(uint32_t index, bool setScissors = true);

        /** Set a scissor.
            \param[in] index Scissor index
            \param[in] sc Scissor to set
        */
        void setScissors(uint32_t index, const Scissor& sc);

        /** Get a Scissor.
            \param[in] index Scissor index
        */
        const Scissor& getScissors(uint32_t index) const { return mScissors[index]; }

        /** Get the array of all the current scissors.
        */
        const std::vector<Scissor>& getScissors() const { return mScissors; }

        /** Push a current Scissor and sets a new one
            \param[in] index Scissor index
            \param[in] sc Scissor to push
        */
        void pushScissors(uint32_t index, const Scissor& sc);

        /** Pops the last Scissor from a stack and sets it
            \param[in] index Scissor index
        */
        void popScissors(uint32_t index);

        /** Bind a program to the pipeline.
        */
        GraphicsState& setProgram(const GraphicsProgram::SharedPtr& pProgram) { FALCOR_ASSERT(pProgram); mpProgram = pProgram; return *this; }

        /** Get the currently bound program.
        */
        GraphicsProgram::SharedPtr getProgram() const { return mpProgram; }

        /** Set a blend-state.
        */
        GraphicsState& setBlendState(BlendState::SharedPtr pBlendState);

        /** Get the currently bound blend-state.
        */
        BlendState::SharedPtr getBlendState() const { return mDesc.getBlendState(); }

        /** Set a rasterizer-state.
        */
        GraphicsState& setRasterizerState(RasterizerState::SharedPtr pRasterizerState);

        /** Get the currently bound rasterizer-state.
        */
        RasterizerState::SharedPtr getRasterizerState() const { return mDesc.getRasterizerState(); }

        /** Set a depth-stencil state.
        */
        GraphicsState& setDepthStencilState(DepthStencilState::SharedPtr pDepthStencilState);

        /** Get the currently bound depth-stencil state.
        */
        DepthStencilState::SharedPtr getDepthStencilState() const { return mDesc.getDepthStencilState(); }

        /** Set the sample mask.
        */
        GraphicsState& setSampleMask(uint32_t sampleMask);

        /** Get the current sample mask.
        */
        uint32_t getSampleMask() const { return mDesc.getSampleMask(); }

        /** Get the active graphics state object.
        */
        virtual GraphicsStateObject::SharedPtr getGSO(const GraphicsVars* pVars);

        /** Get the desc
        */
        const GraphicsStateObject::Desc& getDesc() const { return mDesc; }

    private:
        GraphicsState();
        Vao::SharedConstPtr mpVao;
        Fbo::SharedPtr mpFbo;
        GraphicsProgram::SharedPtr mpProgram;
        GraphicsStateObject::Desc mDesc;
        uint8_t mStencilRef = 0;
        std::vector<Viewport> mViewports;
        std::vector<Scissor> mScissors;

        std::stack<Fbo::SharedPtr> mFboStack;
        std::vector<std::stack<Viewport>> mVpStack;
        std::vector<std::stack<Scissor>> mScStack;

        struct CachedData
        {
            const ProgramKernels* pProgramKernels = nullptr;
            const Fbo::Desc* pFboDesc = nullptr;
        };
        CachedData mCachedData;

        using StateGraph = StateGraph<GraphicsStateObject::SharedPtr, void*>;
        StateGraph::SharedPtr mpGsoGraph;
    };
}
