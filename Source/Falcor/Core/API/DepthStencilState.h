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
#include "Common.h"
#include "Handles.h"
#include "Core/Macros.h"
#include <memory>

namespace Falcor
{
    /** Depth-Stencil state
    */
    class FALCOR_API DepthStencilState
    {
    public:
        using SharedPtr = std::shared_ptr<DepthStencilState>;
        using SharedConstPtr = std::shared_ptr<const DepthStencilState>;

        /** Used for stencil control.
        */
        enum class Face
        {
            Front,          ///< Front-facing primitives
            Back,           ///< Back-facing primitives
            FrontAndBack    ///< Front and back-facing primitives
        };

        /** Comparison function
        */
        using Func = ComparisonFunc;

        /** Stencil operation
        */
        enum class StencilOp
        {
            Keep,               ///< Keep the stencil value
            Zero,               ///< Set the stencil value to zero
            Replace,            ///< Replace the stencil value with the reference value
            Increase,           ///< Increase the stencil value by one, wrap if necessary
            IncreaseSaturate,   ///< Increase the stencil value by one, clamp if necessary
            Decrease,           ///< Decrease the stencil value by one, wrap if necessary
            DecreaseSaturate,   ///< Decrease the stencil value by one, clamp if necessary
            Invert              ///< Invert the stencil data (bitwise not)
        };

        /** Stencil descriptor
        */
        struct StencilDesc
        {
            Func func = Func::Disabled;                     ///< Stencil comparison function
            StencilOp stencilFailOp = StencilOp::Keep;      ///< Stencil operation in case stencil test fails
            StencilOp depthFailOp = StencilOp::Keep;        ///< Stencil operation in case stencil test passes but depth test fails
            StencilOp depthStencilPassOp = StencilOp::Keep; ///< Stencil operation in case stencil and depth tests pass
        };

        /** Depth-stencil descriptor
        */
        class FALCOR_API Desc
        {
        public:
            friend class DepthStencilState;

            /** Enable/disable depth-test
            */
            Desc& setDepthEnabled(bool enabled) { mDepthEnabled = enabled; return *this; }

            /** Set the depth-function
            */
            Desc& setDepthFunc(Func depthFunc) { mDepthFunc = depthFunc; return *this; }

            /** Enable or disable depth writes into the depth buffer
            */
            Desc& setDepthWriteMask(bool writeDepth) { mWriteDepth = writeDepth; return *this; }

            /** Enable/disable stencil-test
            */
            Desc& setStencilEnabled(bool enabled) { mStencilEnabled = enabled; return *this; }

            /** Set the stencil write-mask
            */
            Desc& setStencilWriteMask(uint8_t mask);

            /** Set the stencil read-mask
            */
            Desc& setStencilReadMask(uint8_t mask);

            /** Set the stencil comparison function
                \param face Chooses the face to apply the function to
                \param func Comparison function
            */
            Desc& setStencilFunc(Face face, Func func);

            /** Set the stencil operation
                \param face Chooses the face to apply the operation to
                \param stencilFail Stencil operation in case stencil test fails
                \param depthFail Stencil operation in case stencil test passes but depth test fails
                \param depthStencilPass Stencil operation in case stencil and depth tests pass
            */
            Desc& setStencilOp(Face face, StencilOp stencilFail, StencilOp depthFail, StencilOp depthStencilPass);

            /** Set the stencil ref value
            */
            Desc& setStencilRef(uint8_t value) { mStencilRef = value; return *this; };

        protected:
            bool mDepthEnabled = true;
            bool mStencilEnabled = false;
            bool mWriteDepth = true;
            Func mDepthFunc = Func::Less;
            StencilDesc mStencilFront;
            StencilDesc mStencilBack;
            uint8_t mStencilReadMask = (uint8_t)-1;
            uint8_t mStencilWriteMask = (uint8_t)-1;
            uint8_t mStencilRef = 0;
        };

        ~DepthStencilState();

        /** Create a new depth-stencil state object.
            \param desc Depth-stencil descriptor.
            \return New object, or throws an exception if an error occurred.
        */
        static SharedPtr create(const Desc& desc);

        /** Check if depth test is enabled or disabled
        */
        bool isDepthTestEnabled() const { return mDesc.mDepthEnabled; }

        /** Check if depth write is enabled or disabled
        */
        bool isDepthWriteEnabled() const { return mDesc.mWriteDepth; }

        /** Get the depth comparison function
        */
        Func getDepthFunc() const { return mDesc.mDepthFunc; }

        /** Check if stencil is enabled or disabled
        */
        bool isStencilTestEnabled() const { return mDesc.mStencilEnabled; }

        /** Get the stencil descriptor for the selected face
        */
        const StencilDesc& getStencilDesc(Face face) const;

        /** Get the stencil read mask
        */
        uint8_t getStencilReadMask() const { return mDesc.mStencilReadMask; }

        /** Get the stencil write mask
        */
        uint8_t getStencilWriteMask() const { return mDesc.mStencilWriteMask; }

        /** Get the stencil ref value
        */
        uint8_t getStencilRef() const { return mDesc.mStencilRef; }

        /** Get the API handle
        */
        const DepthStencilStateHandle& getApiHandle() const;

    private:
        DepthStencilStateHandle mApiHandle;
        DepthStencilState(const Desc& Desc) : mDesc(Desc) {}
        Desc mDesc;
    };
}
