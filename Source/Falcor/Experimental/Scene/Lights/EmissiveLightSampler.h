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
#include "EmissiveLightSamplerType.slangh"

namespace Falcor
{
    /** Base class for emissive light sampler implementations.

        All light samplers follows the same interface to make them interchangeable.
        If an unrecoverable error occurs, these functions may throw exceptions.
    */
    class dlldecl EmissiveLightSampler : public std::enable_shared_from_this<EmissiveLightSampler>
    {
    public:
        using SharedPtr = std::shared_ptr<EmissiveLightSampler>;
        virtual ~EmissiveLightSampler() = default;

        /** Updates the sampler to the current frame.
            \param[in] pRenderContext The render context.
            \return True if the sampler was updated.
        */
        virtual bool update(RenderContext* pRenderContext) { return false; }

        /** Add compile-time specialization to program to use this light sampler.
            This function must be called every frame before the sampler is bound.
            Note that ProgramVars may need to be re-created after this call, check the return value.
            \param[in] pProgram The Program to add compile-time specialization to.
            \return True if the ProgramVars needs to be re-created.
        */
        virtual bool prepareProgram(Program* pProgram) const;

        /** Bind the light sampler data to a given shader var
        */
        virtual bool setShaderData(const ShaderVar& var) const { return true; }

        /** Render the GUI.
            \return True if settings that affect the rendering have changed.
        */
        virtual bool renderUI(Gui::Widgets& widget) { return false; }

        /** Returns the type of emissive light sampler.
            \return The type of the derived class.
        */
        EmissiveLightSamplerType getType() const { return mType; }

    protected:
        EmissiveLightSampler(EmissiveLightSamplerType type, Scene::SharedPtr pScene) : mType(type), mpScene(pScene) {}

        // Internal state
        const EmissiveLightSamplerType mType;       ///< Type of emissive sampler. See EmissiveLightSamplerType.slangh.
        Scene::SharedPtr mpScene;
    };
}
