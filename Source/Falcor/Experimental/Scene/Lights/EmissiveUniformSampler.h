/***************************************************************************
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include "EmissiveLightSampler.h"
#include "LightCollection.h"

namespace Falcor
{
    /** Emissive light sampler using uniform sampling of the lights.

        This class wraps a LightCollection object, which holds the set of lights to sample.
    */
    class dlldecl EmissiveUniformSampler : public EmissiveLightSampler, inherit_shared_from_this<EmissiveLightSampler, EmissiveUniformSampler>
    {
    public:
        using SharedPtr = std::shared_ptr<EmissiveUniformSampler>;
        using SharedConstPtr = std::shared_ptr<const EmissiveUniformSampler>;

        /** EmissiveUniformSampler configuration.
            Note if you change options, please update registerScriptBindings().
        */
        // TODO: Rename to shorter name when scoped struct names can be used with the scripting.
        //struct Options : Falcor::ScriptBindings::enable_to_string
        struct EmissiveUniformSamplerOptions : Falcor::ScriptBindings::enable_to_string
        {
            // TODO
        };

        virtual ~EmissiveUniformSampler() = default;

        /** Creates a EmissiveUniformSampler for a given scene.
            \param[in] pRenderContext The render context.
            \param[in] pScene The scene.
            \param[in] options The options to override the default behavior.
        */
        static SharedPtr create(RenderContext* pRenderContext, Scene::SharedPtr pScene, const EmissiveUniformSamplerOptions& options = EmissiveUniformSamplerOptions());

        /** Updates the sampler to the current frame.
            \param[in] pRenderContext The render context.
            \return True if the lighting in the scene has changed.
        */
        virtual bool update(RenderContext* pRenderContext) override;

        /** Add compile-time specialization to program to use this light sampler.
            This function must be called every frame before the sampler is bound.
            Note that ProgramVars may need to be re-created after this call, check the return value.
            \param[in] pProgram The Program to add compile-time specialization to.
            \return True if the ProgramVars needs to be re-created.
        */
        virtual bool prepareProgram(ProgramBase* pProgram) const override;

        /** Render the GUI.
            \return True if setting the refresh flag is needed, false otherwise.
        */
        virtual bool renderUI(Gui::Widgets& widget) override;

        /** Returns the number of active lights.
            The caller can use this to determine if light sampling should be enabled for the
            current frame. Note that the number may change after each call to update().
            \return Number of currently active lights.
        */
        virtual uint32_t getLightCount() const override { return mpLights ? mpLights->getActiveLightCount() : 0; }

        /** Returns the current configuration.
        */
        const EmissiveUniformSamplerOptions& getOptions() const { return mOptions; }

        /** Returns the internal LightCollection object.
        */
        virtual LightCollection::SharedConstPtr getLightCollection() const { return mpLights; }

        static void registerScriptBindings(ScriptBindings::Module& m);

    protected:
        EmissiveUniformSampler() : EmissiveLightSampler(EmissiveLightSamplerType::Uniform) {}

        bool init(RenderContext* pRenderContext, Scene::SharedPtr pScene, const EmissiveUniformSamplerOptions& options);

        /** Bind the light sampler data to a given constant buffer in a parameter block.
            Note that prepareProgram() must have been called before this function.
            \param[in] pBlock The parameter block to set the data into (possibly the default parameter block).
            \param[in] pCB The constant buffer in the parameter block to set the data into.
            \param[in] varName The name of the data variable.
            \return True if successful, false otherwise.
        */
        virtual bool setIntoBlockCommon(const ParameterBlock::SharedPtr& pBlock, const ConstantBuffer::SharedPtr& pCB, const std::string& varName) const override;

        // Configuration
        EmissiveUniformSamplerOptions   mOptions;               ///< Current configuration options.

        // Internal state
        LightCollection::SharedPtr      mpLights;               ///< The collection of lights to sample.
    };
}
