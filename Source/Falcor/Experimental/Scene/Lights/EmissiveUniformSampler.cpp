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
#include "stdafx.h"
#include "EmissiveUniformSampler.h"

namespace Falcor
{
    EmissiveUniformSampler::SharedPtr EmissiveUniformSampler::create(RenderContext* pRenderContext, Scene::SharedPtr pScene, const EmissiveUniformSamplerOptions& options)
    {
        SharedPtr ptr = SharedPtr(new EmissiveUniformSampler());
        return ptr->init(pRenderContext, pScene, options) ? ptr : nullptr;
    }

    bool EmissiveUniformSampler::update(RenderContext* pRenderContext)
    {
        PROFILE("EmissiveUniformSampler::update");

        // Update the light collection.
        assert(mpLights);
        bool lightingChanged = mpLights->update(pRenderContext);

        return lightingChanged;
    }

    bool EmissiveUniformSampler::prepareProgram(ProgramBase* pProgram) const
    {
        // Call the base class first.
        bool varsChanged = EmissiveLightSampler::prepareProgram(pProgram);

        // Specialize the program for the light collection.
        assert(mpLights);
        varsChanged |= mpLights->prepareProgram(pProgram);

        return varsChanged;
    }

    bool EmissiveUniformSampler::renderUI(Gui::Widgets& widget)
    {
        bool dirty = false;

        auto collectionGroup = Gui::Group(widget, "LightCollection");
        if (collectionGroup.open())
        {
            assert(mpLights);
            dirty = mpLights->renderUI(collectionGroup);
            collectionGroup.release();
        }
        
        return dirty;
    }

    bool EmissiveUniformSampler::init(RenderContext* pRenderContext, Scene::SharedPtr pScene, const EmissiveUniformSamplerOptions& options)
    {
        mOptions = options;

        // Create light collection for the scene.
        mpLights = LightCollection::create(pRenderContext, pScene);
        if (!mpLights) return false;

        return true;
    }

    bool EmissiveUniformSampler::setIntoBlockCommon(const ParameterBlock::SharedPtr& pBlock, const ConstantBuffer::SharedPtr& pCB, const std::string& varName) const
    {
        assert(pBlock);
        assert(pCB);

        // Check that the struct exists.
        if (pCB->getVariableOffset(varName) == ConstantBuffer::kInvalidOffset)
        {
            logError("EmissiveUniformSampler::setIntoBlockCommon() - Variable " + varName + " does not exist");
            return false;
        }
        std::string prefix = varName + ".";

        // Ok. The struct exists.
        // In the following we validate it has the correct fields and set the data.

        // Bind the lights first.
        assert(mpLights);
        if (!mpLights->setIntoBlockCommon(pBlock, pCB, prefix + "_lights"))
        {
            logError("EmissiveUniformSampler::setIntoBlockCommon() - Failed to bind lights");
            return false;
        }

        return true;
    }

    void EmissiveUniformSampler::registerScriptBindings(ScriptBindings::Module& m)
    {
        EmissiveLightSampler::registerScriptBindings(m);

        if (!m.classExists<EmissiveUniformSamplerOptions>())
        {
            auto options = m.regClass(EmissiveUniformSamplerOptions);
#define field(f_) rwField(#f_, &EmissiveUniformSamplerOptions::f_)
            // TODO
            //options.field(usePreintegration);
#undef field
        }
    }
}
