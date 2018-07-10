/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "RenderPass.h"
#include "Graphics/Program/ProgramVars.h"

namespace Falcor
{
    RenderPass::RenderPass(const std::string& name, std::shared_ptr<Scene> pScene, RenderDataChangedFunc pDataChangedCB) : mName(name), mpRenderDataChangedCallback(pDataChangedCB)
    {
        setScene(pScene);
    }

    RenderPass::~RenderPass() = default;

    void RenderPass::setScene(const std::shared_ptr<Scene>& pScene)
    {
        mpScene = pScene;
        sceneChangedCB();
    }

    std::shared_ptr<Resource> RenderPass::getOutput(const std::string& name) const
    {
        logWarning(mName + " doesn't have an output resource called `" + name + "`");
        return nullptr;
    }

    std::shared_ptr<Resource> RenderPass::getInput(const std::string& name) const
    {
        logWarning(mName + " doesn't have an input resource called `" + name + "`");
        return nullptr;
    }

    bool RenderPass::addFieldFromProgramVars(const std::string& name,
        bool input,
        const std::shared_ptr<ProgramVars>& pVars, 
        ResourceFormat requiredFormat, 
        Resource::BindFlags requiredFlags, 
        uint32_t requiredWidth, 
        uint32_t requiredHeight, 
        uint32_t requiredDepth, 
        uint32_t requiredSampleCount,
        bool optionalField)
    {
        assert(pVars);
        if (pVars->getReflection()->getResource(name) == nullptr)
        {
            logWarning("RenderPass::addFieldFromProgramVars() - can't find a resource named '" + name + "' in the program reflection");
            return false;
        }

        Reflection::Field f;
        f.pType = std::dynamic_pointer_cast<const ReflectionResourceType>(pVars->getReflection()->getResource(name)->getType());
        if (f.pType == nullptr)
        {
            logWarning("RenderPass::addFieldFromProgramVars() - variable '" + name + "' is not a resource");
            return false;
        }

        f.optional = optionalField;
        f.depth = requiredDepth;
        f.format = requiredFormat;
        f.height = requiredHeight;
        f.name = name;
        f.sampleCount = requiredSampleCount;
        f.width = requiredWidth;

        if (requiredFlags == Resource::BindFlags::None)
        {
            f.bindFlags = Resource::BindFlags::ShaderResource;
            if (input)
            {
                if (f.pType->getShaderAccess() == ReflectionResourceType::ShaderAccess::ReadWrite) f.bindFlags |= Resource::BindFlags::UnorderedAccess;
            }
            else
            {
                f.bindFlags |= Resource::BindFlags::RenderTarget;
            }
        }
        else
        {
            f.bindFlags = requiredFlags;
        }
        return true;
    }
}