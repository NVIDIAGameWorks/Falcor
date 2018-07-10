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
#include "API/FBO.h"

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

    bool RenderPass::addInputCommon(const Reflection::Field& field, Input::Type t, const Fbo::SharedPtr& pFbo, const std::shared_ptr<ProgramVars>& pVars)
    {
        if (mInputs.find(field.name) != mInputs.end())
        {
            logWarning("Error when adding the field `" + field.name + "` to render-pass `" + mName + "`. A field with the same name already exists");
            return false;
        }

        mReflection.inputs.push_back(field);

        Input input;
        input.type = t;
        input.pVars = pVars;
        input.pFbo = pFbo;
        input.pField = &mReflection.inputs.back();
        mInputs[field.name] = input;

        return true;
    }

    static RenderPass::Reflection::Field initField(const std::string& name,
        ResourceFormat format,
        Resource::BindFlags bindFlags,
        uint32_t width,
        uint32_t height,
        uint32_t depth,
        uint32_t sampleCount,
        bool optionalField,
        const ReflectionResourceType::SharedConstPtr& pType
        )
    {
        RenderPass::Reflection::Field f;

        f.optional = optionalField;
        f.depth = depth;
        f.format = format;
        f.height = height;
        f.name = name;
        f.sampleCount = sampleCount;
        f.width = width;
        f.pType = pType;

        if (bindFlags == Resource::BindFlags::None)
        {
            f.bindFlags = Resource::BindFlags::ShaderResource;
            if (f.pType->getShaderAccess() == ReflectionResourceType::ShaderAccess::ReadWrite) f.bindFlags |= Resource::BindFlags::UnorderedAccess;
        }
        else
        {
            f.bindFlags = bindFlags;
        }

        return f;
    }

    bool RenderPass::addInputFieldFromProgramVars(const std::string& name,
        const std::shared_ptr<ProgramVars>& pVars, 
        ResourceFormat format, 
        Resource::BindFlags bindFlags, 
        uint32_t width, 
        uint32_t height, 
        uint32_t depth, 
        uint32_t sampleCount,
        bool optionalField)
    {
        assert(pVars);
        if (pVars->getReflection()->getResource(name) == nullptr)
        {
            logWarning("Error when adding the field `" + name + "` to render-pass `" + mName + "`. Can't find the variable in the program reflection");
            return false;
        }

        const auto& pType = std::dynamic_pointer_cast<const ReflectionResourceType>(pVars->getReflection()->getResource(name)->getType());
        if (pType == nullptr)
        {
            logWarning("Error when adding the field `" + name + "` to render-pass `" + mName + "`. The variable is not a resource");
            return false;
        }

        auto f = initField(name, format, bindFlags, width, height, depth, sampleCount, optionalField, pType);
        return addInputCommon(f, Input::Type::ShaderResource, nullptr, pVars);

        return true;
    }

    bool RenderPass::addDepthBufferField(const std::string& name,
        bool input,
        const std::shared_ptr<Fbo>& pFbo,
        ResourceFormat format,
        Resource::BindFlags flags,
        uint32_t width,
        uint32_t height,
        uint32_t depth,
        uint32_t sampleCount,
        bool optionalField)
    {
        assert(pFbo);

        if (isDepthFormat(format) == false)
        {
            logWarning("Error when adding the depth-buffer field `" + name + "` to render-pass `" + mName + "`. The provided format is not a depth-stencil format");
            return false;
        }

        auto dims = depth > 1 ? ReflectionResourceType::Dimensions::Texture3D : ReflectionResourceType::Dimensions::Texture2D;
        auto pType = ReflectionResourceType::create(ReflectionResourceType::Type::Texture, dims, ReflectionResourceType::StructuredType::Invalid, ReflectionResourceType::ReturnType::Unknown, ReflectionResourceType::ShaderAccess::ReadWrite);
        auto f = initField(name, format, Resource::BindFlags::DepthStencil | Resource::BindFlags::ShaderResource, width, height, depth, sampleCount, optionalField, pType);

        if (input)
        {
            addInputCommon(f, Input::Type::Depth, pFbo, nullptr);
        }
        else
        {

        }

        return true;
    }

    bool RenderPass::setInput(const std::string& name, const std::shared_ptr<Resource>& pResource)
    {
        // Check if the name exists here
        const auto& inputIt = mInputs.find(name);
        if (inputIt == mInputs.end())
        {
            logWarning("Error when binding a resource to a render-pass. The input `" + name + "` doesn't exist");
            return false;
        }

        // Currently we only support textures
        Texture::SharedPtr pTexture = std::dynamic_pointer_cast<Texture>(pResource);
        if (pTexture == nullptr)
        {
            logWarning("Error when binding a resource to a render-pass. The resource provided for the input `" + name + "` is not a texture");
            return false;
        }

        const auto& input = inputIt->second;
        switch (input.type)
        {
        case Input::Type::ShaderResource:
            input.pVars->setTexture(name, pTexture);
            break;
        case Input::Type::Depth:
            input.pFbo->attachDepthStencilTarget(pTexture);
            break;
        default:
            should_not_get_here();
        }
        return true;
    }
}