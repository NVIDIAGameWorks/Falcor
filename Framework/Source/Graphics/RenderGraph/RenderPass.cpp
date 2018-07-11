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

    bool RenderPass::addVariableCommon(bool inputVar, const Reflection::Field& field, Variable::Type t, const std::shared_ptr<Fbo>& pFbo, const std::shared_ptr<ProgramVars>& pVars)
    {
        auto& map = inputVar ? mInputs : mOutputs;
        auto& reflectionMap = inputVar ? mReflection.inputs : mReflection.outputs;

        if (map.find(field.name) != map.end())
        {
            logWarning("Error when adding the field `" + field.name + "` to render-pass `" + mName + "`. A field with the same name already exists");
            return false;
        }

        reflectionMap.push_back(field);

        Variable var;
        var.type = t;
        var.pVars = pVars;
        var.pFbo = pFbo;
        var.pField = &mReflection.inputs.back();
        map[field.name] = var;

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
        return addVariableCommon(true, f, Variable::Type::ShaderResource, nullptr, pVars);

        return true;
    }

    bool RenderPass::addDepthBufferField(const std::string& name,
        bool input,
        const std::shared_ptr<Fbo>& pFbo,
        ResourceFormat format,
        Resource::BindFlags bindFlags,
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
        auto f = initField(name, format, bindFlags, width, height, depth, sampleCount, optionalField, pType);

        return addVariableCommon(input, f, Variable::Type::Depth, pFbo, nullptr);
    }

    bool RenderPass::addRenderTargetField(const std::string& name, const std::shared_ptr<Fbo>& pFbo, ResourceFormat format, Resource::BindFlags bindFlags, uint32_t width, uint32_t height, uint32_t depth, uint32_t sampleCount, bool optionalField)
    {
        assert(pFbo);

        auto dims = depth > 1 ? ReflectionResourceType::Dimensions::Texture3D : ReflectionResourceType::Dimensions::Texture2D;
        auto pType = ReflectionResourceType::create(ReflectionResourceType::Type::Texture, dims, ReflectionResourceType::StructuredType::Invalid, ReflectionResourceType::ReturnType::Unknown, ReflectionResourceType::ShaderAccess::ReadWrite);
        auto f = initField(name, format, bindFlags, width, height, depth, sampleCount, optionalField, pType);

        return addVariableCommon(false, f, Variable::Type::RenderTarget, pFbo, nullptr);

    }

    template<bool input>
    bool RenderPass::setVariableCommon(const std::string& name, const std::shared_ptr<Resource>& pResource) const
    {
        const auto& varMap = input ? mInputs : mOutputs;
        const std::string varType = input ? "input" : "output";

        // Check if the name exists here
        const auto& varIt = varMap.find(name);
        if (varIt == varMap.end())
        {
            logWarning("Error when binding a resource to a render-pass. The " + varType +" `" + name + "` doesn't exist");
            return false;
        }

        // Currently we only support textures
        Texture::SharedPtr pTexture = std::dynamic_pointer_cast<Texture>(pResource);
        if (pTexture == nullptr)
        {
            logWarning("Error when binding a resource to a render-pass. The resource provided for the " + varType + " `" + name + "` is not a texture");
            return false;
        }

        const auto& var = varIt->second;
        switch (var.type)
        {
        case Variable::Type::ShaderResource:
            assert(input);
            var.pVars->setTexture(name, pTexture);
            break;
        case Variable::Type::Depth:
            var.pFbo->attachDepthStencilTarget(pTexture);
            break;
        case Variable::Type::RenderTarget:
            assert(input == false);
            var.pFbo->attachColorTarget(pTexture, 0);
            break;
        default:
            should_not_get_here();
            return false;
        }
        return true;
    }

    bool RenderPass::setInput(const std::string& name, const std::shared_ptr<Resource>& pResource)
    {
        return setVariableCommon<true>(name, pResource);
    }

    bool RenderPass::setOutput(const std::string& name, const std::shared_ptr<Resource>& pResource)
    {
        return setVariableCommon<false>(name, pResource);
    }
}