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
#pragma once
#include "Graphics/Program/ProgramReflection.h"
#include "Renderer.h"

namespace Falcor
{
    class Scene;
    class Resource;
    class Gui;
    class RenderContext;
    class ProgramVars;

    /** Base class for render-passes. The class inherits from Renderer
    */
    class RenderPass : public Renderer, inherit_shared_from_this<Renderer, RenderPass>
    {
    public:
        using SharedPtr = std::shared_ptr<RenderPass>;
        using RenderDataChangedFunc = std::function<void(void)>;

        virtual ~RenderPass() = 0;

        /** This struct describes the available input/output resources fields by the render-pass
        */
        struct Reflection
        {
            struct Field
            {
                std::string name;                        ///< The field's name
                ReflectionResourceType::SharedConstPtr pType; ///< The resource type
                uint32_t width = 0;         ///< For output resources, 0 means use the window size(textures) or the size in bytes (buffers). For input resources 0 means don't care
                uint32_t height = 0;        ///< For output resources, 0 means use the window size. For input resources 0 means don't care
                uint32_t depth = 0;         ///< For output resources, 0 means use the window size. For input resources 0 means don't care
                uint32_t sampleCount = 0;   ///< 0 means don't care (which means 1 for output resources)
                ResourceFormat format = ResourceFormat::Unknown; ///< Unknown means use the back-buffer format for output resources, don't care for input resources
                Resource::BindFlags bindFlags = Resource::BindFlags::None;  ///< The required bind flags
                bool optional = false;      ///< If this is false, then the render-pass will not work if this field is not set. Otherwise, this field is optional
            };

            std::vector<Field> inputs;
            std::vector<Field> outputs;
        };

        /** Execute the pass
        */
        virtual void execute(RenderContext* pContext) = 0;

        /** Get the render-pass data
        */
        virtual const Reflection& getReflection() const final { return mReflection; }

        /** Set an input resource. The function will return true if the resource fulfills the slot requirements, otherwise it will return false
        */
        virtual bool setInput(const std::string& name, const std::shared_ptr<Resource>& pResource);

        /** Set an input resource. The function will return true if the resource fulfills the slot requirements, otherwise it will return false
        */
        virtual bool setOutput(const std::string& name, const std::shared_ptr<Resource>& pResource) = 0;

        /** Get an input resource
        */
        virtual std::shared_ptr<Resource> getInput(const std::string& name) const;

        /** Get an output resource
        */
        virtual std::shared_ptr<Resource> getOutput(const std::string& name) const;

        /** Call this after the input/output resources are set to make sure the render-pass is ready for execution
        */
        virtual bool isValid(std::string& log = std::string()) = 0;

        /** Set a scene into the render-pass
        */
        void setScene(const std::shared_ptr<Scene>& pScene);

        /** Get the currently bound scene
        */
        const std::shared_ptr<Scene>& getScene() const { return mpScene; }

        /** Optional callback function which will be invoked whenever a scene is set
        */
        virtual void sceneChangedCB() {};

        /** Optional serialization function. Use this to export custom data into the json file
        */
        virtual void serializeJson() const {}

        /** Set the DataChanged callback
        */
        void setRenderDataChangedCallback(RenderDataChangedFunc pDataChangedCB) { mpRenderDataChangedCallback = pDataChangedCB; }
    protected:
        RenderPass(const std::string& name, std::shared_ptr<Scene> pScene, RenderDataChangedFunc pDataChangedCB = nullptr);
        
        std::string mName;
        std::shared_ptr<Scene> mpScene;
        RenderDataChangedFunc mpRenderDataChangedCallback;

        bool addInputFieldFromProgramVars(const std::string& name, 
            const std::shared_ptr<ProgramVars>& pVars, 
            ResourceFormat format = ResourceFormat::Unknown,
            Resource::BindFlags bindFlags = Resource::BindFlags::None, 
            uint32_t width = 0, 
            uint32_t height = 0, 
            uint32_t depth = 0,
            uint32_t sampleCount = 0,
            bool optionalField = false);

        bool addDepthBufferField(const std::string& name,
            bool input,
            const std::shared_ptr<Fbo>& pFbo,
            ResourceFormat format = ResourceFormat::D32Float,
            Resource::BindFlags bindFlags = Resource::BindFlags::DepthStencil,
            uint32_t width = 0,
            uint32_t height = 0,
            uint32_t depth = 0,
            uint32_t sampleCount = 0,
            bool optionalField = false);

        Reflection mReflection;

        struct Input
        {
            enum class Type
            {
                Depth,
                ShaderResource
            };

            Type type;
            std::shared_ptr<ProgramVars> pVars;
            std::shared_ptr<Fbo> pFbo;
            const Reflection::Field* pField;
        };

        bool addInputCommon(const Reflection::Field& field, Input::Type t, const std::shared_ptr<Fbo>& pFbo, const std::shared_ptr<ProgramVars>& pVars);

        std::unordered_map<std::string, Input> mInputs;
    };
}