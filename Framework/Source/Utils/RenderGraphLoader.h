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
#include "Graphics/RenderGraph/RenderGraph.h"

namespace Falcor
{
    // Move this
    // If in editor mode, if a type of pass is unknown, still create a node for it with the correct 
    // reflection and warn. Do not let exectute be called on these.
    class DummyEditorPass : public RenderPass, public inherit_shared_from_this<RenderPass, DummyEditorPass>
    {
    public:
        using SharedPtr = std::shared_ptr<DummyEditorPass>;

        static SharedPtr create(const std::string& name = "Unknown");

        virtual void reflect(RenderPassReflection& reflector) const override;
        virtual void execute(RenderContext* pContext, const RenderData* pRenderData) override;
        virtual void renderUI(Gui* pGui, const char* uiGroup) override;

    private:
        DummyEditorPass(const std::string& name);

        RenderPassReflection mReflector;
    };

    class RenderGraphLoader
    {
    public:

        class ScriptParameter : public Scene::UserVariable
        {
        public:

            template<typename T>
            ScriptParameter(const T& val) { get<T>() = val; }
            
            ScriptParameter() {}

            template<typename T>
            void operator=(const T& val)
            {
                get<T>() = val;
            }

            template <typename T>
            T& get();

            void operator=(const std::string& val);

            void operator=(const ScriptParameter& param) 
            {
                type = param.type;
                d64 = param.d64;
                str = param.str;
                vec2 = param.vec2;
                vec3 = param.vec3;
                vec4 = param.vec4;
                vector = param.vector; 
            }
        };

        class ScriptBinding
        {
        public:
            ScriptBinding() {}
            ScriptBinding(const ScriptBinding&& ref) : mParameters(ref.mParameters), mExecute(ref.mExecute) {}

            using ScriptFunc = std::function<void(ScriptBinding& scriptBinding, RenderGraph& renderGraph)>;

            std::vector<ScriptParameter > mParameters;
            ScriptFunc mExecute;
        };

        RenderGraphLoader();

        static void runScript(const std::string& scriptData, RenderGraph& renderGraph);
        static void runScript(const char* scriptData, size_t dataSize, RenderGraph& renderGraph);

        static void LoadAndRunScript(const std::string& fileNameString, RenderGraph& renderGraph);

        /** Serializes given render graph into a script that can reproduce it
         */
        static void SaveRenderGraphAsScript(const std::string& fileNameString, const RenderGraph& renderGraph);

        static std::string saveRenderGraphAsScriptBuffer(const RenderGraph& renderGraph);

        static void ExecuteStatement(const std::string& statement, RenderGraph& renderGraph);

        static std::string sGraphOutputString;

        // Set for 
        static bool sSharedEditingMode;

    private:
        
        template<typename ... U>
        void RegisterStatement(const std::string& keyword, const ScriptBinding::ScriptFunc& function, U ... defaultValues)
        {
            ScriptBinding newBinding;
            newBinding.mExecute = function;
            newBinding.mParameters = std::vector<ScriptParameter>{ defaultValues ... };
            mScriptBindings.emplace(std::make_pair(keyword, std::move(newBinding)));
        }

        static std::unordered_map<std::string, ScriptBinding> mScriptBindings;
        static std::unordered_map<std::string, ScriptParameter> mActiveVariables;
    };
}
