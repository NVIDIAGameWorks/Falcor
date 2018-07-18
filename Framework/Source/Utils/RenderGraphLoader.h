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
    class RenderGraphLoader
    {
    public:

        class ScriptParameter
        {
        public:

            template<typename T>
            ScriptParameter(const T& val) { get<T>() = val; }

            template<typename T>
            void operator=(const T& val)
            {
                get<T>() = val;
            }

            template <typename T>
            T& get();

            void operator=(const std::string& val);

        private:

            enum VariantType
            {
                Float = 0, UInt, Int, Bool, String
            };

            VariantType mType;

            // simple variant
            union var
            {
                var() :mString(){}
                var(const var& val) : mString({}) { mString = val.mString; }
                ~var() {}

                float mFloat; 
                uint32_t mUInt;
                int32_t mInt; 
                bool mBool;
                std::string mString;
            };
            
            var mData;
        };

        class ScriptBinding
        {
        public:
            ScriptBinding() {}
            ScriptBinding(const ScriptBinding&& ref) : mParameters(ref.mParameters), mExecute(ref.mExecute) {}

            std::vector<ScriptParameter > mParameters;
            std::function<void(ScriptBinding& scriptBinding, RenderGraph& renderGraph)> mExecute;
        };

        RenderGraphLoader();

        static void LoadAndRunScript(const char* fileNameString, RenderGraph& renderGraph);

        static void LoadAndRunScript(const std::string& fileNameString, RenderGraph& renderGraph);

        /** Serializes given render graph into a script that can reproduce it
         */
        static void SaveRenderGraphAsScript(const std::string& fileNameString, RenderGraph& renderGraph);

        static void ExecuteStatement(const std::string& statement, RenderGraph& renderGraph);

        static std::string sGraphOutputString;

        // simple lookup to create render pass type from string
        static std::unordered_map<std::string, std::function<RenderPass::SharedPtr()> > sBaseRenderCreateFuncs;

    private:
        
        template<typename ... U>
        void RegisterStatement(const std::string& keyword, const std::function<void(ScriptBinding& scriptBinding, RenderGraph& renderGraph)>& function, U ... defaultValues)
        {
            ScriptBinding newBinding;
            newBinding.mExecute = function;
            newBinding.mParameters = std::vector<ScriptParameter>{ defaultValues ... };
            mScriptBindings.emplace(std::make_pair(keyword, std::move(newBinding)));
        }

        static std::unordered_map<std::string, ScriptBinding> mScriptBindings;
    };
}
