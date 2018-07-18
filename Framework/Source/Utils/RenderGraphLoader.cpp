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
#include "RenderGraphLoader.h"
#include "Framework.h"
#include "RenderPasses/BlitPass.h"
#include "RenderPasses/DepthPass.h"
#include "RenderPasses/ShadowPass.h"
#include "RenderPasses/SceneLightingPass.h"
#include "Effects/ToneMapping/ToneMapping.h"
#include <fstream>

namespace Falcor
{
    std::unordered_map<std::string, RenderGraphLoader::ScriptBinding> RenderGraphLoader::mScriptBindings;
    std::unordered_map<std::string, std::function<RenderPass::SharedPtr()>> RenderGraphLoader::sBaseRenderCreateFuncs;
    std::string RenderGraphLoader::sGraphOutputString;

    const std::string kAddRenderPassCommand = std::string("AddRenderPass");
    const std::string kAddEdgeCommand = std::string("AddEdge");

    // moved to framework with the constant buffer ui stuff
#define concat_strings_(a_, b_) a_##b_
#define concat_strings(a, b) concat_strings_(a, b)

#define script_parameter_get(type_, memberName_)  template <> type_& RenderGraphLoader::ScriptParameter::get<type_>() \
        { \
            mType = VariantType::memberName_; \
            return concat_strings(mData.m, memberName_); \
        } \

    script_parameter_get( int32_t, Int);
    script_parameter_get(uint32_t, UInt);
    script_parameter_get(bool, Bool);
    script_parameter_get(float, Float);

#undef concat_strings
#undef concat_strings_
#undef  script_parameter_get

    static RenderGraphLoader sRenderGraphLoaderInstance;

    template <> std::string& RenderGraphLoader::ScriptParameter::get<std::string>()
    {
        mType = VariantType::String;
        return mData.mString;
    }

    void RenderGraphLoader::ScriptParameter::operator=(const std::string& val)
    {
        switch (mType)
        {
        case VariantType::Float:
            get<float>() = static_cast<float>(std::atof(val.c_str()));
            break;
        case VariantType::UInt:
            get<uint32_t>() = static_cast<uint32_t>(std::atoi(val.c_str()));
            break;
        case VariantType::Int:
            get<int32_t>() = std::atoi(val.c_str());
            break;
        case VariantType::Bool:
            if (val == "true") get<bool>() = true;
            if (val == "false") get<bool>() = false;
            break;
        case VariantType::String:
            get<std::string>() = val;
            break;
        default:
            should_not_get_here();
        }
    }

    void RenderGraphLoader::SaveRenderGraphAsScript(const std::string& fileNameString, RenderGraph& renderGraph)
    {
        std::ofstream scriptFile(fileNameString);
        assert(scriptFile.is_open());

        // More generic schema for copying serialization ??

        std::unordered_map<uint16_t, std::string> linkIDToSrcPassName;
        std::string currentCommand;

        // do a pre-pass to map all of the outgoing connections to the names of the passes
        for (const auto& nameToIndex : renderGraph.mNameToIndex)
        {
            auto pCurrentPass = renderGraph.mpGraph->getNode(nameToIndex.second);

            // add all of the add render pass commands here
            currentCommand = kAddRenderPassCommand + " " + nameToIndex.first + " " 
                + renderGraph.mNodeData[nameToIndex.second].pPass->getName() + "\n";
            scriptFile.write(currentCommand.data(), currentCommand.size());

            for (uint32_t i = 0; i < pCurrentPass->getOutgoingEdgeCount(); ++i)
            {
                uint32_t edgeID = pCurrentPass->getOutgoingEdge(i);

                linkIDToSrcPassName[edgeID] = nameToIndex.first;
            }
        }

        // add all of the add edge commands
        for (const auto& nameToIndex : renderGraph.mNameToIndex)
        {
            auto pCurrentPass = renderGraph.mpGraph->getNode(nameToIndex.second);
            // pCurrentPass->

            // just go through incoming edges for each node
            for (uint32_t i = 0; i < pCurrentPass->getIncomingEdgeCount(); ++i)
            {
                uint32_t edgeID = pCurrentPass->getIncomingEdge(i);
                auto currentEdge = renderGraph.mEdgeData[edgeID];
                
                currentCommand = kAddEdgeCommand + " " + linkIDToSrcPassName[edgeID] + "." + currentEdge.srcField + " "
                    + nameToIndex.first + "."  + currentEdge.dstField + "\n";

                scriptFile.write(currentCommand.data(), currentCommand.size());
            }
        }

        // set graph output command
        for (const auto& graphOutput : renderGraph.mOutputs)
        {
            currentCommand = "AddGraphOutput ";

            auto pCurrentPass = renderGraph.mpGraph->getNode(graphOutput.nodeId);

            // if nodes have been deleted but graph outputs remain, if have to check if the nodeID is valid
            if (pCurrentPass == nullptr)
            {
                logWarning(std::string("Failed to save graph output '") + graphOutput.field + "'. Render graph output is from a pass that no longer exists or is invalid.");
                continue;
            }

            if (pCurrentPass->getOutgoingEdgeCount())
            {
                currentCommand += linkIDToSrcPassName[pCurrentPass->getOutgoingEdge(0)];
            }
            else
            {
                for (const auto& it : renderGraph.mNameToIndex)
                {
                    if (it.second == graphOutput.nodeId)
                    {
                        currentCommand += it.first;
                        break;
                    }
                }
            }

            currentCommand += "." + graphOutput.field + "\n";
            scriptFile.write(currentCommand.data(), currentCommand.size());
        }

        scriptFile.close();
    }

    void RenderGraphLoader::LoadAndRunScript(const char* fileData, RenderGraph& renderGraph)
    {
        size_t offset = 0;

        
        {

        }
    }

    void RenderGraphLoader::LoadAndRunScript(const std::string& fileNameString, RenderGraph& renderGraph)
    {
        std::ifstream scriptFile(fileNameString);

        std::string line;
        line.resize(255, '0');

        assert(scriptFile.is_open());

        while (!scriptFile.eof())
        {
            scriptFile.getline(&*line.begin(), line.size());
            if (!line.size()) { break; }

            ExecuteStatement(line.substr(0, line.find_first_of('\0')), renderGraph);
        }

        scriptFile.close();
    }

    void RenderGraphLoader::ExecuteStatement(const std::string& statement, RenderGraph& renderGraph)
    {
        if (!statement.size())
        {
            // log warning??
            return;
        }

        logInfo(std::string("Executing statement: ") + statement);

        // split the statement into keyword and parameters
        size_t charIndex = 0; 
        size_t lastIndex = 0;
        
        std::vector<std::string> statementPeices;

        for (;;)
        {
            charIndex = statement.find_first_of(' ', lastIndex);
            if (charIndex == std::string::npos)
            {
                statementPeices.push_back(statement.substr(lastIndex, statement.size() - lastIndex));
                break;
            }
            statementPeices.push_back(statement.substr(lastIndex, charIndex - lastIndex));
            lastIndex = charIndex + 1;
        }

        // 0th I
        auto binding = mScriptBindings.find(statementPeices[0]);

        if (binding == mScriptBindings.end())
        {
            logWarning(std::string("Unknown Command Skipped: ") + statementPeices[0]);
            return;
        }

        for (uint32_t i = 0; i < statementPeices.size() - 1; ++i)
        {
            binding->second.mParameters[i] = statementPeices[i + 1];
        }
        
        binding->second.mExecute(binding->second, renderGraph);
    }

    RenderGraphLoader::RenderGraphLoader()
    {
        // default script bindings
        sGraphOutputString.resize(255, '0');

        RegisterStatement<std::string, std::string>("AddRenderPass", [](ScriptBinding& scriptBinding, RenderGraph& renderGraph) { 
            std::string passTypeName = scriptBinding.mParameters[1].get<std::string>();
            auto createPassCallbackIt = sBaseRenderCreateFuncs.find(passTypeName);
            if (createPassCallbackIt == sBaseRenderCreateFuncs.end())
            {
                logWarning("Failed on attempt to create unknown pass : " + passTypeName);
                return;
            }
            renderGraph.addRenderPass(createPassCallbackIt->second(), scriptBinding.mParameters[0].get<std::string>());
        }, {}, {});

        RegisterStatement<std::string, std::string>("AddEdge", [](ScriptBinding& scriptBinding, RenderGraph& renderGraph) {
            renderGraph.addEdge(scriptBinding.mParameters[0].get<std::string>(), scriptBinding.mParameters[1].get<std::string>());
        }, {}, {});

        RegisterStatement<std::string>("RemoveRenderPass", [](ScriptBinding& scriptBinding, RenderGraph& renderGraph) {
            renderGraph.removeRenderPass(scriptBinding.mParameters[0].get<std::string>());
        }, {});

        RegisterStatement<std::string>("AddGraphOutput", [](ScriptBinding& scriptBinding, RenderGraph& renderGraph) {
            sGraphOutputString = scriptBinding.mParameters[0].get<std::string>();
            renderGraph.markGraphOutput(sGraphOutputString);
        }, {});

        RegisterStatement<std::string>("RemoveGraphOutput", [](ScriptBinding& scriptBinding, RenderGraph& renderGraph) {
            renderGraph.unmarkGraphOutput(scriptBinding.mParameters[0].get<std::string>());
        }, {});

        RegisterStatement<std::string>("SetScene", [](ScriptBinding& scriptBinding, RenderGraph& renderGraph) {
            Scene::SharedPtr pScene =  Scene::loadFromFile(scriptBinding.mParameters[0].get<std::string>());
            if (!pScene) { logWarning("Failed to load scene for current render graph"); return; }
            renderGraph.setScene(pScene);
        }, {});



        // register static passes
#define register_render_pass(renderPassType) \
    sBaseRenderCreateFuncs.insert(std::make_pair(#renderPassType, std::function<RenderPass::SharedPtr()> ( \
        []() { return renderPassType::create(); }) )\
    );

        register_render_pass(BlitPass);
        register_render_pass(DepthPass);
        register_render_pass(ShadowPass);

#undef register_render_pass
#define register_resource_type() 

        // specialized registries

        
        sBaseRenderCreateFuncs.insert(std::make_pair("SceneLightingPass", std::function<RenderPass::SharedPtr()>(
            []() { 
                SceneLightingPass::Desc lightDesc;
                lightDesc.setColorFormat(ResourceFormat::RGBA32Float).setMotionVecFormat(ResourceFormat::RG16Float).setNormalMapFormat(ResourceFormat::RGBA8Unorm).setSampleCount(1);
                return SceneLightingPass::create(lightDesc); 
            }
        )));

        sBaseRenderCreateFuncs.insert(std::make_pair("ToneMappingPass", std::function<RenderPass::SharedPtr()>(
            []() { return ToneMapping::create(ToneMapping::Operator::Aces); }))
        );
    }

}
