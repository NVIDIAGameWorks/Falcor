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
#include "Falcor.h"
#include <fstream>
#include <sstream>

namespace Falcor
{
    std::unordered_map<std::string, RenderGraphLoader::ScriptBinding> RenderGraphLoader::mScriptBindings;
    std::unordered_map<std::string, RenderGraphLoader::ScriptParameter> RenderGraphLoader::mActiveVariables;
    std::string RenderGraphLoader::sGraphOutputString;
    bool RenderGraphLoader::sSharedEditingMode = false;

    const std::string kAddRenderPassCommand = std::string("AddRenderPass");
    const std::string kAddEdgeCommand = std::string("AddEdge");

    DummyEditorPass::DummyEditorPass(const std::string& name) : RenderPass(name)
    {
    }

    DummyEditorPass::SharedPtr DummyEditorPass::create(const std::string& name)
    {
        try
        {
            return SharedPtr(new DummyEditorPass(name));
        }
        catch (const std::exception&)
        {
            return nullptr;
        }
    }

    void DummyEditorPass::reflect(RenderPassReflection& reflector) const
    {
        reflector = mReflector;
    }
    
    void DummyEditorPass::execute(RenderContext*, const RenderData*)
    {
        should_not_get_here();
    }
    
    void DummyEditorPass::renderUI(Gui* pGui, const char* uiGroup)
    {
        pGui->addText((std::string("This Render Pass Type") + mName + " is not registered").c_str());
    }

#define script_parameter_get(type_, member_, typeNameEnum_)  template <> type_& RenderGraphLoader::ScriptParameter::get<type_>() \
        { \
            type = Type::typeNameEnum_; \
            return member_; \
        }

    script_parameter_get(int32_t, i32, Int)
    script_parameter_get(uint32_t, u32, Uint);
    script_parameter_get(bool,   b, Bool);
    script_parameter_get(double, d64, Double);
    script_parameter_get(std::string, str, String);

#undef  script_parameter_get

    static RenderGraphLoader sRenderGraphLoaderInstance;

    void RenderGraphLoader::ScriptParameter::operator=(const std::string& val)
    {
        switch (type)
        {
        case Type::Double:
            get<double>() = static_cast<double>(std::atof(val.c_str()));
            break;
        case Type::Uint:
            get<uint32_t>() = static_cast<uint32_t>(std::atoi(val.c_str()));
            break;
        case Type::Int:
            get<int32_t>() = std::atoi(val.c_str());
            break;
        case Type::Bool:
            if (val == "true") get<bool>() = true;
            if (val == "false") get<bool>() = false;
            break;
        case Type::String:
            get<std::string>() = val;
            break;
        default:
            should_not_get_here();
        }
    }

    std::string RenderGraphLoader::saveRenderGraphAsScriptBuffer(const RenderGraph& renderGraph)
    {
        std::string scriptString;
        std::unordered_map<uint16_t, std::string> linkIDToSrcPassName;
        std::string currentCommand;
        std::string sceneFilename = mActiveVariables["gSceneFilename"].get<std::string>() = renderGraph.getScene()->mFileName;
        Scene::UserVariable var = renderGraph.getScene()->getUserVariable("sky_box");
        assert(var.type == Scene::UserVariable::Type::String);
        mActiveVariables["gSkyBoxFilename"] = ScriptParameter(var.str);

        // first set the name of the scene for the passes that dependent on it during their initialization
        currentCommand = "SetScene ";
        currentCommand += sceneFilename + '\n';
        scriptString.insert(scriptString.end(), currentCommand.begin(), currentCommand.end());

        // do a pre-pass to map all of the outgoing connections to the names of the passes
        for (const auto& nameToIndex : renderGraph.mNameToIndex)
        {
            auto pCurrentPass = renderGraph.mpGraph->getNode(nameToIndex.second);
            std::string renderPassClassName = renderGraph.mNodeData.find(nameToIndex.second)->second.pPass->getName();
            
            // need to deserialize the serialization data. stored in the RenderPassLibrary
            RenderPassSerializer& renderPassSerializerRef = RenderPassLibrary::getRenderPassSerializer(renderPassClassName.c_str());

            for (size_t i = 0; i < renderPassSerializerRef.getVariableCount(); ++i )
            {
                const auto& variableRef = renderPassSerializerRef.getValue(i);
                const std::string& varName =  renderPassSerializerRef.getVariableName(i);
                if (varName[0] == 'g') continue;

                switch (variableRef.type)
                {
                case Scene::UserVariable::Type::Bool:
                    currentCommand = "VarBool ";
                    currentCommand += varName + " " + (renderPassSerializerRef.getValue(i).b ? "true" : "false") + '\n';
                    break;
                case Scene::UserVariable::Type::Double:
                    currentCommand = "VarFloat ";
                    currentCommand += varName + " " + std::to_string(static_cast<float>(renderPassSerializerRef.getValue(i).d64)) + '\n';
                    break;
                case Scene::UserVariable::Type::String:
                    currentCommand = "VarString ";
                    currentCommand += varName + " " + renderPassSerializerRef.getValue(i).str + '\n';
                    break;
                case Scene::UserVariable::Type::Uint:
                    currentCommand = "VarUInt ";
                    currentCommand += varName + " " + std::to_string(renderPassSerializerRef.getValue(i).u32) + '\n';
                    break;
                case Scene::UserVariable::Type::Int:
                    currentCommand = "VarInt ";
                    currentCommand += varName + " " + std::to_string(renderPassSerializerRef.getValue(i).i32) + '\n';
                    break;
                default:
                    should_not_get_here();
                };
                
                scriptString.insert(scriptString.end(), currentCommand.begin(), currentCommand.end());
            }

            // add all of the add render pass commands here
            currentCommand = kAddRenderPassCommand + " " + nameToIndex.first + " " + renderPassClassName + "\n";
            scriptString.insert(scriptString.end(), currentCommand.begin(), currentCommand.end());

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
            
            // just go through incoming edges for each node
            for (uint32_t i = 0; i < pCurrentPass->getIncomingEdgeCount(); ++i)
            {
                uint32_t edgeID = pCurrentPass->getIncomingEdge(i);
                auto currentEdge = renderGraph.mEdgeData.find(edgeID)->second;

                currentCommand = kAddEdgeCommand + " " + linkIDToSrcPassName[edgeID] + "." + currentEdge.srcField + " "
                    + nameToIndex.first + "." + currentEdge.dstField + "\n";

                scriptString.insert(scriptString.end(), currentCommand.begin(), currentCommand.end());
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
            scriptString.insert(scriptString.end(), currentCommand.begin(), currentCommand.end());
        }

        return scriptString;
    }

    void RenderGraphLoader::SaveRenderGraphAsScript(const std::string& fileNameString, const RenderGraph& renderGraph)
    {
        std::ofstream scriptFile(fileNameString);
        assert(scriptFile.is_open());
        std::string script = saveRenderGraphAsScriptBuffer(renderGraph);
        scriptFile.write(script.c_str(), script.size());
        scriptFile.close();
    }

    void RenderGraphLoader::runScript(const std::string& scriptData, RenderGraph& renderGraph)
    {
        runScript(scriptData.data(), scriptData.size(), renderGraph);
    }

    void RenderGraphLoader::runScript(const char* scriptData, size_t dataSize, RenderGraph& renderGraph)
    {
        if (!dataSize) return;
        size_t offset = 0;
        std::istringstream scriptStream(scriptData);
        std::string nextCommand;
        nextCommand.resize(255);

        // run through scriptdata
        while (scriptStream.getline(&nextCommand.front(), 255))
        {
            if (!nextCommand.front()) { break; }

            ExecuteStatement(nextCommand.substr(0, nextCommand.find_first_of('\0')), renderGraph);
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
            if (line.find_first_of('\0') == 0) { break; }

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
        std::string nextStatement = statement;
        nextStatement.erase(std::remove(nextStatement.begin(), nextStatement.end(), '\r'), nextStatement.end());

        for (;;)
        {
            charIndex = nextStatement.find_first_of(' ', lastIndex);
            if (charIndex == std::string::npos)
            {
                statementPeices.push_back(nextStatement.substr(lastIndex, nextStatement.size() - lastIndex));
                break;
            }
            statementPeices.push_back(nextStatement.substr(lastIndex, charIndex - lastIndex));
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
            
            RenderPassSerializer renderPassSerializer = RenderPassLibrary::getRenderPassSerializer(passTypeName.c_str());
            for (size_t i = 0; i < renderPassSerializer.getVariableCount(); ++i)
            {
                std::string variableName = renderPassSerializer.getVariableName(i);
                renderPassSerializer.setValue(variableName, mActiveVariables[variableName]);
            }
            
            auto pRenderPass = RenderPassLibrary::createRenderPass(passTypeName.c_str(), renderPassSerializer);

            if (pRenderPass == nullptr)
            {
                if (sSharedEditingMode)
                {
                    renderGraph.addRenderPass(DummyEditorPass::create(passTypeName), scriptBinding.mParameters[0].get<std::string>());
                }

                logWarning("Failed on attempt to create unknown pass : " + passTypeName);
                return;
            }
            renderGraph.addRenderPass(pRenderPass, scriptBinding.mParameters[0].get<std::string>());
        }, {}, {});

        RegisterStatement<std::string, std::string>("RemoveEdge", [](ScriptBinding& scriptBinding, RenderGraph& renderGraph) {
            renderGraph.removeEdge(scriptBinding.mParameters[0].get<std::string>(), scriptBinding.mParameters[1].get<std::string>());
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
            const std::string& sceneFilename = scriptBinding.mParameters[0].get<std::string>();
            Scene::SharedPtr pScene =  Scene::loadFromFile(sceneFilename);
            if (!pScene) { logWarning("Failed to load scene for current render graph"); return; }
            renderGraph.setScene(pScene); 

            mActiveVariables["gSceneFilename"] = ScriptParameter( sceneFilename );

            Scene::UserVariable var = pScene->getUserVariable("sky_box");
            assert(var.type == Scene::UserVariable::Type::String);
            mActiveVariables["gSkyBoxFilename"] = ScriptParameter( var.str );
        }, {});

#define register_var_statement(_type, _statement) \
        RegisterStatement<std::string, _type>(_statement, [](ScriptBinding& scriptBinding, RenderGraph& renderGraph) { \
            mActiveVariables[scriptBinding.mParameters[0].get<std::string>()] = scriptBinding.mParameters[1]; \
        }, {}, {});

        register_var_statement(std::string, "VarString");
        register_var_statement(uint32_t, "VarUInt");
        register_var_statement(int32_t, "VarInt");
        register_var_statement(bool, "VarBool");
        register_var_statement(double, "VarFloat");

#undef register_var_statement
    }

}
