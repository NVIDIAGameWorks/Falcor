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
#include "Framework.h"
#include "Graphics/Program/Program.h"
#include "Graphics/Program/ProgramVersion.h"
#include "ProgramVars.h"

namespace Falcor
{
    static bool compareRootSets(const DescriptorSet::Layout& a, const DescriptorSet::Layout& b)
    {
        if (a.getRangeCount() != b.getRangeCount()) return false;
        if (a.getVisibility() != b.getVisibility()) return false;
        for (uint32_t i = 0; i < a.getRangeCount(); i++)
        {
            const auto& rangeA = a.getRange(i);
            const auto& rangeB = b.getRange(i);
            if (rangeA.baseRegIndex != rangeB.baseRegIndex) return false;
            if (rangeA.descCount != rangeB.descCount) return false;
#ifdef FALCOR_D3D12
            if (rangeA.regSpace != rangeB.regSpace) return false;
#endif
            if (rangeA.type != rangeB.type) return false;
        }
        return true;
    }

    static uint32_t findRootIndex(const DescriptorSet::Layout& blockSet, const RootSignature::SharedPtr& pRootSig)
    {
        for (uint32_t i = 0; i < pRootSig->getDescriptorSetCount(); i++)
        {
            const auto& rootSet = pRootSig->getDescriptorSet(i);
            if (compareRootSets(rootSet, blockSet))
            {
#ifdef FALCOR_D3D12
                return i;
#else
                return rootSet.getRange(0).regSpace;
#endif
            }
        }
        should_not_get_here();
        return -1;
    }



    ProgramKernels::ProgramKernels(
        const ProgramReflection::SharedPtr& pReflector,
        Shader::SharedPtr const*            ppShaders,
        size_t                              shaderCount,
        const RootSignature::SharedPtr& pRootSignature,
        const std::string& name) 
        : mName(name), mpReflector(pReflector)
    {
        for( size_t i = 0; i < shaderCount; ++i )
        {
            auto pShader = ppShaders[i];
            mpShaders[(uint32_t)pShader->getType()] = pShader;
        }
        mpRootSignature = pRootSignature;

        auto parameterBlockCount = pReflector->getParameterBlockCount();
        for (uint32_t i = 0; i < parameterBlockCount; i++)
        {
            const auto& pBlockReflection = pReflector->getParameterBlock(i);
            BlockData data;

            // For each set, find the matching root-index. 
            const auto& sets = pBlockReflection->getDescriptorSetLayouts();
            data.rootIndex.resize(sets.size());
            for (size_t i = 0; i < sets.size(); i++)
            {
                data.rootIndex[i] = findRootIndex(sets[i], mpRootSignature);
            }

            mParameterBlocks.push_back(data);
        }
    }

    ProgramKernels::SharedPtr ProgramKernels::create(
        const ProgramReflection::SharedPtr& pReflector,
        Shader::SharedPtr const*            ppShaders,
        size_t                              shaderCount,
        const RootSignature::SharedPtr&     pRootSignature,
        std::string&                        log,
        const std::string&                  name)
    {
        SharedPtr pProgram = SharedPtr(new ProgramKernels(pReflector, ppShaders, shaderCount, pRootSignature, name));
        if (pProgram->init(log) == false)
        {
            return nullptr;
        }
        return pProgram;
    }

    ProgramKernels::SharedPtr ProgramKernels::create(
        ProgramReflection::SharedPtr const& pReflector,
        const Shader::SharedPtr& pVS,
        const Shader::SharedPtr& pPS,
        const Shader::SharedPtr& pGS,
        const Shader::SharedPtr& pHS,
        const Shader::SharedPtr& pDS,
        const RootSignature::SharedPtr& pRootSignature,
        std::string& log,
        const std::string& name)
    {
        // We must have at least a VS.
        if(pVS == nullptr)
        {
            log = "Program " + name + " doesn't contain a vertex-shader. This is illegal.";
            return nullptr;
        }

        static const size_t kMaxShaderCount = 5;
        Shader::SharedPtr shaders[kMaxShaderCount];
        size_t shaderCount = 0;

        if(pVS) shaders[shaderCount++] = pVS;
        if(pPS) shaders[shaderCount++] = pPS;
        if(pGS) shaders[shaderCount++] = pGS;
        if(pHS) shaders[shaderCount++] = pHS;
        if(pDS) shaders[shaderCount++] = pDS;

        SharedPtr pProgram = SharedPtr(new ProgramKernels(pReflector, shaders, shaderCount, pRootSignature, name));

        if(pProgram->init(log) == false)
        {
            return nullptr;
        }

        return pProgram;
    }

    ProgramKernels::SharedPtr ProgramKernels::create(
        const ProgramReflection::SharedPtr& pReflector,
        const Shader::SharedPtr& pCS,
        const RootSignature::SharedPtr& pRootSignature,
        std::string& log,
        const std::string& name)
    {
        // We must have at least a CS
        if (pCS == nullptr)
        {
            log = "Program " + name + " doesn't contain a compute-shader. This is illegal.";
            return nullptr;
        }
        SharedPtr pProgram = SharedPtr(new ProgramKernels(pReflector, &pCS, 1, pRootSignature, name));

        if (pProgram->init(log) == false)
        {
            return nullptr;
        }
        return pProgram;
    }

    ProgramKernels::~ProgramKernels()
    {
        deleteApiHandle();
    }

    ProgramVersion::SharedPtr ProgramVersion::create(
        std::shared_ptr<Program>     const& pProgram,
        DefineList                   const& defines,
        ProgramReflectors            const& reflectors,
        std::string                  const& name,
        SlangCompileRequest*                pSlangRequest)
    {
        return SharedPtr(new ProgramVersion(pProgram, defines, reflectors, name, pSlangRequest));
    }

    ProgramVersion::ProgramVersion(
        std::shared_ptr<Program>     const& pProgram,
        DefineList                   const& defines,
        ProgramReflectors            const& reflectors,
        std::string                  const& name,
        SlangCompileRequest*                pSlangRequest)
        : mpProgram(pProgram)
        , mDefines(defines)
        , mReflectors(reflectors)
        , mName(name)
        , mpSlangRequest(pSlangRequest)
    {
        mpKernelGraph = KernelGraph::create();
    }

    ProgramVersion::~ProgramVersion()
    {
        spDestroyCompileRequest(mpSlangRequest);
    }

    ProgramKernels::SharedConstPtr ProgramVersion::getKernels(ProgramVars const* pVars) const
    {
        // The compiled kernels will depend on the types of the parameter blocks
        // bound in `pVars`. We will walk a `Graph` to look up the right kernels
        // to use.
        //
        auto parameterBlockCount = pVars->getParameterBlockCount();
        for(uint32_t i = 0; i < parameterBlockCount; ++i)
        {
            auto blockTypeID = pVars->getParameterBlock(i)->getTypeId();
            mpKernelGraph->walk(blockTypeID);
        }

        // If we already have cached data at the chosen graph node, then use it.
        //
        auto entry = mpKernelGraph->getCurrentNode();
        if(entry.pKernels)
            return entry.pKernels;

        // Otherwise, we will construct the list of all the parameter block type IDs,
        // and use that as a key when searching for a matching graph node.
        //
        for(uint32_t i = 0; i < parameterBlockCount; ++i)
        {
            auto blockTypeID = pVars->getParameterBlock(i)->getTypeId();
            entry.parameterBlockTypes.push_back(blockTypeID);
        }

        KernelGraph::CompareFunc cmpFunc = [&](KernelGraphEntry const& other)
        {
            if(parameterBlockCount != other.parameterBlockTypes.size()) return false;
            for(uint32_t i = 0; i < parameterBlockCount; ++i)
            {
                if(entry.parameterBlockTypes[i] != other.parameterBlockTypes[i]) return false;
            }
            return true;
        };

        if(mpKernelGraph->scanForMatchingNode(cmpFunc))
        {
            entry = mpKernelGraph->getCurrentNode();
        }
        else
        {
            entry.pKernels = createKernels(pVars);
            mpKernelGraph->setCurrentNodeData(entry);
        }

        return entry.pKernels;
    }

    ProgramKernels::SharedConstPtr ProgramVersion::createKernels(ProgramVars const* pVars) const
    {
        // Loop so that user can trigger recompilation on error
        for(;;)
        {
            std::string log;
            auto pKernels = mpProgram->preprocessAndCreateProgramKernels(this, pVars, log);
            if( pKernels )
            {
                // Success.
                return pKernels;
            }
            else
            {
                // Failure
                std::string error = std::string("Program Linkage failed.\n\n");
                error += getName() + "\n";
                error += log;
                 
                if(msgBox(error, MsgBoxType::RetryCancel) == MsgBoxButton::Cancel)
                {
                    // User has chosen not to retry
                    logError(error);
                    return nullptr;
                }

                // Continue loop to keep trying...
            }
        }
    }

    ParameterBlockReflection::SharedConstPtr ProgramVersion::getParameterBlockReflectorForType(std::string const& name) const
    {
        auto ii = mParameterBlockTypes.find(name);
        if( ii != mParameterBlockTypes.end() )
        {
            return ii->second;
        }

        auto pSlangReflector = slang::ShaderReflection::get(mpSlangRequest);
        auto pSlangType = pSlangReflector->findTypeByName(name.c_str());
        auto pSlangTypeLayout = pSlangReflector->getTypeLayout(pSlangType);

        auto pTypeReflector = reflectType(pSlangTypeLayout);

        ParameterBlockReflection::SharedPtr pBlockReflector = ParameterBlockReflection::create(name, pTypeReflector);
        mParameterBlockTypes.insert(std::make_pair(name, pBlockReflector));
        return pBlockReflector;
    }

}
