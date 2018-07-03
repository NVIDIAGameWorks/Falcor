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
#include "RenderGraph.h"
#include "API/FBO.h"
#include "Utils/DirectedGraphTraversal.h"

namespace Falcor
{
    RenderGraph::SharedPtr RenderGraph::create()
    {
        try
        {
            return SharedPtr(new RenderGraph);
        }
        catch (const std::exception&)
        {
            return nullptr;
        }
    }

    RenderGraph::RenderGraph()
    {
        mpGraph = DAG::create();

        auto& pG = DAG::create();

        DirectedGraphDfsTraversal<DAG> traverser(pG, 0);

        uint32_t a = pG->addNode(nullptr);
        uint32_t b = pG->addNode(nullptr);
        uint32_t c = pG->addNode(nullptr);
        uint32_t d = pG->addNode(nullptr);
        uint32_t e = pG->addNode(nullptr);

        pG->addEdge(a, b, {});
        pG->addEdge(b, c, {});
        pG->addEdge(b, d, {});
        pG->addEdge(d, e, {});
        pG->addEdge(a, e, {});

        traverser.reset(0);

        uint32_t n = 0;
        n = traverser.traverse();
        n = traverser.traverse();
        n = traverser.traverse();
        n = traverser.traverse();
        n = traverser.traverse();
        n = traverser.traverse();
        n = traverser.traverse();
        n = traverser.traverse();
    }

    uint32_t RenderGraph::getPassIndex(const std::string& name) const
    {
        auto& it = mNameToIndex.find(name);
        return (it == mNameToIndex.end()) ? kInvalidIndex : it->second;
    }

    void RenderGraph::setScene(const std::shared_ptr<Scene>& pScene)
    {
        mpScene = pScene;
        for (auto& it : mNameToIndex)
        {
            (*mpGraph->getNodeData(it.second))->setScene(pScene);
        }
    }

    bool RenderGraph::addRenderPass(const RenderPass::SharedPtr& pPass, const std::string& passName)
    {
        assert(pPass);
        if (getPassIndex(passName) != kInvalidIndex)
        {
            logWarning("Pass named `" + passName + "' already exists. Pass names must be unique");
            return false;
        }

        pPass->setScene(mpScene);
        mNameToIndex[passName] = mpGraph->addNode(pPass);
        mRecompile = true;
        return true;
    }

    void RenderGraph::removeRenderPass(const std::string& name)
    {
        uint32_t index = getPassIndex(name);
        if (index == kInvalidIndex)
        {
            logWarning("Can't remove pass `" + name + "`. Pass doesn't exist");
            return;
        }

        // Update the indices
        mNameToIndex.erase(name);

        // Remove all the edges associated with this pass
        mpGraph->removeNode(index);
        mRecompile = true;
    }

    const RenderPass::SharedPtr& RenderGraph::getRenderPass(const std::string& name) const
    {
        uint32_t index = getPassIndex(name);
        if (index == kInvalidIndex)
        {
            static RenderPass::SharedPtr pNull;
            logWarning("RenderGraph::getRenderPass() - can't find a pass named `" + name + "`");
            return pNull;
        }
        return (*mpGraph->getNodeData(index));
    }
    
    using str_pair = std::pair<std::string, std::string>;
    
    template<bool input>
    static bool checkRenderPassIoExist(const RenderPass* pPass, const std::string& name)
    {
        const auto& ioVec = input ? pPass->getRenderPassData().inputs : pPass->getRenderPassData().outputs;
        for (const auto& f : ioVec)
        {
            if (f.name == name) return true;
        }
        return false;
    }

    static bool parseFieldName(const std::string& fullname, str_pair& strPair)
    {
        if (std::count(fullname.begin(), fullname.end(), '.') != 1)
        {
            logWarning("RenderGraph node field string is incorrect. Must be in the form of `PassName.FieldName` but got `" + fullname + "`");
            return false;
        }

        size_t dot = fullname.find_first_of('.');
        strPair.first = fullname.substr(0, dot);
        strPair.second = fullname.substr(dot + 1);
        return true;
    }

    template<bool input>
    static RenderPass* getRenderPassAndNamePair(const RenderGraph* pGraph, const std::string& fullname, const std::string& errorPrefix, str_pair& nameAndField)
    {
        if (parseFieldName(fullname, nameAndField) == false) return false;

        RenderPass* pPass = pGraph->getRenderPass(nameAndField.first).get();
        if (!pPass)
        {
            logWarning(errorPrefix + " - can't find render-pass named '" + nameAndField.first + "'");
            return nullptr;
        }

        if (checkRenderPassIoExist<input>(pPass, nameAndField.second) == false)
        {
            logWarning(errorPrefix + "- can't find field named `" + nameAndField.second + "` in render-pass `" + nameAndField.first + "`");
            return nullptr;
        }
        return pPass;
    }

    bool RenderGraph::addEdge(const std::string& src, const std::string& dst)
    {
        EdgeData newEdge;
        str_pair srcPair, dstPair;
        const auto& pSrc = getRenderPassAndNamePair<false>(this, src, "Invalid src string in RenderGraph::addEdge()", srcPair);
        const auto& pDst = getRenderPassAndNamePair<true>(this, dst, "Invalid dst string in RenderGraph::addEdge()", dstPair);
        newEdge.srcField = srcPair.second;
        newEdge.dstField = dstPair.second;

        if (pSrc == nullptr || pDst == nullptr) return false;

        // Check that the dst field is not already initialized
        const DAG::Node* pNode = mpGraph->getNode(mNameToIndex[dstPair.first]);

        for (uint32_t e = 0 ; e < pNode->getOutgoingEdgeCount() ; e++)
        {
            const auto& pEdgeData = mpGraph->getEdgeData(pNode->getIncomingEdge(e));
            if (pEdgeData->dstField == newEdge.dstField)
            {
                logWarning("RenderGraph::addEdge() - destination `" + dst + "` is already initialized. Please remove the existing connection before trying to add an edge");
                return false;
            }
        }
        
        mpGraph->addEdge(mNameToIndex[srcPair.first], mNameToIndex[dstPair.first], newEdge);
        mRecompile = true;
        return true;
    }

    bool RenderGraph::isValid(std::string& log) const
    {
        bool valid = true;
        size_t logSize = log.size();

        for (const auto& passIndex : mNameToIndex)
        {
            RenderPass* pPass = (*mpGraph->getNodeData(passIndex.second)).get();
            if (pPass->isValid(log) == false)
            {
                valid = false;
                if (log.size() != logSize && log.back() != '\n')
                {
                    log += '\n';
                    logSize = log.size();
                }
            }
        }
        return valid;
    }

    Texture::SharedPtr RenderGraph::createTextureForPass(const RenderPass::PassData::Field& field)
    {
        uint32_t width = field.width ? field.width : mSwapChainData.width;
        uint32_t height = field.height ? field.height : mSwapChainData.height;
        uint32_t depth = field.depth ? field.depth : 1;
        uint32_t sampleCount = field.sampleCount ? field.sampleCount : 1;
        ResourceFormat format = field.format == ResourceFormat::Unknown ? mSwapChainData.colorFormat : field.format;
        Texture::SharedPtr pTexture;

        if (depth > 1)
        {
            assert(sampleCount == 1);
            pTexture = Texture::create3D(width, height, depth, format, 1, nullptr, field.bindFlags | Resource::BindFlags::ShaderResource);
        }
        else if (height > 1 || sampleCount > 1)
        {
            if (sampleCount > 1)
            {
                pTexture = Texture::create2DMS(width, height, format, sampleCount, 1, field.bindFlags | Resource::BindFlags::ShaderResource);
            }
            else
            {
                pTexture = Texture::create2D(width, height, format, 1, 1, nullptr, field.bindFlags | Resource::BindFlags::ShaderResource);
            }
        }
        else
        {
            pTexture = Texture::create1D(width, format, 1, 1, nullptr, field.bindFlags | Resource::BindFlags::ShaderResource);
        }

        return pTexture;
    }

    void RenderGraph::compile()
    {
        if(mRecompile)
        {   
            for (const auto& passIndex : mNameToIndex)
            {
                const DAG::Node* pNode = mpGraph->getNode(passIndex.second);
                RenderPass* pSrcPass = pNode->getData().get();
                const RenderPass::PassData& passData = pSrcPass->getRenderPassData();

                // Allocate everything that is required
                for (const auto& src : passData.outputs)
                {
                    if (src.required)
                    {
                        // Only allocate it if the user didn't set it
                        if(pSrcPass->getOutput(src.name) == nullptr)
                        {
                            Texture::SharedPtr pTexture = createTextureForPass(src);
                            pSrcPass->setOutput(src.name, pTexture);
                        }
                    }
                }

                // Now go over the edges, allocate the required resources and attach them to the input pass
                for (uint32_t e = 0; e < pNode->getOutgoingEdgeCount(); e++)
                {
                    const auto& pEdge = mpGraph->getEdge(pNode->getOutgoingEdge(e));
                    const auto& edgeData = pEdge->getData();

                    // Find the input
                    for (const auto& src : passData.outputs)
                    {
                        if (src.name == edgeData.srcField)
                        {
                            Texture::SharedPtr pTexture = createTextureForPass(src);
                            pSrcPass->setOutput(src.name, pTexture);

                            // Connect it to the dst pass
                            RenderPass* pDstPass = (*mpGraph->getNodeData(pEdge->getDestNode())).get();
                            pDstPass->setInput(edgeData.dstField, pTexture);
                            break;
                        }
                        else should_not_get_here();
                    }
                }
            }
        }
        mRecompile = false;
    }

    void RenderGraph::execute(RenderContext* pContext)
    {
        compile();

        std::string log;
        if (!isValid(log))
        {
            logWarning("Failed to compile RenderGraph\n" + log +"Ignoreing RenderGraph::execute() call");
            return;
        }

        for (const auto& passIndex : mNameToIndex)
        {
            (*mpGraph->getNodeData(passIndex.second))->execute(pContext);
        }
    }

    bool RenderGraph::setInput(const std::string& name, const std::shared_ptr<Resource>& pResource)
    {
        str_pair strPair;
        RenderPass* pPass = getRenderPassAndNamePair<true>(this, name, "RenderGraph::setInput()", strPair);
        if (pPass == nullptr) return false;
        return pPass->setInput(strPair.second, pResource);
    }

    bool RenderGraph::setOutput(const std::string& name, const std::shared_ptr<Resource>& pResource)
    {
        str_pair strPair;
        RenderPass* pPass = getRenderPassAndNamePair<false>(this, name, "RenderGraph::setOutput()", strPair);
        if (pPass == nullptr) return false;
        if (pPass->setOutput(strPair.second, pResource) == false) return false;
        markGraphOutput(name);
        return true;
    }

    void RenderGraph::markGraphOutput(const std::string& name)
    {
        str_pair strPair;
        const auto& pPass = getRenderPassAndNamePair<false>(this, name, "RenderGraph::markGraphOutput()", strPair);
        if (pPass == nullptr) return;

        GraphOut newOut;
        newOut.field = strPair.second;
        newOut.nodeId = mNameToIndex[strPair.first];

        // Check that this is not already marked
        for (const auto& o : mOutputs)
        {
            if (newOut.nodeId == o.nodeId && newOut.field == o.field) return;
        }

        mOutputs.push_back(newOut);
        mRecompile = true;
    }

    void RenderGraph::unmarkGraphOutput(const std::string& name)
    {
        str_pair strPair;
        const auto& pPass = getRenderPassAndNamePair<false>(this, name, "RenderGraph::unmarkGraphOutput()", strPair);
        if (pPass == nullptr) return;

        GraphOut removeMe;
        removeMe.field = strPair.second;
        removeMe.nodeId = mNameToIndex[strPair.first];

        for (size_t i = 0 ; i < mOutputs.size() ; i++)
        {
            if (mOutputs[i].nodeId == removeMe.nodeId && mOutputs[i].field == removeMe.field)
            {
                mOutputs.erase(mOutputs.begin() + i);
                mRecompile = true;
                return;
            }
        }
    }

    const Resource::SharedPtr RenderGraph::getOutput(const std::string& name)
    {
        static const Resource::SharedPtr pNull;
        str_pair strPair;
        RenderPass* pPass = getRenderPassAndNamePair<false>(this, name, "RenderGraph::getOutput()", strPair);
        
        return pPass ? pPass->getOutput(strPair.second) : pNull;
    }

    void RenderGraph::onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height)
    {
        // Store the back-buffer values
        const Fbo* pFbo = pSample->getCurrentFbo().get();
        const Texture* pColor = pFbo->getColorTexture(0).get();
        const Texture* pDepth = pFbo->getDepthStencilTexture().get();
        assert(pColor && pDepth);

        // If the back-buffer values changed, recompile
        mRecompile = mRecompile || (mSwapChainData.colorFormat != pColor->getFormat());
        mRecompile = mRecompile || (mSwapChainData.depthFormat != pDepth->getFormat());
        mRecompile = mRecompile || (mSwapChainData.width != width);
        mRecompile = mRecompile || (mSwapChainData.height != height);

        // Store the values
        mSwapChainData.colorFormat = pColor->getFormat();
        mSwapChainData.depthFormat = pDepth->getFormat();
        mSwapChainData.width = width;
        mSwapChainData.height = height;

        // Invoke the passes' callback
        for (auto& passIndex : mNameToIndex)
        {
            (*mpGraph->getNodeData(passIndex.second))->onResizeSwapChain(pSample, width, height);
        }
    }
}