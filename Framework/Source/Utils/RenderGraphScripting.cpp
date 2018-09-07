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
#include "RenderGraphScriptContext.h"
#include "Framework.h"
#include "Scripting.h"
#include <fstream>
#include <sstream>

namespace Falcor
{
    RenderGraphScripting::SharedPtr RenderGraphScripting::create()
    {
        return SharedPtr(new RenderGraphScripting());
    }

    RenderGraphScripting::SharedPtr RenderGraphScripting::create(const std::string& filename)
    {
        SharedPtr pThis = create();

        if (findFileInDataDirectories(filename, pThis->mFilename) == false)
        {
            logError("Error when importing render-graphs. Can't find the file `" + filename + "`");
            return nullptr;
        }

        pThis->runScript(readFile(pThis->mFilename));
        return pThis;
    }

    RenderGraphScripting::GraphVec RenderGraphScripting::importGraphsFromFile(const std::string& filename)
    {
        SharedPtr pThis = create(filename);
        return pThis ? pThis->getGraphs() : GraphVec();
    }

    bool RenderGraphScripting::runScript(const std::string& script)
    {
        std::string log;
        if (Scripting::runScript(script, log, mContext) == false)
        {
            logError("Can't run render-graphs script.\n" + log);
            return false; 
        }

        mGraphVec = mContext.getObjects<RenderGraph::SharedPtr>();
        return true;
    }

    void RenderGraphScripting::addGraph(const std::string& name, const RenderGraph::SharedPtr& pGraph)
    {
        try
        {
            mContext.getObject<RenderGraph::SharedPtr>(name);
            logWarning("RenderGraph `" + name + "` already exists. Replacing the current object");
        }
        catch (std::exception) {}
        mContext.setObject(name, pGraph);
    }

    RenderGraph::SharedPtr RenderGraphScripting::getGraph(const std::string& name) const
    {
        try
        {
            return mContext.getObject<RenderGraph::SharedPtr>(name);
        }
        catch (std::exception) 
        {
            logWarning("Can't find RenderGraph `" + name + "` in RenderGraphScriptContext");
            return nullptr;
        }
    }
}