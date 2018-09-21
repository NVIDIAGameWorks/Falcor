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
#include "RenderGraphIR.h"
#include "RenderGraph.h"
#include "RenderGraphScripting.h"
#include "Utils/Dictionary.h"

namespace Falcor
{
    std::string addQuotes(const std::string& s)
    {
        return '"' + s + '"';
    }

    std::string getArgsString()
    {
        return "";
    }

    template <class T>
    std::string getArgsString(const T& arg)
    {
        return arg;
    }

    template <class T, class... Ts>
    std::string getArgsString(const T& first, const Ts&... args) 
    {
        std::string s = first + ", ";
        s += getArgsString(args...);
        return s;
    }

    template<typename... Ts>
    std::string funcCall(const std::string& funcName, const Ts&... args)
    {
        std::string call = funcName + '(';
        call += getArgsString(args...);
        call += ")\n";
        return call;
    }

    RenderGraphIR::RenderGraphIR(const std::string& name, bool newGraph) : mName(name)
    {
        if(newGraph)
        {
            mIR += "def render_graph_" + mName + "():\n";
            mIndentation = "\t";
            mGraphPrefix += mIndentation;
            mIR += mIndentation + mName + " = " + funcCall(RenderGraphScripting::kCreateGraph);
        }
        mGraphPrefix += mName + '.';
    };

    RenderGraphIR::SharedPtr RenderGraphIR::create(const std::string& name, bool newGraph)
    {
        return SharedPtr(new RenderGraphIR(name, newGraph));
    }

    void RenderGraphIR::addPass(const std::string& passClass, const std::string& passName, const Dictionary& dictionary)
    {
        mIR += mIndentation + passName + " = ";
        if(dictionary.size())
        {
            std::string dictionaryStr = dictionary.toString();
            mIR += funcCall(RenderGraphScripting::kCreatePass, addQuotes(passClass), dictionaryStr);
        }
        else
        {
            mIR += funcCall(RenderGraphScripting::kCreatePass, addQuotes(passClass));
        }
        mIR += mGraphPrefix + funcCall(RenderGraphScripting::kAddPass, passName, addQuotes(passName));
    }

    void RenderGraphIR::updatePass(const std::string& passName, const Dictionary& dictionary)
    {
        mIR += mGraphPrefix + funcCall(RenderGraphScripting::kUpdatePass, addQuotes(passName), dictionary.toString());
    }

    void RenderGraphIR::removePass(const std::string& passName)
    {
        mIR += mGraphPrefix + funcCall(RenderGraphScripting::kRemovePass, addQuotes(passName));
    }

    void RenderGraphIR::addEdge(const std::string& src, const std::string& dst)
    {
        mIR += mGraphPrefix + funcCall(RenderGraphScripting::kAddEdge, addQuotes(src), addQuotes(dst));
    }

    void RenderGraphIR::removeEdge(const std::string& src, const std::string& dst)
    {
        mIR += mGraphPrefix + funcCall(RenderGraphScripting::kRemoveEdge, addQuotes(src), addQuotes(dst));
    }

    void RenderGraphIR::markOutput(const std::string& name)
    {
        mIR += mGraphPrefix + funcCall(RenderGraphScripting::kMarkOutput, addQuotes(name));
    }

    void RenderGraphIR::unmarkOutput(const std::string& name)
    {
        mIR += mGraphPrefix + funcCall(RenderGraphScripting::kUnmarkOutput, addQuotes(name));
    }

    void RenderGraphIR::autoGenEdges()
    {
        mIR += mGraphPrefix + funcCall(RenderGraphScripting::kAutoGenEdges);
    }
}