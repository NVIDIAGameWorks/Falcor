/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
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
 **************************************************************************/
#include "RenderGraphIR.h"
#include "RenderGraph.h"
#include "Utils/StringUtils.h"
#include "Utils/Scripting/Scripting.h"
#include "Utils/Scripting/ScriptWriter.h"

namespace Falcor
{

RenderGraphIR::RenderGraphIR(const std::string& name, bool newGraph) : mName(name)
{
    if (newGraph)
    {
        mIR += "from pathlib import WindowsPath, PosixPath\n";
        mIR += "from falcor import *\n";
        mIR += "\n";
        mIR += "def " + getFuncName(mName) + "():\n";
        mIndentation = "    ";
        mGraphPrefix += mIndentation;
        mIR += mIndentation + "g" + " = " + ScriptWriter::makeFunc("RenderGraph", mName);
    }
    mGraphPrefix += "g.";
}

void RenderGraphIR::createPass(const std::string& passClass, const std::string& passName, const Properties& props)
{
    mIR += mGraphPrefix + ScriptWriter::makeFunc("create_pass", passName, passClass, props.toPython());
}

void RenderGraphIR::updatePass(const std::string& passName, const Properties& props)
{
    mIR += mGraphPrefix + ScriptWriter::makeFunc("update_pass", passName, props.toPython());
}

void RenderGraphIR::removePass(const std::string& passName)
{
    mIR += mGraphPrefix + ScriptWriter::makeFunc("remove_pass", passName);
}

void RenderGraphIR::addEdge(const std::string& src, const std::string& dst)
{
    mIR += mGraphPrefix + ScriptWriter::makeFunc("add_edge", src, dst);
}

void RenderGraphIR::removeEdge(const std::string& src, const std::string& dst)
{
    mIR += mGraphPrefix + ScriptWriter::makeFunc("remove_edge", src, dst);
}

void RenderGraphIR::markOutput(const std::string& name, const TextureChannelFlags mask)
{
    if (mask == TextureChannelFlags::RGB)
    {
        // Leave out mask parameter for the default case (RGB).
        mIR += mGraphPrefix + ScriptWriter::makeFunc("mark_output", name);
    }
    else
    {
        mIR += mGraphPrefix + ScriptWriter::makeFunc("mark_output", name, mask);
    }
}

void RenderGraphIR::unmarkOutput(const std::string& name)
{
    mIR += mGraphPrefix + ScriptWriter::makeFunc("unmark_output", name);
}

std::string RenderGraphIR::getFuncName(const std::string& graphName)
{
    std::string name = "render_graph_" + graphName;
    name = replaceCharacters(name, " /\\", '_');
    return name;
}

} // namespace Falcor
