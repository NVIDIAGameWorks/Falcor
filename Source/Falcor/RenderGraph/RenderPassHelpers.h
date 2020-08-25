/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#pragma once

#include "RenderPass.h"
#include "Core/Program/Program.h"

namespace Falcor
{
    // TODO: Move this out of the global scope, e.g. into a RenderPassHelpers class.
    // TODO: Update render passes to use addRenderPass*() helpers.

    /** Helper struct with metadata for a render pass input/output.
    */
    struct ChannelDesc
    {
        std::string name;       ///< Render pass I/O pin name.
        std::string texname;    ///< Name of corresponding resource in the shader, or empty if it's not a shader variable.
        std::string desc;       ///< Human-readable description of the data.
        bool optional = false;  ///< Set to true if the resource is optional.
        ResourceFormat format = ResourceFormat::RGBA32Float;
    };

    using ChannelList = std::vector<ChannelDesc>;

    /** Creates a list of defines to determine if optional render pass resources are valid to be accessed.
        This function creates a define for every optional channel in the form of:

            #define <prefix><desc.texname> 1       if resource is available
            #define <prefix><desc.texname> 0       otherwise

        \param[in] channels List of channel descriptors.
        \param[in] renderData Render data containing the channel resources.
        \param[in] prefix Prefix used for defines.
        \return Returns a list of defines to add to the progrem.
    */
    inline Program::DefineList getValidResourceDefines(const ChannelList& channels, const RenderData& renderData, const std::string& prefix = "is_valid_")
    {
        Program::DefineList defines;

        for (const auto& desc : channels)
        {
            if (desc.optional && !desc.texname.empty())
            {
                defines.add(prefix + desc.texname, renderData[desc.name] != nullptr ? "1" : "0");
            }
        }

        return defines;
    }

    /** Adds a list of input channels to the render pass reflection.
        \param[in] reflector Render pass reflection object.
        \param[in] channels List of channels.
        \param[in] bindFlags Optional bind flags. The default is 'ShaderResource' for all inputs.
    */
    inline void addRenderPassInputs(RenderPassReflection& reflector, const ChannelList& channels, ResourceBindFlags bindFlags = ResourceBindFlags::ShaderResource)
    {
        for (const auto& it : channels)
        {
            auto& buffer = reflector.addInput(it.name, it.desc);
            buffer.bindFlags(bindFlags);
            if (it.format != ResourceFormat::Unknown) buffer.format(it.format);
            if (it.optional) buffer.flags(RenderPassReflection::Field::Flags::Optional);
        }
    }

    /** Adds a list of output channels to the render pass reflection.
        \param[in] reflector Render pass reflection object.
        \param[in] channels List of channels.
        \param[in] bindFlags Optional bind flags. The default is 'UnorderedAccess' for all outputs.
    */
    inline void addRenderPassOutputs(RenderPassReflection& reflector, const ChannelList& channels, ResourceBindFlags bindFlags = ResourceBindFlags::UnorderedAccess)
    {
        for (const auto& it : channels)
        {
            auto& buffer = reflector.addOutput(it.name, it.desc);
            buffer.bindFlags(bindFlags);
            if (it.format != ResourceFormat::Unknown) buffer.format(it.format);
            if (it.optional) buffer.flags(RenderPassReflection::Field::Flags::Optional);
        }
    }
}
