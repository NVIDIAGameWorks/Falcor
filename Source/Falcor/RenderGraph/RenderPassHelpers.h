/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "RenderPassReflection.h"
#include "Core/Macros.h"
#include "Core/API/Formats.h"
#include "Core/API/RenderContext.h"
#include "Core/Program/Program.h"
#include "Utils/UI/Gui.h"
#include <string>
#include <vector>

namespace Falcor
{
    struct FALCOR_API RenderPassHelpers
    {
        /** Enum for commonly used render pass I/O sizes.
        */
        enum class IOSize : uint32_t
        {
            Default,    ///< Use the default size. The size is determined based on whatever is bound (the system will use the window size by default).
            Fixed,      ///< Use fixed size in pixels.
            Full,       ///< Use full window size.
            Half,       ///< Use half window size.
            Quarter,    ///< Use quarter window size.
            Double,     ///< Use double window size.
        };

        /** UI dropdown for the IOSize enum values.
        */
        static inline Gui::DropdownList kIOSizeList =
        {
            { (uint32_t)IOSize::Default, "Default" },
            { (uint32_t)IOSize::Fixed, "Fixed" },
            { (uint32_t)IOSize::Full, "Full window" },
            { (uint32_t)IOSize::Half, "Half window" },
            { (uint32_t)IOSize::Quarter, "Quarter window" },
            { (uint32_t)IOSize::Double, "Double window" },
        };

        /** Helper for calculating desired I/O size in pixels based on selected mode.
        */
        static uint2 calculateIOSize(const IOSize selection, const uint2 fixedSize, const uint2 windowSize);
    };

    // TODO: Move below out of the global scope, e.g. into RenderPassHelpers struct.
    // TODO: Update render passes to use addRenderPass*() helpers.

    /** Helper struct with metadata for a render pass input/output.
    */
    struct ChannelDesc
    {
        std::string name;       ///< Render pass I/O pin name.
        std::string texname;    ///< Name of corresponding resource in the shader, or empty if it's not a shader variable.
        std::string desc;       ///< Human-readable description of the data.
        bool optional = false;  ///< Set to true if the resource is optional.
        ResourceFormat format = ResourceFormat::Unknown; ///< Default format is 'Unknown', which means let the system decide.
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
        \param[in] dim Optional dimension. The default (0,0) means use whatever is bound (the system will use the window size by default).
    */
    inline void addRenderPassInputs(
        RenderPassReflection& reflector,
        const ChannelList& channels,
        ResourceBindFlags bindFlags = ResourceBindFlags::ShaderResource,
        const uint2 dim = {})
    {
        for (const auto& it : channels)
        {
            auto& tex = reflector.addInput(it.name, it.desc).texture2D(dim.x, dim.y);
            tex.bindFlags(bindFlags);
            if (it.format != ResourceFormat::Unknown) tex.format(it.format);
            if (it.optional) tex.flags(RenderPassReflection::Field::Flags::Optional);
        }
    }

    /** Adds a list of output channels to the render pass reflection.
        \param[in] reflector Render pass reflection object.
        \param[in] channels List of channels.
        \param[in] bindFlags Optional bind flags. The default is 'UnorderedAccess' for all outputs.
        \param[in] dim Optional dimension. The default (0,0) means use whatever is bound (the system will use the window size by default).
    */
    inline void addRenderPassOutputs(
        RenderPassReflection& reflector,
        const ChannelList& channels,
        ResourceBindFlags bindFlags = ResourceBindFlags::UnorderedAccess,
        const uint2 dim = {})
    {
        for (const auto& it : channels)
        {
            auto& tex = reflector.addOutput(it.name, it.desc).texture2D(dim.x, dim.y);
            tex.bindFlags(bindFlags);
            if (it.format != ResourceFormat::Unknown) tex.format(it.format);
            if (it.optional) tex.flags(RenderPassReflection::Field::Flags::Optional);
        }
    }

    /** Clears all available channels.
        \param[in] pRenderContext Render context.
        \param[in] channels List of channel descriptors.
        \param[in] renderData Render data containing the channel resources.
    */
    inline void clearRenderPassChannels(RenderContext* pRenderContext, const ChannelList& channels, const RenderData& renderData)
    {
        for (const auto& channel : channels)
        {
            auto pTex = renderData.getTexture(channel.name);
            if (pTex)
            {
                if (isIntegerFormat(pTex->getFormat()))
                {
                    pRenderContext->clearUAV(pTex->getUAV().get(), uint4(0));
                }
                else
                {
                    pRenderContext->clearUAV(pTex->getUAV().get(), float4(0.f));
                }
            }
        }
    }
}
