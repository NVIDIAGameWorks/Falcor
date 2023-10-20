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
#include "RenderPass.h"

namespace Falcor
{
RenderData::RenderData(
    const std::string& passName,
    ResourceCache& resources,
    Dictionary& dictionary,
    const uint2& defaultTexDims,
    ResourceFormat defaultTexFormat
)
    : mName(passName), mResources(resources), mDictionary(dictionary), mDefaultTexDims(defaultTexDims), mDefaultTexFormat(defaultTexFormat)
{}

const ref<Resource>& RenderData::getResource(const std::string_view name) const
{
    return mResources.getResource(fmt::format("{}.{}", mName, name));
}

ref<Texture> RenderData::getTexture(const std::string_view name) const
{
    auto pResource = getResource(name);
    return pResource ? pResource->asTexture() : nullptr;
}

ref<RenderPass> RenderPass::create(std::string_view type, ref<Device> pDevice, const Properties& props, PluginManager& pm)
{
    // Try to load a plugin of the same name, if render pass class is not registered yet.
    if (!pm.hasClass<RenderPass>(type))
        pm.loadPluginByName(type);

    return pm.createClass<RenderPass>(type, pDevice, props);
}
} // namespace Falcor
