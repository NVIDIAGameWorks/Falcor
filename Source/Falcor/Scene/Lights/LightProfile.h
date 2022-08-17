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

#include "Core/Macros.h"
#include "Core/API/Texture.h"
#include "Core/API/Sampler.h"
#include "Utils/UI/Gui.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace Falcor
{
    struct ShaderVar;

    class FALCOR_API LightProfile
    {
    public:
        using SharedPtr = std::shared_ptr<LightProfile>;

        static SharedPtr createFromIesProfile(const std::filesystem::path& filename, bool normalize);

        void bake(RenderContext* pRenderContext);

        /** Set the light profile into a shader var.
        */
        void setShaderData(const ShaderVar& var) const;

        /** Render the UI.
        */
        void renderUI(Gui::Widgets& widget);

    private:
        LightProfile(const std::string& name, const std::vector<float>& rawData);

        std::string mName;
        std::vector<float> mRawData;
        Texture::SharedPtr mpTexture;
        Sampler::SharedPtr mpSampler;
        float mFluxFactor = 0.f;
    };
}
