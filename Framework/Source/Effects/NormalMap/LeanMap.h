/***************************************************************************
# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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
#pragma once
#include <map>
#include <memory>
#include "API/Texture.h"
#include "API/Sampler.h"

namespace Falcor
{
    class Texture;
    class Scene;
    class Material;
    class ProgramVars;

    class LeanMap
    {
    public:
        using UniquePtr = std::unique_ptr<LeanMap>;

        /** Create Lean maps from materials used in a scene
        */
        static UniquePtr create(const Falcor::Scene* pScene);

        /** Create a Lean map from a normal map
        */
        static Falcor::Texture::SharedPtr createFromNormalMap(const Falcor::Texture* pNormalMap);

        /** Get a generated Lean map.
            \param[in] sceneMaterialID Material ID to get Lean map for. Use Material::getId.
        */
        Falcor::Texture* getLeanMap(uint32_t sceneMaterialID) { return mpLeanMaps[sceneMaterialID].get(); }

        /** Set Lean map texture into a program.
            \param[in] pVars Program vars to set into
            \param[in] pSampler Sampler to use when sampling the Lean map
        */
        void setIntoProgramVars(ProgramVars* pVars, const Sampler::SharedPtr& pSampler) const;

        /** Get the array size required in the shader to hold Lean maps. Lean map index in shaders
            match 1:1 with the material ID, but Lean maps for a contiguous range of materials may not have been generated.
        */
        uint32_t getRequiredLeanMapShaderArraySize() const { return mShaderArraySize; }

    private:
        LeanMap() = default;
        bool createLeanMap(const Falcor::Material* pMaterial);
        std::map<uint32_t, Falcor::Texture::SharedPtr> mpLeanMaps;
        uint32_t mShaderArraySize = 0;
    };
}