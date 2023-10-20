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
#pragma once
#include "Material.h"
#include "MERLMaterialData.slang"

namespace Falcor
{
    class MERLFile;

    /** Class representing a measured material from the MERL BRDF database.

        For details refer to:
        Wojciech Matusik, Hanspeter Pfister, Matt Brand and Leonard McMillan.
        "A Data-Driven Reflectance Model". ACM Transactions on Graphics,
        vol. 22(3), 2003, pages 759-769.
    */
    class FALCOR_API MERLMaterial : public Material
    {
        FALCOR_OBJECT(MERLMaterial)
    public:
        static ref<MERLMaterial> create(ref<Device> pDevice, const std::string& name, const std::filesystem::path& path) { return make_ref<MERLMaterial>(pDevice, name, path); }

        MERLMaterial(ref<Device> pDevice, const std::string& name, const std::filesystem::path& path);
        MERLMaterial(ref<Device> pDevice, const MERLFile& merlFile);

        bool renderUI(Gui::Widgets& widget) override;
        Material::UpdateFlags update(MaterialSystem* pOwner) override;
        bool isEqual(const ref<Material>& pOther) const override;
        MaterialDataBlob getDataBlob() const override { return prepareDataBlob(mData); }
        ProgramDesc::ShaderModuleList getShaderModules() const override;
        TypeConformanceList getTypeConformances() const override;

        size_t getMaxBufferCount() const override { return 1; }

    protected:
        void init(const MERLFile& merlFile);

        std::filesystem::path mPath;        ///< Full path to the BRDF loaded.
        std::string mBRDFName;              ///< This is the file basename without extension.

        MERLMaterialData mData;             ///< Material parameters.
        ref<Buffer> mpBRDFData;             ///< GPU buffer holding all BRDF data as float3 array.
        ref<Texture> mpAlbedoLUT;           ///< Precomputed albedo lookup table.
        ref<Sampler> mpLUTSampler;          ///< Sampler for accessing the LUT texture.
    };
}
