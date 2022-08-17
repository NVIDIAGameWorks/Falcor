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
#include "Material.h"
#include "MERLMaterialData.slang"

namespace Falcor
{
    /** Class representing a measured material from the MERL BRDF database.

        For details refer to:
        Wojciech Matusik, Hanspeter Pfister, Matt Brand and Leonard McMillan.
        "A Data-Driven Reflectance Model". ACM Transactions on Graphics,
        vol. 22(3), 2003, pages 759-769.
    */
    class FALCOR_API MERLMaterial : public Material
    {
    public:
        using SharedPtr = std::shared_ptr<MERLMaterial>;

        /** Create a new MERL material.
            \param[in] name The material name.
            \param[in] path Path of BRDF file to load.
            \return A new object, or throws an exception if creation failed.
        */
        static SharedPtr create(const std::string& name, const std::filesystem::path& path);

        bool renderUI(Gui::Widgets& widget) override;
        Material::UpdateFlags update(MaterialSystem* pOwner) override;
        bool isEqual(const Material::SharedPtr& pOther) const override;
        MaterialDataBlob getDataBlob() const override { return prepareDataBlob(mData); }
        Program::ShaderModuleList getShaderModules() const override;
        Program::TypeConformanceList getTypeConformances() const override;

        int getBufferCount() const override { return 1; }

    protected:
        MERLMaterial(const std::string& name, const std::filesystem::path& path);

        bool loadBRDF(const std::filesystem::path& path);
        void prepareData(const int dims[3], const std::vector<double>& data);
        void prepareAlbedoLUT(RenderContext* pRenderContext);
        void computeAlbedoLUT(RenderContext* pRenderContext);

        std::filesystem::path mPath;        ///< Full path to the BRDF loaded.
        std::string mBRDFName;              ///< This is the file basename without extension.

        MERLMaterialData mData;             ///< Material parameters.
        Buffer::SharedPtr mpBRDFData;       ///< GPU buffer holding all BRDF data as float3 array.
        Texture::SharedPtr mpAlbedoLUT;     ///< Precomputed albedo lookup table.
        Sampler::SharedPtr mpLUTSampler;    ///< Sampler for accessing the LUT texture.
    };
}
