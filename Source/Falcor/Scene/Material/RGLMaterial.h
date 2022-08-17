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
#include "RGLMaterialData.slang"
#include <filesystem>

namespace Falcor
{
    /** Class representing a measured material from the RGL BRDF database.

        For details refer to:
        Jonathan Dupuy, Wenzel Jakob
        "An Adaptive Parameterization for Efficient Material Acquisition and Rendering".
        Transactions on Graphics (Proc. SIGGRAPH Asia 2018)
    */
    class FALCOR_API RGLMaterial : public Material
    {
    public:
        using SharedPtr = std::shared_ptr<RGLMaterial>;

        /** Create a new RGL material.
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

        virtual int getBufferCount() const override { return 12; }

        bool loadBRDF(const std::filesystem::path& path);

    protected:
        RGLMaterial(const std::string& name, const std::filesystem::path& path);

        void prepareData(const int dims[3], const std::vector<double>& data);
        void prepareAlbedoLUT(RenderContext* pRenderContext);
        void computeAlbedoLUT(RenderContext* pRenderContext);

        std::filesystem::path mFilePath;    ///< Full path to the BRDF loaded.
        std::string mBRDFName;              ///< This is the file basename without extension.
        std::string mBRDFDescription;       ///< Description of the BRDF given in the BRDF file.

        bool mBRDFUploaded = false;         ///< True if BRDF data buffers have been uploaded to the material system.
        RGLMaterialData mData;              ///< Material parameters.
        Buffer::SharedPtr mpThetaBuf;
        Buffer::SharedPtr mpPhiBuf;
        Buffer::SharedPtr mpSigmaBuf;
        Buffer::SharedPtr mpNDFBuf;
        Buffer::SharedPtr mpVNDFBuf;
        Buffer::SharedPtr mpLumiBuf;
        Buffer::SharedPtr mpRGBBuf;
        Buffer::SharedPtr mpVNDFMarginalBuf;
        Buffer::SharedPtr mpLumiMarginalBuf;
        Buffer::SharedPtr mpVNDFConditionalBuf;
        Buffer::SharedPtr mpLumiConditionalBuf;
        Texture::SharedPtr mpAlbedoLUT;     ///< Precomputed albedo lookup table.
        Sampler::SharedPtr mpSampler;       ///< Sampler for accessing BRDF textures.

        ComputePass::SharedPtr mBRDFTesting;
    };
}
