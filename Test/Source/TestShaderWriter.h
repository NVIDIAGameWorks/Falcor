/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
#include "Falcor.h"
#include <string>
#include <map>
#include <vector>
#include <set>
#include <random>
#include <algorithm>    // std::random_shuffle
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand

using namespace Falcor;

class TestShaderWriter 
{

public:


    //  The Output Target.
    struct ShaderTarget
    {
        uint32_t targetIndex;
    };

    //  
    struct StructVariable
    {
        //  Whether or not the variable is used.
        bool isUsed = true;

        //  The name of the variable in the struct.
        std::string structVariable;

        //  The name of the variable as used in the main function.
        std::string useStructVariable;

        //  The Type of the Variable.
        ProgramReflection::Variable::Type variableType;
    };


    struct ShaderResource
    {
        //  The Variable used in the Generated Shader Code.
        std::string variable;

        //  Flag for Whether this Resource should be used.
        bool isUsedResource = false;

        //  Flag for Explicit Index.
        bool isExplicitIndex = false;

        //  The Register Index.
        uint32_t regIndex = 0;

        //  Flag for Explicit Space.
        bool isExplicitSpace = false;

        //  The Space Index.
        uint32_t regSpace = 0;


        //  Whether this is an array.
        bool isArray = false;

        //  Whether the array is Unbounded.
        bool isUnbounded = false;

        //  Array Sizes.
        std::vector<uint32_t> arrayDimensions;

        //  Whether it is within a Constant Buffer.
        bool isWithinConstantBuffer = false;

        //  The Constant Buffer it is inside, by name.
        std::string cbVariable = "";

    };


    //  Texture Resource,
    struct TextureResource : public ShaderResource
    {

        //  The name of the variable as used in the main function.
        std::string useTextureVariable;

        //
        ProgramReflection::Variable::Type textureType;

        //  The Texture Resource Type.
        ProgramReflection::Resource::Dimensions textureResourceType;

        //  Shader Access.
        ProgramReflection::ShaderAccess shaderAccess;

        //  Whether or not this texture is Sampled. 
        bool isSampled = false;

        //  The Sampler is associated with this texture.
        std::string samplerVariable = "";
  
    };
    
    //  Structured Buffer Resource.
    struct StructuredBufferResource : public ShaderResource
    {
        //  
        ProgramReflection::Variable::Type sbType;

        //  Shader Access.
        ProgramReflection::ShaderAccess shaderAccess;

        //  Whether or not the Structured Buffer uses a Struct.
        bool usesExternalStructVariables;

        //  The Struct Variable.
        std::string structVariable;

        //  The Variables in the Struct.
        std::vector<StructVariable> externalStructVariables;

        //  The Type of the Structured Buffer.
        ProgramReflection::BufferReflection::StructuredType bufferType;

    };

    //  Raw Buffer Resource.
    struct RawBufferResource : public ShaderResource
    {
        //  Shader Access.
        ProgramReflection::ShaderAccess shaderAccess;
  
    };

    //  Sampler Resource.
    struct SamplerResource : public ShaderResource
    {


    };


    //  Constant Buffer Resource.
    struct ConstantBufferResource
    {
        //  Constant Buffer Name.
        std::string variable;

        //  The Register Index and the Space Index.
        uint32_t regIndex = 0;
        uint32_t regSpace = 0;
        

        //  Flag for Explicit Index.
        bool isExplicitIndex = false;
        
        //  Flag for Explicit Space.
        bool isExplicitSpace = false;
        
        //  Flag for Whether this Constant Buffer should be used.
        bool isUsedResource = false;

        //  Whether to write this as an External Struct.
        bool usesExternalStruct = false;

        //  The External struct.
        std::string externalStructVariable = "";

        //  The Variables of the cbuffer.
        std::vector<StructVariable> structVariables;

        //  Whether the Constant Buffer is an Array.
        bool isArray = false;

        //  Whether the Constant Buffer Array is bounded.
        bool isUnbounded = false;

        //  The Array size of the Constant Buffer.
        uint32_t arraySize = 0;

    };


    //  PS Meta Data.
    struct ShaderResourcesData
    {
        //  The Vector of Render Target Indices.
        std::vector<uint32_t> inTargets;
        std::vector<uint32_t> outTargets;
        
        //  Constant Buffer Resources.
        std::vector<ConstantBufferResource> mUCBs;
        
        //  Structured Buffer Resources.
        std::vector<StructuredBufferResource> sbs;

        //  Texture Resources.
        std::vector<TextureResource> textures;

        //  Raw Buffer Resources.
        std::vector<RawBufferResource> rbs;

        //  Sampler Resources.
        std::vector<SamplerResource> samplers;
    };

 
    //  Get the Vertex Shader Code.
    std::string getVertexShaderCode() const;
    
    //  Get the Hull Shader Code.
    std::string getHullShaderCode() const;

    //  Get the Domain Shader Code.
    std::string getDomainShaderCode() const;

    //  Get the Geometry Shader Code.
    std::string getGeometryShaderCode() const;

    //  Get the Pixel Shader Code.
    std::string getPixelShaderCode() const;

    //  Get the Compute Shader Code.
    std::string getComputeShaderCode() const;


    //  Return the Vertex Shader Data.
    ShaderResourcesData * getVSResourceData();

    //  Return the Hull Shader Data.
    ShaderResourcesData * getHSResourceData();

    //  Return the Domain Shader Data.
    ShaderResourcesData * getDSResourceData();

    //  Return the Geometry Shader Data.
    ShaderResourcesData * getGSResourceData();

    //  Return the Pixel Shader Data.
    ShaderResourcesData * getPSResourceData();

    //  Return the Compute Shader Data.
    ShaderResourcesData * getCSResourceData();


    //  Return the Vertex Shader Data.
    const ShaderResourcesData & viewVSResourceData() const;

    //  Return the Hull Shader Data.
    const ShaderResourcesData & viewHSResourceData() const;

    //  Return the Domain Shader Data.
    const ShaderResourcesData & viewDSResourceData() const;

    //  Return the Geometry Shader Data.
    const ShaderResourcesData & viewGSResourceData() const;

    //  Return the Pixel Shader Data.
    const ShaderResourcesData & viewPSResourceData() const;
    
    //  Return the Compute Shader Data.
    const ShaderResourcesData & viewCSResourceData() const;





private:

    //  Create a Vertex Shader.
    static std::string writeVSShaderCode(const ShaderResourcesData & vsMetaData);

    //  Create a Hull Shader.
    static std::string writeHSShaderCode(const ShaderResourcesData & hsMetaData);

    //  Create a Domain Shader.
    static std::string writeDSShaderCode(const ShaderResourcesData & dsMetaData);

    //  Create a Geometry Shader.
    static std::string writeGSShaderCode(const ShaderResourcesData & gsMetaData);

    //  Create a Pixel Shader.
    static std::string writePSShaderCode(const ShaderResourcesData & psMetaData);
    
    //  Create a Compute Shader.
    static std::string writeCSShaderCode(const ShaderResourcesData & csMetaData);

    //  Return the Vertex Shader Code for the Input Struct.
    static std::string writeShaderInputStruct(const ShaderResourcesData & psMetaData, const std::string & inputStructName, const std::string & semanticPrefix);

    //  Return the Output Struct for the Shader, given the name of the output struct and the semantic prefix for each output.
    static std::string writeShaderOutputStruct(const ShaderResourcesData & rsMetaData, const std::string & outputStructName, const std::string & semanticPrefix);

    //  Return the Vertex Shader Code for the Output Lines.
    static std::string writeShaderReturnCode(const ShaderResourcesData & rsMetaData, const std::string & inputStruct, const std::string & outputStruct);

    //  Return in the requested type.
    static std::string convertToType(const std::string & variable, const ProgramReflection::Variable::Type & startType, const ProgramReflection::Variable::Type & requestedType);


    //  Write the Declare and Use the Resources.
    static std::string writeDeclareResources(const ShaderResourcesData & shaderResourcesData);


    //  Write Declare Constant Buffer Resources.
    static std::string writeDeclareConstantBufferResources(const ShaderResourcesData & shaderResourcesData);
    static std::string writeUseConstantBufferResources(const ShaderResourcesData & shaderResourcesData);

    //  Write Declare Structured Buffer Resources.
    static std::string writeDeclareStructuredBufferResources(const std::vector<StructuredBufferResource> & givenStructuredBufferResources);
    static std::string writeDeclaredStructuredBufferResource(const StructuredBufferResource & givenStructuredBufferResource);

    //  Write Use Structured Buffer Resources.
    static std::string writeUseStructuredBufferResource(const StructuredBufferResource & givenStructuredBufferResource);

    //  Write Declare Sampler Resources.
    static std::string writeDeclareSamplerResources(const std::vector<SamplerResource> & givenSamplers);
    static std::string writeDeclareSamplerResource(const SamplerResource & givenSampler);

    //  Write Declare Texture Resources.
    static std::string writeDeclareTextureResources(const std::vector<TextureResource> & givenTextures);
    static std::string writeDeclareTextureResource(const TextureResource & givenTexture);
    
    //  Write Use Texture Resources.
    static std::string writeUseTextureResource(const TextureResource & givenTexture);

    //  Return the Texture Access Code for Textures that are not Sampled.
    static std::string writeNonSampleTextureAccessCode(const TextureResource & textureResource);

    //  Return the Texture Access Code for Textures that are Samples.
    static std::string writeSamplerTextureAccessCode(const TextureResource & textureResource);

    //  Write Declare Raw Buffer Resources.
    static std::string writeDeclareRawBufferResources(const std::vector<RawBufferResource> & givenRawBuffers);
    static std::string writeDeclareRawBufferResource(const RawBufferResource & givenRawBuffers);
    
    //  Write the Use Raw Buffer Resource.
    static std::string writeUseRawBufferResource(const RawBufferResource & givenRawBuffers);

    //  Write the Combined Resource Code.
    static std::string writeCombinedResourceCode(const ShaderResource & shaderResource, std::string declareType, std::string defaultRegisterType, ProgramReflection::ShaderAccess resourceAccess, bool isArray, bool isUnbounded, std::vector<uint32_t> arrayDimensions);


    //  Vertex Shader Data.
    ShaderResourcesData mVSData;

    //  Hull Shader Data.
    ShaderResourcesData mHSData;

    //  Domain Shader Data.
    ShaderResourcesData mDSData;

    //  Geometry Shader Data.
    ShaderResourcesData mGSData;

    //  Pixel Shader Data.
    ShaderResourcesData mPSData;

    //  Compute Shader Data;
    ShaderResourcesData mCSData;

};