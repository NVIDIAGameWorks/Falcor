#pragma once

#include "Falcor.h"
#include "TestShaderWriter.h"
#include <string.h>
#include <map>
#include <vector>
#include <set>
#include <random>
#include <algorithm>    // std::random_shuffle
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand

#include "Utils/OS.h"
#include "Externals/RapidJson/include/rapidjson/document.h"
#include "Externals/RapidJson/include/rapidjson/stringbuffer.h"
#include "Externals/RapidJson/include/rapidjson/prettywriter.h"
#include "Externals/RapidJson/include/rapidjson/error/en.h"
#include <sstream>
#include <fstream>


class TestShaderMaker
{

public:

    struct TestShaderLayout
    {
        //  The Random Seed.
        uint32_t randomseed;

        //  The Version for the Shader Test Layout.
        std::string version;

        //  The Shader Stages.
        std::vector<std::string> stages;

        //  The Declarations.
        std::vector<std::string> declarations;

        //  The Constant Buffers Count.
        uint32_t constantbuffers;

        //  The Texture Buffers Count.
        uint32_t texturebuffers;

        //  The Structured Buffers Count.
        uint32_t structuredbuffers;
        uint32_t rwStructuredBuffers;

        //  The Raw Buffers Count.
        uint32_t rawbuffers;
        uint32_t rwRawBuffers;

        //  The Textures Count.
        uint32_t textures;
        uint32_t rwTextures;

        //  The Samplers Count.
        uint32_t samplers;

        //  The Mode.
        std::string mode;

        //  
        std::string invalidResourceType;


        //  Whether or not we have a Vertex Shader.
        bool hasVertexShader;

        //  Whether or not we have a Hull Shader.
        bool hasHullShader;

        //  Whether or not we have a DomainShader.
        bool hasDomainShader;

        //  Whether or not we have a Geometry Shader.
        bool hasGeometryShader;

        //  Whether or not we have a Pixel Shader.
        bool hasPixelShader;

        //  Compute Shader.
        bool hasComputeShader;

        //  Some general constants.

        //  Maximum number of Struct Variables.
        uint32_t maxStructVariablesCount = 3;

        
        //  Maximum UAV Targets.
        uint32_t maxUAVTargets = 8;



        //  Allow Unbounded Constant Buffers.
        bool allowUnboundedConstantBuffers = false;
        
        //  Allow Constant Buffers Array.
        bool allowConstantBuffersArray = false;
        
        //  Allow Struct Constant Buffers.
        bool allowStructConstantBuffers = false;

        //  Allow Unbounded Resource Arrays.
        bool allowUnboundedResourceArrays = false;

        //  Allow Explicit Spaces.
        bool allowExplicitSpaces = false;
        
        
        //  Vertex Shader Outputs.
        std::vector<uint32_t> vsOutputs;
        
        //  Hull Shader Inputs.
        std::vector<uint32_t> hsInputs;
        
        //  Hull Shader Outputs.
        std::vector<uint32_t> hsOutputs;
        
        //  Domain Shader Inputs.
        std::vector<uint32_t> dsInputs;
        
        //  Domain Shader Outputs.
        std::vector<uint32_t> dsOutputs;
        
        //  Geometry Shader Inputs.
        std::vector<uint32_t> gsInputs;
        
        //  Geometry Shader Outputs.
        std::vector<uint32_t> gsOutputs;

        //  Pixel Shader Inputs.
        std::vector<uint32_t> psInputs;
        
        //  Pixel Shader Outputs.
        std::vector<uint32_t> psOutputs;


        //  The Maximum Number of Threads in the Compute Shader.
        uint32_t maxComputeThreads = 1024;

    };


    //  
    struct RegisterCount
    {
        bool isMaxed = false;
        uint32_t nextRegisterIndex = 0;
    };


    //  Load the Configuration File.
    bool loadConfigFile(const std::string & configFile, std::string errorMessage);

    //  Set the Version Restricted Constants.
    void generateVersionRestrictedConstants();

    //  Return the Shader Test Layout in a const format.
    const TestShaderLayout & viewShaderTestLayout() const;

    //  Return the Shader Test Layout.
    TestShaderLayout & getShaderTestLayout();

    //  Generate the Shader Program.
    void generateShaderProgram();

    //  Get Vertex Shader Code.
    std::string getVertexShaderCode() const;

    //  Get Hull Shader Code.
    std::string getHullShaderCode() const;

    //  Get Domain Shader Code.
    std::string getDomainShaderCode() const;

    //  Get Geometry Shader Code.
    std::string getGeometryShaderCode() const;

    //  Get Pixel Shader Code.
    std::string getPixelShaderCode() const;

    //  Get Compute Shader Code.
    std::string getComputeShaderCode() const;

    
    //  Verify that the Resources are in keeping with the Program Creation.
    bool verifyProgramResources(ProgramReflection::SharedConstPtr programReflection, std::string & errorMessage);


private:

    //  Generate the Constant Buffer Resources that cause an error.
    void generateErrorConstantBufferResources(std::vector<TestShaderWriter::ConstantBufferResource> & mUCBs);

    //  Generate the Structured Buffer Resources that cause an error.
    void generateErrorStructuredBufferResources(std::vector<TestShaderWriter::StructuredBufferResource> & sbs);

    //  Generate the Raw Buffer Resources that cause an error.
    void generateErrorRawBufferResources(std::vector<TestShaderWriter::RawBufferResource> & rbs);

    //  Generate the Texture Resources that cause an error.
    void generateErrorTextureResources(std::vector<TestShaderWriter::TextureResource> & texs);

    //  Generate the Sampler Resources that cause an error.
    void generateErrorSamplersResources(std::vector<TestShaderWriter::SamplerResource> & samps);

    //  Generate the UAV Resources that cause an error.
    void generateErrorUAVResources(std::vector<TestShaderWriter::StructuredBufferResource> & rwSBs, std::vector<TestShaderWriter::TextureResource> & rwTexs);

    //  Generate the Constant Buffer Resources.
    void generateConstantBufferResources(std::vector<TestShaderWriter::ConstantBufferResource> & mUCBs);

    //  Generate the Texture Buffer Resources.

    //  Generate the Structured Buffer Resources.
    void generateStructuredBufferResources(std::vector<TestShaderWriter::StructuredBufferResource> & sbs, std::vector<TestShaderWriter::StructuredBufferResource> & rwSBs);

    //  Generate the Raw Buffer Resources.
    void generateRawBufferResources(std::vector<TestShaderWriter::RawBufferResource> & rbs, std::vector<TestShaderWriter::RawBufferResource> & rwRBs);

    //  Generate the Sampler Resources.
    void generateSamplersResources(std::vector<TestShaderWriter::SamplerResource> & samps);

    //  Generate the Texture Resources.
    void generateTextureResources(std::vector<TestShaderWriter::TextureResource> & texs, std::vector<TestShaderWriter::TextureResource> & rwTexs);

    //  Return a pointer to the Resource Meta Data of the specified Stage. 
    TestShaderWriter::ShaderResourcesData * getShaderStageResourceData(const std::string & stage);

    //  
    void getRegisterIndexAndSpace(bool isUsingExplicitSpace, bool isExplicitIndex, bool isExplicitSpace, bool isArray, bool isUnboundedArray, std::vector<uint32_t> arrayDiemsions, std::map<uint32_t, RegisterCount> & spaceRegisterIndexes, std::vector<uint32_t> & spaces, uint32_t & maxSpace, uint32_t & newIndex, uint32_t & newSpace);

    //  Return a Randomized Selection of Stages.
    std::vector<uint32_t> generateRandomizedStages();

    //  Distribute the Constant Buffer Resources.
    void distributeConstantBuffers(std::vector<TestShaderWriter::ConstantBufferResource> & mUCBs);

    //  Distribute the Structured Buffers.
    void distributeStructuredBuffers(std::vector<TestShaderWriter::StructuredBufferResource> & sbs, std::vector<TestShaderWriter::StructuredBufferResource> & rwSBs);

    //  Distribute the Samplers.
    void distributeSamplers(std::vector<TestShaderWriter::SamplerResource> & samplerResources);

    //  Distribute the Textures.
    void distributeTextures(std::vector<TestShaderWriter::TextureResource> & texs, std::vector<TestShaderWriter::TextureResource> & rwTexs);

    //  Distribute the Raw Buffers.
    void distributeRawBuffers(std::vector<TestShaderWriter::RawBufferResource> & rBs, std::vector<TestShaderWriter::RawBufferResource> & rwRBs);

    //  Verify All Resources.
    bool verifyAllResources(const TestShaderWriter::ShaderResourcesData & rsData, ProgramReflection::SharedConstPtr graphicsReflection, std::string & errorMessage);

    //  Verify Common Resource.
    bool verifyCommonResourceMatch(const TestShaderWriter::ShaderResource & srData, ProgramReflection::Resource::ResourceType resourceType, ProgramReflection::SharedConstPtr  graphicsReflection, std::string & errorMessage);


    //  JSON Document.
    rapidjson::Document mJDoc;

    //  Shader Maker.
    TestShaderWriter mShaderMaker;

    //  Shader Test Layout.
    TestShaderLayout mShaderTestLayout;

    //  b Registers.
    std::map<uint32_t, RegisterCount> mBSpacesIndexes;
    std::vector<uint32_t> mBSpaces;
    uint32_t mMaxBSpace;

    //  s Register.
    std::map<uint32_t, RegisterCount> mSSpacesIndexes;
    std::vector<uint32_t> mSSpaces;
    uint32_t mMaxSSpace;

    //  t Registers.
    std::map<uint32_t, RegisterCount> mTSpacesIndexes;
    std::vector<uint32_t> mTSpaces;
    uint32_t mMaxTSpace;

    //  u Registers.
    std::map<uint32_t, RegisterCount> mUSpacesIndexes;
    std::vector<uint32_t> mUSpaces;
    uint32_t mMaxUSpace;

};

