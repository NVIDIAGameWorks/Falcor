#pragma once
#include "CommonShaderDescs.h"
#include "ShaderProgramMaker.h"
#include "HLSLDescs.h"

using namespace Falcor;
using namespace HLSLDescs;


/**
*   HLSL Program Maker:
*   Can be used to Procedurally Generate HLSL Shaders based on the provided specifications.
*   
*   
*/

//  HLSL Program Maker.
class HLSLProgramMaker : public ShaderProgramMaker
{

public:

    //  Allocation Range Struct.
    struct AllocationRange
    {
        bool isArray = false;
        std::vector<uint32_t> dimensions;
        bool isAttachmentPointExplicit;
        bool isAttachmentSubpointExplicit;
        uint32_t attachmentPoint;
        uint32_t attachmentSubpoint;
    };



    struct HLSLProgramDesc : ProgramDesc
    {
        //  The Constant Buffers Count.
        uint32_t cbsCount = 0;

        //  Constant Buffers.
        bool allowStructCBs = false;
        bool allowArrayCBs = false;

        //  Resource Arrays.
        bool allowResourceArrays = false;
        bool allowExplicitSpaces = false;

        //  The Texture Buffers Count.
        uint32_t tbsCount = 0;

        //  The Structured Buffers Count.
        uint32_t sbsCount = 0;
        uint32_t rwSBsCount = 0;

        //  The Raw Buffers Count.
        uint32_t rawBufferCount = 0;
        uint32_t rwRawBufferCount = 0;

        //  The Textures Count.
        uint32_t texturesCount = 0;
        uint32_t rwTexturesCount = 0;

        //  The Samplers Count.
        uint32_t samplerCount = 0;

        //  
        //  b Registers.
        uint32_t bRegistersMaxPerSpace = 15;
        uint32_t bRegisterMaxSpace = 0;

        //  s Registers.
        uint32_t sRegistersMaxPerSpace = 15;
        uint32_t sRegistersMaxSpace = 0;

        //  t Registers.
        uint32_t tRegistersMaxPerSpace = 15;
        uint32_t tRegistersMaxSpace = 0;

        //  u Registers.
        uint32_t uRegistersMaxPerSpace = 15;
        uint32_t uRegistersMaxSpace = 0;

    };


    //  Default HLSL Program Maker Constructor.
    HLSLProgramMaker(const HLSLProgramDesc & programDesc);

    //  Default HLSL Program Maker Destructor.
    ~HLSLProgramMaker() = default;

    //  Get the Constant Buffers.
    virtual const std::vector<ConstantBufferData> & getConstantBuffers() const;

    //  Get the Structured Buffers.
    virtual const std::vector<HLSLResourcesDesc> & getStructuredBuffers() const;

    //  Get the Raw Buffers.
    virtual const std::vector<HLSLResourcesDesc> & getRawBuffers() const;

    //  Get the Textures.
    virtual const std::vector<HLSLResourcesDesc> & getTextures() const;

    //  Get the Samplers.
    virtual const std::vector<HLSLResourcesDesc> & getSamplers() const;


    //  Generate the T Register Resources.
    void generateTRegisterResources(const HLSLProgramDesc & programDesc);

    //  Generate the B Register Resources - Constant Buffers Only.
    void generateBRegisterResources(const HLSLProgramDesc & programDesc);

    //  Generate the S Register Resources - Samplers Only.
    void generateSRegisterResources(const HLSLProgramDesc & programDesc);

    //  Generate the U Register Resources.
    void generateURegisterResources(const HLSLProgramDesc & programDesc);

    //  Generate the Texture Resources.
    void generateTextureResources(const std::vector<AllocationRange> & allocationRange, bool isReadWrite);

    //  Generate the Structured Buffers.
    void generateStructuredBuffers(const std::vector<AllocationRange> & allocationRange, bool isReadWrite);

    //  Generate the Raw Buffers.
    void generateRawBuffers(const std::vector<AllocationRange> & allocationRange, bool isReadWrite);

    //  Apply the Allocation Range.
    virtual void applyAllocationRange(HLSLResourcesDesc & rDesc, const AllocationRange & allocationRange);
    
    //  Generate the Allocation Range.
    virtual void generateAllocationRanges(std::map<uint32_t, std::vector<AllocationRange>> & allocationRanges, uint32_t maxPerSpaceIndexValue, uint32_t maxSpaceValue, const std::vector<uint32_t> allocableResourceTypes, uint32_t totalRequestedResources, const std::map<uint32_t, uint32_t> maxAllocations, bool allowArrays, bool allowExplicitSpaces);



private:

    //  Constant Buffers.
    std::vector<ConstantBufferData> mCBs;

    //  Structured Buffers.
    std::vector<HLSLResourcesDesc> mSBs;

    //  Samplers.
    std::vector<HLSLResourcesDesc> mSamplers;

    //  Textures.
    std::vector<HLSLResourcesDesc> mTextures;

    //  Raw Buffers.
    std::vector<HLSLResourcesDesc> mRawBuffers;
};