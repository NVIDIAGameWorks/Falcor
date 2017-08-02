#include "HLSLProgramMaker.h"


//  Generate Shader Resources.
HLSLProgramMaker::HLSLProgramMaker(const HLSLProgramDesc & programDesc) : ShaderProgramMaker(programDesc)
{
    //  VS, HS, DS, GS, PS.
    mVSStageMaker = std::make_unique<VertexShaderStage>("VS_IN", "VS_OUT");
    mHSStageMaker = std::make_unique<HullShaderStage>("HS_IN", "HS_OUT");
    mDSStageMaker = std::make_unique<DomainShaderStage>("DS_IN", "DS_OUT");
    mGSStageMaker = std::make_unique<GeometryShaderStage>("GS_IN", "GS_OUT");
    mPSStageMaker = std::make_unique<PixelShaderStage>("PS_IN", "PS_OUT");


    //  Generate all the resources for each type of register.
    generateTRegisterResources(programDesc);
    generateBRegisterResources(programDesc);
    generateURegisterResources(programDesc);
    generateSRegisterResources(programDesc);

}

//  Get Constant Buffers.
const std::vector<ConstantBufferData> & HLSLProgramMaker::getConstantBuffers() const
{
    return mCBs;
}

//  Get the Structured Buffers.
const std::vector<HLSLResourcesDesc> & HLSLProgramMaker::getStructuredBuffers() const
{
    return mSBs;
}

//  Get the Raw Buffers.
const std::vector<HLSLResourcesDesc> & HLSLProgramMaker::getRawBuffers() const
{
    return mRawBuffers;
}

//  Get the Textures.
const std::vector<HLSLResourcesDesc> & HLSLProgramMaker::getTextures() const
{
    return mTextures;
}

//  Return the Samplers. 
const std::vector<HLSLResourcesDesc> & HLSLProgramMaker::getSamplers() const
{
    return mSamplers;
}

//  Generate the T Register Resources.
void HLSLProgramMaker::generateTRegisterResources(const HLSLProgramDesc & programDesc)
{
    //  Get the constants.
    uint32_t maxPerSpaceIndexValue = programDesc.tRegistersMaxPerSpace;
    uint32_t maxSpaceValue = programDesc.tRegistersMaxSpace + 1u;
    uint32_t totalAllocatableCount = maxPerSpaceIndexValue * (programDesc.tRegistersMaxSpace + 1u);

    //  Compute the Total Requested Resources.
    uint32_t totalRequestedResources = programDesc.sbsCount + programDesc.texturesCount + programDesc.rawBufferCount;

    //  Allocations.
    std::map<uint32_t, std::vector<AllocationRange>> allocationRanges;
    std::map<uint32_t, uint32_t> allocationCounts;
    std::map<uint32_t, uint32_t> maxAllocations;

    //  The Types Array.
    std::vector<uint32_t> allocableResourceTypes = { 0, 1, 2};

    //  The Current Space, Index and Allocation Count.
    uint32_t currentSpace = 0;
    uint32_t currentIndex = 0;
    uint32_t allocatedCount = 0;

    //  Maximum Allocations.
    maxAllocations[allocableResourceTypes[0]] = programDesc.sbsCount;
    maxAllocations[allocableResourceTypes[1]] = programDesc.texturesCount;
    maxAllocations[allocableResourceTypes[2]] = programDesc.rawBufferCount;

    //  Generate the Allocation Ranges.
    generateAllocationRanges(allocationRanges, maxPerSpaceIndexValue, maxSpaceValue, allocableResourceTypes, totalRequestedResources, maxAllocations, programDesc.allowResourceArrays, programDesc.allowExplicitSpaces);

    //  Generate the Structured Buffer Resources.
    generateStructuredBuffers(allocationRanges[0], false);
    
    //  Generate the Texture Resources.
    generateTextureResources(allocationRanges[1], false);
    
    //  Generate the Raw Buffers.
    generateRawBuffers(allocationRanges[2], false);

}

//  Generate the B Register Resources - just Constant Buffers.
void HLSLProgramMaker::generateBRegisterResources(const HLSLProgramDesc & programDesc)
{
    //  Get the constants.
    uint32_t maxPerSpaceIndex = programDesc.bRegistersMaxPerSpace;
    uint32_t maxSpace = programDesc.bRegisterMaxSpace + 1u;
    uint32_t totalAllocatableCount = maxPerSpaceIndex * (programDesc.bRegisterMaxSpace + 1u);

    //  The Current Space, Index and Allocation Count.
    uint32_t currentSpace = 0;
    uint32_t currentIndex = 0;
    uint32_t allocatedCount = 0;

    //  Keep allocating, until we run out of space or allocations.
    while (currentSpace < maxSpace && currentIndex < maxPerSpaceIndex && allocatedCount < programDesc.cbsCount)
    {
        //  Create the Constant Buffer.
        ConstantBufferData cbData("cb" + std::to_string(allocatedCount));
        cbData.baseResourceType = HLSLBaseResourceType::BufferType;
        cbData.bufferType = HLSLBufferType::ConstantBuffer;

        //  Check whether we can create a constant buffer array.
        uint32_t maxArraySize = min(programDesc.cbsCount - allocatedCount, maxPerSpaceIndex - currentIndex);
        bool isArray = ((rand() % 2) == 0) && programDesc.allowArrayCBs && (maxArraySize > 1u);

        //  Check if we are creating an array.
        if (!isArray)
        {
            //  Check if we allow explicit spaces.
            if (programDesc.allowExplicitSpaces)
            {
                //  Set the Register Space and Index.
                bool isAttachmentPointExplicit = (currentSpace > 0) ? true : ((rand() % 2) == 0);
                bool isAttachmentSubpointExplicit = isAttachmentPointExplicit || ((currentSpace > 0) ? true : ((rand() % 3) == 0));
                cbData.setAttachmentDesc(currentIndex, currentSpace, isAttachmentSubpointExplicit, isAttachmentPointExplicit);
            }
            else
            {
                //  
                bool isAttachmentSubpointExplicit = ((currentSpace > 0) ? true : ((rand() % 3) == 0));
                cbData.setAttachmentDesc(currentIndex, currentSpace, isAttachmentSubpointExplicit, false);
            }

            //  Increment the space index.
            currentIndex = currentIndex + 1;
            allocatedCount = allocatedCount + 1;

            //  Add the Constant Buffer.
            mCBs.push_back(cbData);
        }
        else
        {
            //  Get the array size.
            uint32_t arraySize = maxArraySize > 2 ? ((rand() % (maxArraySize - 2)) + 2) : maxArraySize;

        }


        //  Reset to the next space.
        if (currentIndex == maxPerSpaceIndex)
        {
            currentIndex = 0;
            currentSpace = currentSpace + 1;
        }

    }

}

//  Generate the S Register Resources - just the Samplers.
void HLSLProgramMaker::generateSRegisterResources(const HLSLProgramDesc & programDesc)
{
    //  Get the constants.
    uint32_t maxPerSpaceIndexValue = programDesc.sRegistersMaxPerSpace;
    uint32_t maxSpaceValue = programDesc.sRegistersMaxSpace + 1u;
    uint32_t totalAllocatableCount = maxPerSpaceIndexValue * (programDesc.sRegistersMaxSpace + 1u);

    //  Compute the Total Requested Resources.
    uint32_t totalRequestedResources = programDesc.samplerCount;

    //  Allocations.
    std::map<uint32_t, std::vector<AllocationRange>> allocationRanges;
    std::map<uint32_t, uint32_t> maxAllocations;

    //  The Types Array.
    std::vector<uint32_t> allocableResourceTypes = {0};

    //  Max Allocations.
    maxAllocations[allocableResourceTypes[0]] = programDesc.samplerCount;

    //  Generate the Allocation Ranges.
    generateAllocationRanges(allocationRanges, maxPerSpaceIndexValue, maxSpaceValue, allocableResourceTypes, totalRequestedResources, maxAllocations, programDesc.allowResourceArrays, programDesc.allowExplicitSpaces);


    //  
    for (uint32_t i = 0; i < allocationRanges[0].size(); i++)
    {
        //  Create the Resource.
        HLSLResourcesDesc srDesc("samplerRArray" + std::to_string(i));
        srDesc.attachmentDesc.registerType = "s";
        srDesc.baseResourceType = HLSLBaseResourceType::SamplerType;

        applyAllocationRange(srDesc, allocationRanges[0][i]);
        mSamplers.push_back(srDesc);
    }

}

//  Generate the U Register Resources.
void HLSLProgramMaker::generateURegisterResources(const HLSLProgramDesc & programDesc)
{

    //  Get the constants.
    uint32_t maxPerSpaceIndexValue = programDesc.uRegistersMaxPerSpace;
    uint32_t maxSpaceValue = programDesc.uRegistersMaxSpace + 1u;
    uint32_t totalAllocatableCount = maxPerSpaceIndexValue * (maxSpaceValue + 1u);

    //  Compute the Total Requested Resources.
    uint32_t totalRequestedResources = programDesc.rwSBsCount + programDesc.rwTexturesCount + programDesc.rwRawBufferCount;

    //  Allocations.
    std::map<uint32_t, std::vector<AllocationRange>> allocationRanges;
    std::map<uint32_t, uint32_t> maxAllocations;

    //  The Types Array.
    std::vector<uint32_t> allocableResourceTypes = { 0, 1, 2 };

    //  The Current Space, Index and Allocation Count.
    uint32_t currentSpace = 0;
    uint32_t currentIndex = 0;
    uint32_t allocatedCount = 0;

    //  Maximum Allocations.
    maxAllocations[allocableResourceTypes[0]] = programDesc.sbsCount;
    maxAllocations[allocableResourceTypes[1]] = programDesc.texturesCount;
    maxAllocations[allocableResourceTypes[2]] = programDesc.rawBufferCount;
    
    //  Generate the Allocation Ranges.
    generateAllocationRanges(allocationRanges, maxPerSpaceIndexValue, maxSpaceValue, allocableResourceTypes, totalRequestedResources, maxAllocations, programDesc.allowResourceArrays, programDesc.allowExplicitSpaces);

    //  Generate the Structured Buffer Resources.
    generateStructuredBuffers(allocationRanges[0], true);

    //  Generate the Texture Resources.
    generateTextureResources(allocationRanges[1], true);

    //  Generate the Raw Buffers.
    generateRawBuffers(allocationRanges[2], true);
}

//  Generate the Texture Resources.
void HLSLProgramMaker::generateTextureResources(const std::vector<AllocationRange> & allocationRanges, bool isReadWrite)
{
    //  
    for (uint32_t i = 0; i < allocationRanges.size(); i++)
    {
        //  Texture R.
        std::string name = "textureR" + std::to_string(i);
        name = isReadWrite ? "RW" + name : name;

        //  Create the Appropriate Resource.
        HLSLResourcesDesc rDesc(name);
        rDesc.baseResourceType = HLSLBaseResourceType::TextureType;
        rDesc.accessType = isReadWrite ? CommonShaderDescs::AccessType::ReadWrite : CommonShaderDescs::AccessType::Read;

        //  Apply the details of the allocation range.
        applyAllocationRange(rDesc, allocationRanges[i]);
        mTextures.push_back(rDesc);
    }

}

//  Generate the Structured Buffers.
void HLSLProgramMaker::generateStructuredBuffers(const std::vector<AllocationRange> & allocationRanges, bool isReadWrite)
{
    //  
    for (uint32_t i = 0; i < allocationRanges.size(); i++)
    {
        //  
        std::string name = "sbR" + std::to_string(i);
        name = isReadWrite ? "RW" + name : name;

        //  Create the Appropriate Resource.
        HLSLResourcesDesc rDesc(name);
        rDesc.baseResourceType = HLSLBaseResourceType::BufferType;
        rDesc.bufferType = HLSLBufferType::StructuredBuffer;
        rDesc.accessType = isReadWrite ? CommonShaderDescs::AccessType::ReadWrite : CommonShaderDescs::AccessType::Read;
        rDesc.structDesc.structVariableType = "sbRStruct" + std::to_string(i);

        //  Add the Strut Variables.
        for (uint32_t i = 0; i < 2; i++)
        {
            HLSLResourcesDesc svDesc("sbR" + std::to_string(i));

            svDesc.baseResourceType = HLSLBaseResourceType::BasicType;

            rDesc.structDesc.structVariables.push_back(svDesc);
        }


        //  Apply the details of the allocation range.
        applyAllocationRange(rDesc, allocationRanges[i]);

        //  Add the Structured Buffers.
        mSBs.push_back(rDesc);
    }
}

//  Generate the Raw Buffers.
void HLSLProgramMaker::generateRawBuffers(const std::vector<AllocationRange> & allocationRanges, bool isReadWrite)
{
    //  
    for (uint32_t i = 0; i < allocationRanges.size(); i++)
    {
        std::string name = "rawbufferR";
        name = isReadWrite ? "RW" + name : name;

        //  Create the Appropriate Resource.
        HLSLResourcesDesc rDesc(name);
        rDesc.bufferType = HLSLBufferType::RawBuffer;
        rDesc.baseResourceType =HLSLBaseResourceType::BufferType;
        rDesc.accessType = isReadWrite ? CommonShaderDescs::AccessType::ReadWrite : CommonShaderDescs::AccessType::Read;


        //  Apply the details of the allocation range.
        applyAllocationRange(rDesc, allocationRanges[i]);
        
        //  
        mSBs.push_back(rDesc);
    }
}

//  Apply Allocation Range.
void HLSLProgramMaker::applyAllocationRange(HLSLResourcesDesc & rDesc, const AllocationRange & allocationRange)
{
    //  
    rDesc.attachmentDesc.attachmentPoint = allocationRange.attachmentPoint;
    rDesc.attachmentDesc.attachmentSubpoint = allocationRange.attachmentSubpoint;
    rDesc.attachmentDesc.isAttachmentPointExplicit = allocationRange.isAttachmentPointExplicit;
    rDesc.attachmentDesc.isAttachmentSubpointExplicit = allocationRange.isAttachmentSubpointExplicit;

    rDesc.arrayDesc.isArray = allocationRange.isArray;
    rDesc.arrayDesc.dimensions = allocationRange.dimensions;
}

//  Generate the Allocation Ranges.
void HLSLProgramMaker::generateAllocationRanges(std::map<uint32_t, std::vector<AllocationRange>> & allocationRanges, uint32_t maxPerSpaceIndexValue, uint32_t maxSpaceValue, const std::vector<uint32_t> allocableResourceTypes, uint32_t totalRequestedResources, const std::map<uint32_t, uint32_t> maxAllocations, bool allowArrays, bool allowExplicitSpaces)
{

    std::map<uint32_t, uint32_t> allocationCounts;

    //  
    for (uint32_t i = 0; i < allocableResourceTypes.size(); i++)
    {
        allocationCounts[allocableResourceTypes[i]] = 0;
    }

    std::vector<uint32_t> cleanedAllocatableResourceTypes;

    //  Clear out the unallocable resources.
    for (uint32_t i = 0; i < allocableResourceTypes.size(); i++)
    {
        allocationCounts[allocableResourceTypes[i]] = 0;
        
        auto maxAllocItr = maxAllocations.find(allocableResourceTypes[i]);
        if (maxAllocItr != maxAllocations.end())
        {
            if (maxAllocItr->second != 0)
            {
                cleanedAllocatableResourceTypes.push_back(allocableResourceTypes[i]);
            }
        }
    }


    //  The Current Space, Index and Allocation Count.
    uint32_t currentSpace = 0;
    uint32_t currentIndex = 0;
    uint32_t allocatedCount = 0;

    //  Keep allocating, until we run out of space or allocations.
    while (currentSpace < maxSpaceValue && currentIndex < maxPerSpaceIndexValue && allocatedCount < totalRequestedResources)
    {
        uint32_t selection = 0;

        if (cleanedAllocatableResourceTypes.size() == 0)
        {
            break;
        }

        if (cleanedAllocatableResourceTypes.size() == 1)
        {
            selection = 0;
        }
        else
        {
            selection = rand() % cleanedAllocatableResourceTypes.size();
        }

        //  Get the Max Allocation.
        auto maxAllocItr = maxAllocations.find(cleanedAllocatableResourceTypes[selection]);
        uint32_t maxSelectionAllocation = 0;
        if (maxAllocItr != maxAllocations.end())
        {
            maxSelectionAllocation = maxAllocItr->second;
        }
        else
        {
            break;
        }

        //  Check whether we can create a resource array.
        uint32_t maxArraySize = min(totalRequestedResources - allocatedCount, maxPerSpaceIndexValue - currentIndex);
        maxArraySize = min(maxArraySize, maxSelectionAllocation - allocationCounts[cleanedAllocatableResourceTypes[selection]]);

        bool isArray = ((rand() % 2) == 0) && allowArrays && (maxArraySize > 1);


        //  Allocation Range.
        AllocationRange currentAllocation;

        //  Check whether we allow explicit spaces, and indexes.
        {
            if (allowExplicitSpaces)
            {
                //  Set whether the Point and Subpoint are explicit.
                currentAllocation.isAttachmentPointExplicit = (currentSpace > 0) ? true : ((rand() % 2) == 0);
                currentAllocation.isAttachmentSubpointExplicit = currentAllocation.isAttachmentPointExplicit || ((currentSpace > 0) ? true : ((rand() % 3) == 0));
            }
            else
            {
                //  Set whether the Subpoint is explicit.
                currentAllocation.isAttachmentSubpointExplicit = currentAllocation.isAttachmentPointExplicit || ((currentSpace > 0) ? true : ((rand() % 3) == 0));
                currentAllocation.isAttachmentPointExplicit = false;
            }

            //  
            currentAllocation.attachmentSubpoint = currentIndex;
            currentAllocation.attachmentPoint = currentSpace;
        }

        //  
        //  Check whether we are creating an Array.
        if (!isArray)
        {

            //  Increment the space index.
            currentIndex = currentIndex + 1;
            allocatedCount = allocatedCount + 1;

            // Add the current allocation range.
            allocationRanges[cleanedAllocatableResourceTypes[selection]].push_back(currentAllocation);

            //  Decrement the remaining allocations.
            allocationCounts[cleanedAllocatableResourceTypes[selection]] = allocationCounts[cleanedAllocatableResourceTypes[selection]] + 1;
        }
        else
        {
            //  Get the array size.
            uint32_t arraySize = maxArraySize > 2 ? ((rand() % (maxArraySize - 2)) + 2) : maxArraySize;

            //
            currentAllocation.isArray = true;
            currentAllocation.dimensions = { arraySize };

            //  
            currentIndex = currentIndex + arraySize;
            allocatedCount = allocatedCount + arraySize;

            // Add the current allocation range.
            allocationRanges[cleanedAllocatableResourceTypes[selection]].push_back(currentAllocation);

            //  Decrement the remaining allocations.
            allocationCounts[cleanedAllocatableResourceTypes[selection]] = allocationCounts[cleanedAllocatableResourceTypes[selection]] + arraySize;
        }

        //    Check if the current type has any allocations left.
        if (maxSelectionAllocation <= allocationCounts[cleanedAllocatableResourceTypes[selection]])
        {
            cleanedAllocatableResourceTypes[selection] = cleanedAllocatableResourceTypes[cleanedAllocatableResourceTypes.size() - 1];
            cleanedAllocatableResourceTypes.pop_back();
        }


        //  Reset to the next space.
        if (currentIndex == maxPerSpaceIndexValue)
        {
            currentIndex = 0;
            currentSpace = currentSpace + 1;
        }
    }
}
