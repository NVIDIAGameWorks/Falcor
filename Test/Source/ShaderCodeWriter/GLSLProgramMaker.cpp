#include "GLSLProgramMaker.h"

//  
GLSLProgramMaker::GLSLProgramMaker(const GLSLProgramDesc & programDesc) : ShaderProgramMaker(programDesc)
{
    //  VS, HS, DS, GS, PS.
    mVSStageMaker = std::make_unique<VertexShaderStage>("VS_IN", "VS_OUT");

    mHSStageMaker = std::make_unique<HullShaderStage>("HS_IN", "HS_OUT");

    mDSStageMaker = std::make_unique<DomainShaderStage>("DS_IN", "DS_OUT");

    mGSStageMaker = std::make_unique<GeometryShaderStage>("GS_IN", "GS_OUT");

    mPSStageMaker = std::make_unique<PixelShaderStage>("PS_IN", "PS_OUT");
    
    //  
    generateAllResources(programDesc);
}

//  
void GLSLProgramMaker::generateAllResources(const GLSLProgramDesc & programDesc)
{
    //  Get the constants.
    uint32_t maxPerSetBindings = programDesc.maxBindingsPerSet;
    uint32_t maxSet = programDesc.maxSets + 1;
    uint32_t totalAllocatableCount = programDesc.maxBindingsPerSet * (programDesc.maxSets + 1u);

    //  Initialize the Allocation Constants.
    uint32_t currentSet = 0;
    uint32_t currentBindings = 0;
    uint32_t allocatedCount = 0;
    
    //  Compute the Total Requested Resources.
    uint32_t totalRequestedResources = programDesc.ubsCount + programDesc.textureCount + programDesc.samplerCount;
    
    //  Allocations.
    std::map<uint32_t, std::vector<AllocationRange>> allocationRanges;
    std::map<uint32_t, uint32_t> allocationCounts;
    std::map<uint32_t, uint32_t> maxAllocations;

    //  The Types Array.
    std::vector<uint32_t> allocatableResourceTypes = { 0, 1, 2, 3, 4 };

    //  Initialize the Maximum Number of Allocations.
    maxAllocations[allocatableResourceTypes[0]] = programDesc.ubsCount;
    maxAllocations[allocatableResourceTypes[1]] = programDesc.ssbosCount;
    maxAllocations[allocatableResourceTypes[2]] = programDesc.textureCount;
    maxAllocations[allocatableResourceTypes[3]] = programDesc.rwTextureCount;
    maxAllocations[allocatableResourceTypes[4]] = programDesc.samplerCount;

    std::vector<uint32_t> cleanedAllocatableResourceTypes;
    //  Clear out the unallocated resources.
    for (uint32_t i = 0; i < allocatableResourceTypes.size(); i++)
    {
        allocationCounts[allocatableResourceTypes[i]] = 0;
        if (maxAllocations[allocatableResourceTypes[i]] != 0)
        {
            cleanedAllocatableResourceTypes.push_back(allocatableResourceTypes[i]);
        }
    }
    allocatableResourceTypes = cleanedAllocatableResourceTypes;


    //  Keep allocating, until we run out of space or allocations.
    while (currentSet < maxSet && currentBindings < maxPerSetBindings && allocatedCount < totalRequestedResources)
    {
        uint32_t selection = 0;

        //  
        if (allocatableResourceTypes.size() == 0)
        {
            break;
        }

        if (allocatableResourceTypes.size() == 1)
        {
            selection = 0;
        }
        else
        {
            selection = rand() % allocatableResourceTypes.size();
        }

        //  Check whether we can create a resource array.
        uint32_t maxArraySize = min(totalRequestedResources - allocatedCount, maxPerSetBindings - currentBindings);
        bool isArray = ((rand() % 2) == 0) && (maxArraySize > 1);

        //  Check if we are creating an array.
        if (!isArray)
        {
            //  
            AllocationRange currentAllocation;

            //  Set whether the Point and Subpoint are explicit.
            currentAllocation.isAttachmentPointExplicit = (currentSet > 0) ? true : ((rand() % 2) == 0);
            currentAllocation.isAttachmentSubpointExplicit = currentAllocation.isAttachmentPointExplicit || ((currentSet > 0) ? true : ((rand() % 3) == 0));

            //  
            currentAllocation.attachmentSubpoint = currentBindings;
            currentAllocation.attachmentPoint = currentSet;

            //  Increment the space index.
            currentBindings = currentBindings + 1;
            allocatedCount = allocatedCount + 1;

            // Add the current allocation range.
            allocationRanges[allocatableResourceTypes[selection]].push_back(currentAllocation);

            //  Decrement the remaining allocations.
            allocationCounts[allocatableResourceTypes[selection]] = allocationCounts[allocatableResourceTypes[selection]] + 1;
        }
        else
        {
            //  Get the array size.
            uint32_t arraySize = maxArraySize >= 2 ? maxArraySize : ((rand() % (maxArraySize - 2)) + 2);

            //  
            AllocationRange currentAllocation;

            //  Set whether the Point and Subpoint are explicit.
            currentAllocation.isAttachmentPointExplicit = (currentSet > 0) ? true : ((rand() % 2) == 0);
            currentAllocation.isAttachmentSubpointExplicit = currentAllocation.isAttachmentPointExplicit || ((currentSet > 0) ? true : ((rand() % 3) == 0));
            
            //
            currentAllocation.isArray = true;
            currentAllocation.dimensions = { arraySize };

            //  
            currentAllocation.attachmentSubpoint = currentBindings;
            currentAllocation.attachmentPoint = currentSet;

            //  
            currentBindings = currentBindings + arraySize;
            allocatedCount = allocatedCount + arraySize;

            // Add the current allocation range.
            allocationRanges[allocatableResourceTypes[selection]].push_back(currentAllocation);

            //  Decrement the remaining allocations.
            allocationCounts[allocatableResourceTypes[selection]] = allocationCounts[allocatableResourceTypes[selection]] + arraySize;
        }

        //    
        if (maxAllocations[allocatableResourceTypes[selection]] - allocationCounts[allocatableResourceTypes[selection]] <= 0)
        {
            allocatableResourceTypes[selection] = allocatableResourceTypes[allocatableResourceTypes.size() - 1];
            allocatableResourceTypes.pop_back();
        }


        //  Reset to the next space.
        if (currentBindings == maxPerSetBindings)
        {
            currentBindings = 0;
            currentSet = currentSet + 1;
        }

    }

    //  Generate the Uniform Buffers.
    generateUniformBuffers(allocationRanges[0]);

    //  Generate the Shader Storage Buffer Objects.
    generateSSBOs(allocationRanges[1]);
    
    //  Generate the Textures.
    generateTextures(allocationRanges[2]);

    //  Generate the Read Write Textures.
    generateImages(allocationRanges[3]);
    
    //  Generate the Sampler Textures.
    generateSamplerTextures(allocationRanges[4]);
}

//  Generate the Uniform Buffers.
void GLSLProgramMaker::generateUniformBuffers(const std::vector<AllocationRange> & allocationRange)
{
    //  
    for (uint32_t i = 0; i < allocationRange.size(); i++)
    {
        //  
        std::string name = "ubR" + std::to_string(i);
        UniformBufferData ub(name);
        ub.bufferType = GLSLBufferType::Uniform;
        ub.baseResourceType = GLSLBaseResourceType::BufferBackedType;
        applyAllocationRange(ub, allocationRange[i]);
        mUBs.push_back(ub);
    }
}


//  Generate the Screen Space Buffer Objects.
void GLSLProgramMaker::generateSSBOs(const std::vector<AllocationRange> & allocationRange)
{
    for (uint32_t i = 0; i < allocationRange.size(); i++)
    {
        std::string name = "ssboR" + std::to_string(i);
        GLSLResourcesDesc ssboR(name);
        ssboR.baseResourceType = GLSLBaseResourceType::BufferBackedType;
        ssboR.bufferType = GLSLBufferType::SSBO;
        applyAllocationRange(ssboR, allocationRange[i]);
        mSSBO.push_back(ssboR);
    }
}

//  Generate the Textures.
void GLSLProgramMaker::generateTextures(const std::vector<AllocationRange> & allocationRange)
{
    for (uint32_t i = 0; i < allocationRange.size(); i++)
    {
        std::string name = "textureR" + std::to_string(i);
        GLSLResourcesDesc textureR(name);
        textureR.baseResourceType = GLSLBaseResourceType::TextureType;
        applyAllocationRange(textureR, allocationRange[i]);
        mTextures.push_back(textureR);
   }
}

//  Generate the Read Write Textures.
void GLSLProgramMaker::generateImages(const std::vector<AllocationRange> & allocationRange)
{
    for (uint32_t i = 0; i < allocationRange.size(); i++)
    {
        std::string name = "imageR" + std::to_string(i);
        GLSLResourcesDesc imageR(name);
        imageR.baseResourceType = GLSLBaseResourceType::ImageType;
        applyAllocationRange(imageR, allocationRange[i]);
        
    }
}

//  Generate the Sampler Textures.
void GLSLProgramMaker::generateSamplerTextures(const std::vector<AllocationRange> & allocationRange)
{
    for (uint32_t i = 0; i < allocationRange.size(); i++)
    {
        std::string name = "samplerR" + std::to_string(i);
        UniformBufferData samplerR(name);
        samplerR.baseResourceType = GLSLBaseResourceType::SamplerType;
        applyAllocationRange(samplerR, allocationRange[i]);
        mSamplers.push_back(samplerR);
    }
}

//  Apply Allocation  Range.
void GLSLProgramMaker::applyAllocationRange(GLSLResourcesDesc & rDesc, const AllocationRange & allocationRange)
{
    //  
    rDesc.attachmentDesc.attachmentPoint = allocationRange.attachmentPoint;
    rDesc.attachmentDesc.attachmentSubpoint = allocationRange.attachmentSubpoint;
    rDesc.attachmentDesc.isAttachmentPointExplicit = allocationRange.isAttachmentPointExplicit;
    rDesc.attachmentDesc.isAttachmentSubpointExplicit = allocationRange.isAttachmentSubpointExplicit;

    rDesc.arrayDesc.isArray = allocationRange.isArray;
    rDesc.arrayDesc.dimensions = allocationRange.dimensions;
}


//  Return the Uniform Buffers.
const std::vector<Falcor::GLSLDescs::UniformBufferData> & GLSLProgramMaker::getUniformBuffers() const
{
    return mUBs;
}

//  Return the SSBOs.
const std::vector<Falcor::GLSLDescs::GLSLResourcesDesc> & GLSLProgramMaker::getSSBOs() const
{
    return mSSBO;
}

//  Return the Textures.
const std::vector<Falcor::GLSLDescs::GLSLResourcesDesc> & GLSLProgramMaker::getTextures() const
{
    return mTextures;
}

//  Return the Samplers.
const std::vector<Falcor::GLSLDescs::GLSLResourcesDesc> & GLSLProgramMaker::getSamplers() const
{
    return mSamplers;
}
