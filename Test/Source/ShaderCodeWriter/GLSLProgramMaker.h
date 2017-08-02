#pragma once
#include "CommonShaderDescs.h"
#include "ShaderProgramMaker.h"
#include "GLSLDescs.h"

using namespace Falcor;
using namespace GLSLDescs;

//  GLSL Program Maker.
class GLSLProgramMaker : public ShaderProgramMaker
{

public:

    struct GLSLProgramDesc : ProgramDesc
    {
        //  The Uniform Buffers Count.
        uint32_t ubsCount;
        uint32_t ssbosCount = 0;

        //  The Textures Count.
        uint32_t textureCount = 0;
        uint32_t rwTextureCount = 0;

        //  The Samplers Count.
        uint32_t samplerCount = 10;

        //  The Maximum number of Bindings per Sets.
        uint32_t maxBindingsPerSet = 45;
        uint32_t maxSets = 0;
    };


    //  Default GLSL Program Maker Constructor.
    GLSLProgramMaker(const GLSLProgramDesc & programDesc);

    //  Default GLSL Program Maker Destructor.
    ~GLSLProgramMaker() = default;

    //  Generate All Resources.
    void generateAllResources(const GLSLProgramDesc & programDesc);

    //  Generate the Uniform Buffers.
    void generateUniformBuffers(const std::vector<AllocationRange> & allocationRange);

    //  Generate the Screen Space Buffer Objects.
    void generateSSBOs(const std::vector<AllocationRange> & allocationRange);
    
    //  Generate the Textures.
    void generateTextures(const std::vector<AllocationRange> & allocationRange);

    //  Generate the Read-Write Textures.
    void generateImages(const std::vector<AllocationRange> & allocationRange);

    //  Generate the Samplers.
    void generateSamplerTextures(const std::vector<AllocationRange> & allocationRange);

    //  Apply the Allocation Range.
    virtual void applyAllocationRange(GLSLResourcesDesc & rDesc, const AllocationRange & allocationRange);

    //  Get the Uniform Buffers.
    virtual const std::vector<UniformBufferData> & getUniformBuffers() const;

    //  Get the SSBO.
    virtual const std::vector<GLSLResourcesDesc> & getSSBOs() const;

    //  Get the Textures.
    virtual const std::vector<GLSLResourcesDesc> & getTextures() const;

    //  Get the Samplers.
    virtual const std::vector<GLSLResourcesDesc> & getSamplers() const;


private:


    //  Get the Uniform Buffers.
    std::vector<UniformBufferData> mUBs;

    //  Get the Screen Space Buffer Object.
    std::vector<GLSLResourcesDesc> mSSBO;

    //  Get the Textures.
    std::vector<GLSLResourcesDesc> mTextures;

    //  Get the Samplers.
    std::vector<GLSLResourcesDesc> mSamplers;
};