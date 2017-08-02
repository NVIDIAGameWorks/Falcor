#include "TestShaderMaker.h"


using namespace Falcor;
using namespace rapidjson;

//  Load the Configuration File.
bool TestShaderMaker::loadConfigFile(const std::string & configFile, std::string errorMessage)
{
    //  The path to the configuration file.
    std::string fullpath;

    //  Find File in Data Directories.
    if (findFileInDataDirectories(configFile, fullpath))
    {
        // Load the file
        std::ifstream fileStream(fullpath);
        std::stringstream strStream;
        strStream << fileStream.rdbuf();
        std::string jsonData = strStream.str();
        rapidjson::StringStream JStream(jsonData.c_str());

        // Get the file directory
        auto last = fullpath.find_last_of("/\\");
        std::string mDirectory = fullpath.substr(0, last);
        mJDoc.ParseStream(JStream);

        //  Check if there is a parse error.
        if (mJDoc.HasParseError())
        {
            size_t line;
            line = std::count(jsonData.begin(), jsonData.begin() + mJDoc.GetErrorOffset(), '\n');
            errorMessage = (std::string("JSON Parse error in line ") + std::to_string(line) + ". " + rapidjson::GetParseError_En(mJDoc.GetParseError()));
            return false;
        }
    }
    else
    {
        errorMessage = (std::string("File Not Found!"));
        return false;
    }

    //  Version.
    Value::MemberIterator version = mJDoc.FindMember("version");
    if (version != mJDoc.MemberEnd())
    {
        mShaderTestLayout.version = version->value.GetString();
    }
    else
    {
        errorMessage = "Version Not Found!";
        return false;
    }


    //  Stages.
    Value::MemberIterator stages = mJDoc.FindMember("stages");
    if (stages != mJDoc.MemberEnd())
    {
        std::string stagesString = stages->value.GetString();
        if (stagesString == "vs/ps")
        {
            mShaderTestLayout.hasVertexShader = true;
            mShaderTestLayout.stages.push_back("vs");
            mShaderTestLayout.hasPixelShader = true;
            mShaderTestLayout.stages.push_back("ps");
        }
        else if (stagesString == "vs/gs/ps")
        {
            mShaderTestLayout.hasVertexShader = true;
            mShaderTestLayout.stages.push_back("vs");
            mShaderTestLayout.hasPixelShader = true;
            mShaderTestLayout.stages.push_back("ps");
            mShaderTestLayout.hasGeometryShader = true;
            mShaderTestLayout.stages.push_back("gs");
        }
        else if (stagesString == "vs/hs/ds/gs/ps")
        {
            mShaderTestLayout.hasVertexShader = true;
            mShaderTestLayout.stages.push_back("vs");
            mShaderTestLayout.hasPixelShader = true;
            mShaderTestLayout.stages.push_back("ps");
            mShaderTestLayout.hasGeometryShader = true;
            mShaderTestLayout.stages.push_back("gs");
            mShaderTestLayout.hasHullShader = true;
            mShaderTestLayout.stages.push_back("hs");
            mShaderTestLayout.hasDomainShader = true;
            mShaderTestLayout.stages.push_back("ds");

        }

        else if (stagesString == "cs")
        {
            mShaderTestLayout.hasComputeShader = true;
            mShaderTestLayout.stages.push_back("cs");
        }
        else
        {
            errorMessage = "Stages Not Valid!";
            return false;
        }
    }
    else
    {
        errorMessage = "Stages Not Found!";
        return false;
    }


    //  Random Seed.
    Value::MemberIterator randomseed = mJDoc.FindMember("random seed");
    if (randomseed != mJDoc.MemberEnd())
    {
        mShaderTestLayout.randomseed = randomseed->value.GetUint64();
    }
    else
    {
        errorMessage = "Random Seed Not Found!";
        return false;
    }




    //  Get the count of each resource.

    //  Constant Buffers.
    Value::MemberIterator mUCBs = mJDoc.FindMember("constant buffers");
    if (mUCBs != mJDoc.MemberEnd())
    {
        if (mUCBs->value.IsUint())
        {
            mShaderTestLayout.constantbuffers = mUCBs->value.GetUint();
        }
        else
        {
            mShaderTestLayout.constantbuffers = 0;
        }
    }
    else
    {
        mShaderTestLayout.constantbuffers = 0;
    }

    //  Texture Buffers.
    Value::MemberIterator tbs = mJDoc.FindMember("texture buffers");
    if (tbs != mJDoc.MemberEnd())
    {
        if (tbs->value.IsUint())
        {
            mShaderTestLayout.texturebuffers = tbs->value.GetUint();
        }
        else
        {
            mShaderTestLayout.texturebuffers = 0;
        }
    }
    else
    {
        mShaderTestLayout.texturebuffers = 0;
    }


    //  Structured Buffers.
    Value::MemberIterator sbs = mJDoc.FindMember("structured buffers");
    if (sbs != mJDoc.MemberEnd())
    {
        if (sbs->value.IsUint())
        {
            mShaderTestLayout.structuredbuffers = sbs->value.GetUint();
        }
        else
        {
            mShaderTestLayout.structuredbuffers = 0;
        }
    }
    else
    {
        mShaderTestLayout.structuredbuffers = 0;
    }

    //  Read/Write Structured Buffers.
    Value::MemberIterator rwSBs = mJDoc.FindMember("rw structured buffers");
    if (rwSBs != mJDoc.MemberEnd())
    {
        if (rwSBs->value.IsUint())
        {
            mShaderTestLayout.rwStructuredBuffers = rwSBs->value.GetUint();
        }
        else
        {
            mShaderTestLayout.rwStructuredBuffers = 0;
        }
    }
    else
    {
        mShaderTestLayout.rwStructuredBuffers = 0;
    }


    //  Raw Buffers.
    Value::MemberIterator rbs = mJDoc.FindMember("raw buffers");
    if (rbs != mJDoc.MemberEnd())
    {
        if (rbs->value.IsUint())
        {
            mShaderTestLayout.rawbuffers = rbs->value.GetUint();
        }
        else
        {
            mShaderTestLayout.rawbuffers = 0;
        }
    }
    else
    {
        mShaderTestLayout.rawbuffers = 0;
    }

    //  Raw Buffers.
    Value::MemberIterator rwRBs = mJDoc.FindMember("rw raw buffers");
    if (rwRBs != mJDoc.MemberEnd())
    {
        if (rwRBs->value.IsUint())
        {
            mShaderTestLayout.rwRawBuffers = rwRBs->value.GetUint();
        }
        else
        {
            mShaderTestLayout.rwRawBuffers = 0;
        }
    }
    else
    {
        mShaderTestLayout.rwRawBuffers = 0;
    }

    //  Textures.
    Value::MemberIterator texs = mJDoc.FindMember("textures");
    if (texs != mJDoc.MemberEnd())
    {
        if (texs->value.IsUint())
        {
            mShaderTestLayout.textures = texs->value.GetUint();
        }
        else
        {
            mShaderTestLayout.textures = 0;
        }
    }
    else
    {
        mShaderTestLayout.textures = 0;
    }

    //  RW Textures.
    Value::MemberIterator rwTexs = mJDoc.FindMember("rw textures");
    if (rwTexs != mJDoc.MemberEnd())
    {
        if (rwTexs->value.IsUint())
        {
            mShaderTestLayout.rwTextures = rwTexs->value.GetUint();
        }
        else
        {
            mShaderTestLayout.rwTextures = 0;
        }
    }
    else
    {
        mShaderTestLayout.rwTextures = 0;
    }



    //  Samplers.
    Value::MemberIterator samps = mJDoc.FindMember("samplers");
    if (samps != mJDoc.MemberEnd())
    {
        if (samps->value.IsUint())
        {
            mShaderTestLayout.samplers = samps->value.GetUint();
        }
        else
        {
            mShaderTestLayout.samplers = 0;
        }

    }
    else
    {
        mShaderTestLayout.samplers = 0;
    }



    //  Mode.
    Value::MemberIterator mode = mJDoc.FindMember("mode");
    if (mode != mJDoc.MemberEnd())
    {
        mShaderTestLayout.mode = mode->value.GetString();
    }
    else
    {
        mShaderTestLayout.mode = "both";
    }

    //  
    Value::MemberIterator iRT = mJDoc.FindMember("invalidResourceType");
    if (iRT != mJDoc.MemberEnd())
    {
        mShaderTestLayout.invalidResourceType = iRT->value.GetString();
    }
    else
    {
        mShaderTestLayout.invalidResourceType = "constant buffer";
    }

    return true;
}

//  Set the Version Restricted Content.
void TestShaderMaker::generateVersionRestrictedConstants()
{
    //  Maximum Struct Variables.
    mShaderTestLayout.maxStructVariablesCount = 3;

    //  
    if (mShaderTestLayout.version == "DX5.0")
    {
        mShaderTestLayout.allowExplicitSpaces = false;
        mShaderTestLayout.allowConstantBuffersArray = false;
        mShaderTestLayout.allowStructConstantBuffers = false;
        mShaderTestLayout.allowUnboundedConstantBuffers = false;
        mShaderTestLayout.allowUnboundedResourceArrays = false;

        if (mShaderTestLayout.hasPixelShader)
        {
            TestShaderWriter::ShaderResourcesData * psstage = getShaderStageResourceData("ps");

            uint32_t pixelInCount = (rand() % 8) + 1;
            for (uint32_t i = 0; i < pixelInCount; i++)
            {
                psstage->inTargets.push_back(i);

            }


            uint32_t pixelOutCount = (rand() % 8) + 1;
            for (uint32_t i = 0; i < pixelOutCount; i++)
            {
                psstage->outTargets.push_back(i);

                mUSpacesIndexes[0].nextRegisterIndex++;
            }

        }
    }

    //  
    if (mShaderTestLayout.version == "DX5.1")
    {
        mShaderTestLayout.allowExplicitSpaces = true;
        mShaderTestLayout.allowConstantBuffersArray = true;
        mShaderTestLayout.allowStructConstantBuffers = true;
        mShaderTestLayout.allowUnboundedConstantBuffers = true;
        mShaderTestLayout.allowUnboundedResourceArrays = true;

        if (mShaderTestLayout.hasPixelShader)
        {
            TestShaderWriter::ShaderResourcesData * psstage = getShaderStageResourceData("ps");

            uint32_t pixelInCount = (rand() % 8) + 1;
            for (uint32_t i = 0; i < pixelInCount; i++)
            {
                psstage->inTargets.push_back(i);

            }

            uint32_t pixelOutCount = (rand() % 8) + 1;
            for (uint32_t i = 0; i < pixelOutCount; i++)
            {
                psstage->outTargets.push_back(i);
            }
        }
    
    }

    if (mShaderTestLayout.hasVertexShader)
    {
        //  Get the Vertex Shader Stage Data, and set the VS Data.
        TestShaderWriter::ShaderResourcesData * vsstage = getShaderStageResourceData("vs");

        uint32_t vertexOutCount = (rand() % 8) + 1;
        for (uint32_t i = 0; i < vertexOutCount; i++)
        {
            vsstage->outTargets.push_back(i);
        }
    }


    if (mShaderTestLayout.hasHullShader)
    {
        TestShaderWriter::ShaderResourcesData * hsstage = getShaderStageResourceData("hs");

        uint32_t gsInCount = (rand() % 8) + 1;
        for (uint32_t i = 0; i < gsInCount; i++)
        {
            hsstage->inTargets.push_back(i);
        }

        uint32_t gsOutCount = (rand() % 8) + 1;
        for (uint32_t i = 0; i < gsOutCount; i++)
        {
            hsstage->outTargets.push_back(i);
        }

    }

    if (mShaderTestLayout.hasDomainShader)
    {
        TestShaderWriter::ShaderResourcesData * dsstage = getShaderStageResourceData("ds");

        uint32_t gsInCount = (rand() % 8) + 1;
        for (uint32_t i = 0; i < gsInCount; i++)
        {
            dsstage->inTargets.push_back(i);
        }

        uint32_t gsOutCount = (rand() % 8) + 1;
        for (uint32_t i = 0; i < gsOutCount; i++)
        {
            dsstage->outTargets.push_back(i);
        }
    }

    if (mShaderTestLayout.hasGeometryShader)
    {
        TestShaderWriter::ShaderResourcesData * gsstage = getShaderStageResourceData("gs");

        uint32_t gsInCount = (rand() % 8) + 1;
        for (uint32_t i = 0; i < gsInCount; i++)
        {
            gsstage->inTargets.push_back(i);
        }

        uint32_t gsOutCount = (rand() % 8) + 1;
        for (uint32_t i = 0; i < gsOutCount; i++)
        {
            gsstage->outTargets.push_back(i);
        }

    }

}

//  
const TestShaderMaker::TestShaderLayout & TestShaderMaker::viewShaderTestLayout() const
{
    return mShaderTestLayout;
}

TestShaderMaker::TestShaderLayout & TestShaderMaker::getShaderTestLayout()
{
    return mShaderTestLayout;
}

//  Return the Vertex Shader Code.
std::string TestShaderMaker::getVertexShaderCode() const
{
    return mShaderMaker.getVertexShaderCode();
}

//  Return the Hull Shader Code.
std::string TestShaderMaker::getHullShaderCode() const
{
    return mShaderMaker.getHullShaderCode();
}


//  Return the Domain Shader Code.
std::string TestShaderMaker::getDomainShaderCode() const
{
    return mShaderMaker.getDomainShaderCode();
}


//  Return the Geometry Shader Code.
std::string TestShaderMaker::getGeometryShaderCode() const
{
    return mShaderMaker.getGeometryShaderCode();
}

//  Return the Pixel Shader Code.
std::string TestShaderMaker::getPixelShaderCode() const
{
    return mShaderMaker.getPixelShaderCode();
}

//  Return the Compute Shader Code.
std::string TestShaderMaker::getComputeShaderCode() const
{
    return mShaderMaker.getComputeShaderCode();
}


//  
bool TestShaderMaker::verifyProgramResources(ProgramReflection::SharedConstPtr programReflection, std::string & errorMessage)
{

    //  Verify the Stages.
    bool vsVerify = true;
    if (mShaderTestLayout.hasVertexShader)
    {
        vsVerify = verifyAllResources(mShaderMaker.viewVSResourceData(), programReflection, errorMessage);
    }

    bool hsVerify = true;
    if (mShaderTestLayout.hasHullShader)
    {
        hsVerify = verifyAllResources(mShaderMaker.viewHSResourceData(), programReflection, errorMessage);
    }


    bool dsVerify = true;
    if (mShaderTestLayout.hasDomainShader)
    {
        dsVerify = verifyAllResources(mShaderMaker.viewDSResourceData(), programReflection, errorMessage);
    }


    bool gsVerify = true;
    if (mShaderTestLayout.hasGeometryShader)
    {
        gsVerify = verifyAllResources(mShaderMaker.viewGSResourceData(), programReflection, errorMessage);
    }


    bool psVerify = true;
    if (mShaderTestLayout.hasPixelShader)
    {
        psVerify = verifyAllResources(mShaderMaker.viewPSResourceData(), programReflection, errorMessage);
    }


    bool csVerify = true;
    if (mShaderTestLayout.hasComputeShader)
    {
        csVerify = verifyAllResources(mShaderMaker.viewCSResourceData(), programReflection, errorMessage);
    }

    return csVerify;


    //  
    return vsVerify && hsVerify && dsVerify && gsVerify && psVerify && csVerify;
}


//  Generate the Shader Program.
void TestShaderMaker::generateShaderProgram()
{
    //
    RegisterCount initialRegisters;

    for (uint32_t i = 0; i < 20; i++)
    {
        mBSpacesIndexes[i] = initialRegisters;
        mBSpaces.push_back(i);

        mSSpacesIndexes[i] = initialRegisters;
        mSSpaces.push_back(i);

        mTSpacesIndexes[i] = initialRegisters;
        mTSpaces.push_back(i);

        mUSpacesIndexes[i] = initialRegisters;
        mUSpaces.push_back(i);
    }

    mMaxBSpace = 20;
    mMaxUSpace = 20;
    mMaxSSpace = 20;
    mMaxTSpace = 20;


    //  Set up some generation constants.
    generateVersionRestrictedConstants();

    //      
    std::vector<TestShaderWriter::ConstantBufferResource> mUCBs;

    //  
    std::vector<TestShaderWriter::StructuredBufferResource> sbs;
    std::vector<TestShaderWriter::StructuredBufferResource> rwSBs;

    //
    std::vector<TestShaderWriter::RawBufferResource> rbs;
    std::vector<TestShaderWriter::RawBufferResource> rwRBs;

    //
    std::vector<TestShaderWriter::TextureResource> texs;
    std::vector<TestShaderWriter::TextureResource> rwTexs;

    //
    std::vector<TestShaderWriter::SamplerResource> samps;


    //  Generate all the Resources.

    if (mShaderTestLayout.mode == "invalid")
    {
        if (mShaderTestLayout.invalidResourceType == "constant buffer")
        {
            generateErrorConstantBufferResources(mUCBs);
        }
        else if (mShaderTestLayout.invalidResourceType == "structured buffer")
        {
            generateErrorStructuredBufferResources(sbs);
        }
        else if (mShaderTestLayout.invalidResourceType == "raw buffers")
        {
            generateErrorRawBufferResources(rbs);
        }
        else if (mShaderTestLayout.invalidResourceType == "textures")
        {
            generateErrorTextureResources(texs);
        }
        else if (mShaderTestLayout.invalidResourceType == "samplers")
        {
            generateErrorSamplersResources(samps);
        }
        else if(mShaderTestLayout.invalidResourceType == "uavs")
        {
            generateErrorUAVResources(rwSBs, rwTexs);
        }
        else
        {
            generateConstantBufferResources(mUCBs);
        }
    }

    generateConstantBufferResources(mUCBs);
    distributeConstantBuffers(mUCBs);


    generateStructuredBufferResources(sbs, rwSBs);
    distributeStructuredBuffers(sbs, rwSBs);

    generateRawBufferResources(rbs, rwRBs);
    distributeRawBuffers(rbs, rwRBs);

    generateTextureResources(texs, rwTexs);
    distributeTextures(texs, rwTexs);

    generateSamplersResources(samps);
    distributeSamplers(samps);

}


//  Generate the Constant Buffer Resources that cause an error.
void TestShaderMaker::generateErrorConstantBufferResources(std::vector<TestShaderWriter::ConstantBufferResource> & mUCBs)
{
        uint32_t stageCount = (uint32_t)mShaderTestLayout.stages.size();

        //  Multiple stages.
        TestShaderWriter::ConstantBufferResource erCBR0;
        erCBR0.variable = "erCBR";
        erCBR0.isUsedResource = true;

        TestShaderWriter::ConstantBufferResource erCBR1;
        erCBR1.variable = "erCBR";
        erCBR1.isUsedResource = true;

        uint32_t structVariableCount = 0;
        uint32_t selectVariable = 0;


        structVariableCount = (rand() % (mShaderTestLayout.maxStructVariablesCount)) + 1;
        selectVariable = rand() % structVariableCount;

        //  
        for (uint32_t svIndex = 0; svIndex < structVariableCount; svIndex++)
        {
            TestShaderWriter::StructVariable sv;

            if (svIndex == selectVariable)
            {
                sv.isUsed = true;
            }
            else
            {
                sv.isUsed = false;
            }

            sv.structVariable = "erCBR_sv" + std::to_string(svIndex);
            sv.useStructVariable = "use_" + sv.structVariable;
            sv.variableType = ProgramReflection::Variable::Type::Float4;
            erCBR0.structVariables.push_back(sv);
        }


        structVariableCount = structVariableCount + 3;
        selectVariable = rand() % structVariableCount;

        //  
        for (uint32_t svIndex = 0; svIndex < structVariableCount; svIndex++)
        {
            TestShaderWriter::StructVariable sv;

            if (svIndex == selectVariable)
            {
                sv.isUsed = true;
            }
            else
            {
                sv.isUsed = false;
            }

            sv.structVariable = "erCBR_sv" + std::to_string(svIndex);
            sv.useStructVariable = "use_" + sv.structVariable;
            sv.variableType = ProgramReflection::Variable::Type::Float4;
            erCBR1.structVariables.push_back(sv);
        }


        //  
        if (stageCount == 1)
        {
            //  We have one stage, in which case we have to go for straight redefinition.
            TestShaderWriter::ShaderResourcesData * sRD = getShaderStageResourceData(mShaderTestLayout.stages[0]);

            sRD->mUCBs.push_back(erCBR0);
            sRD->mUCBs.push_back(erCBR1);
        }
        else
        {

            uint32_t errorType = rand() % 2;

            if (errorType)
            {
                //  Two buffers with the same name, in different stages, with different layouts.
                erCBR0.isExplicitIndex = true;
                erCBR0.regIndex = 0;
                erCBR0.isExplicitSpace = true && mShaderTestLayout.allowExplicitSpaces;
                erCBR0.regSpace = 0;

                erCBR1.isExplicitIndex = true;
                erCBR1.regIndex = 0;
                erCBR1.isExplicitSpace = true && mShaderTestLayout.allowExplicitSpaces;
                erCBR1.regSpace = 0;

                mBSpacesIndexes[0].nextRegisterIndex++;
            }
            else
            {
                erCBR0.isExplicitIndex = true;
                erCBR0.regIndex = 0;
                erCBR0.isExplicitSpace = true && mShaderTestLayout.allowExplicitSpaces;
                erCBR0.regSpace = 0;

                erCBR1.isExplicitIndex = true;
                erCBR1.regIndex = 1;
                erCBR1.isExplicitSpace = true && mShaderTestLayout.allowExplicitSpaces;
                erCBR1.regSpace = 0;

                mBSpacesIndexes[0].nextRegisterIndex++;
                mBSpacesIndexes[0].nextRegisterIndex++;
            }

            for (uint32_t stageIndex = 0; stageIndex < stageCount; stageIndex++)
            {
                TestShaderWriter::ShaderResourcesData * sRD = getShaderStageResourceData(mShaderTestLayout.stages[stageIndex]);

                //  
                if (stageIndex % 2 == 0)
                {
                    sRD->mUCBs.push_back(erCBR0);
                }
                else
                {
                    sRD->mUCBs.push_back(erCBR1);
                }
            }
        }
}


//  Generate the Error Structured Buffer Resources.
void TestShaderMaker::generateErrorStructuredBufferResources(std::vector<TestShaderWriter::StructuredBufferResource> & sbs)
{
    uint32_t stageCount = (uint32_t)mShaderTestLayout.stages.size();

    //  Multiple stages.
    TestShaderWriter::StructuredBufferResource erSBR0;
    erSBR0.variable = "erSBR";
    erSBR0.structVariable = "erSBR0s";
    erSBR0.isUsedResource = true;
    erSBR0.usesExternalStructVariables = true;

    TestShaderWriter::StructuredBufferResource erSBR1;
    erSBR1.variable = "erSBR";
    erSBR1.structVariable = "erSBR1s";
    erSBR1.isUsedResource = true;
    erSBR1.usesExternalStructVariables = true;

    uint32_t structVariableCount = 0;
    uint32_t selectVariable = 0;


    structVariableCount = (rand() % (mShaderTestLayout.maxStructVariablesCount)) + 1;
    selectVariable = rand() % structVariableCount;

    //  
    for (uint32_t svIndex = 0; svIndex < structVariableCount; svIndex++)
    {
        TestShaderWriter::StructVariable sv;

        if (svIndex == selectVariable)
        {
            sv.isUsed = true;
        }
        else
        {
            sv.isUsed = false;
        }

        sv.structVariable = "erSBR_sv_0_" + std::to_string(svIndex);
        sv.useStructVariable = "use_" + sv.structVariable;
        sv.variableType = ProgramReflection::Variable::Type::Float4;
        erSBR0.externalStructVariables.push_back(sv);
    }


    structVariableCount = structVariableCount + 3;
    selectVariable = rand() % structVariableCount;

    //  
    for (uint32_t svIndex = 0; svIndex < structVariableCount; svIndex++)
    {
        TestShaderWriter::StructVariable sv;

        if (svIndex == selectVariable)
        {
            sv.isUsed = true;
        }
        else
        {
            sv.isUsed = false;
        }

        sv.structVariable = "erSBR_sv_1_" + std::to_string(svIndex);
        sv.useStructVariable = "use_" + sv.structVariable;
        sv.variableType = ProgramReflection::Variable::Type::Float4;
        erSBR1.externalStructVariables.push_back(sv);
    }


    //  
    if (stageCount == 1)
    {
        //  We have one stage, in which case we have to go for straight redefinition.
        TestShaderWriter::ShaderResourcesData * sRD = getShaderStageResourceData(mShaderTestLayout.stages[0]);

        sRD->sbs.push_back(erSBR0);
        sRD->sbs.push_back(erSBR1);
    }
    else
    {

        uint32_t errorType = rand() % 2;

        if (errorType)
        {
            //  Two buffers with the same name, in different stages, with different layouts using the same register.
            erSBR0.isExplicitIndex = true;
            erSBR0.regIndex = 0;
            erSBR0.isExplicitSpace = true && mShaderTestLayout.allowExplicitSpaces;
            erSBR0.regSpace = 0;

            erSBR1.isExplicitIndex = true;
            erSBR1.regIndex = 0;
            erSBR1.isExplicitSpace = true && mShaderTestLayout.allowExplicitSpaces;
            erSBR1.regSpace = 0;

            mTSpacesIndexes[0].nextRegisterIndex++;
        }
        else
        {
            //  Two buffers with the same name, in different stages, with different layouts.
            erSBR0.isExplicitIndex = true;
            erSBR0.regIndex = 0;
            erSBR0.isExplicitSpace = true && mShaderTestLayout.allowExplicitSpaces;
            erSBR0.regSpace = 0;

            mTSpacesIndexes[0].nextRegisterIndex++;

            erSBR1.isExplicitIndex = true;
            erSBR1.regIndex = 1;
            erSBR1.isExplicitSpace = true && mShaderTestLayout.allowExplicitSpaces;
            erSBR1.regSpace = 0;

            mTSpacesIndexes[0].nextRegisterIndex++;
        }

        for (uint32_t stageIndex = 0; stageIndex < stageCount; stageIndex++)
        {
            TestShaderWriter::ShaderResourcesData * sRD = getShaderStageResourceData(mShaderTestLayout.stages[stageIndex]);

            //  
            if (stageIndex % 2 == 0)
            {
                sRD->sbs.push_back(erSBR0);
            }
            else
            {
                sRD->sbs.push_back(erSBR1);
            }
        }
    }

}

//  Generate the Error Raw Buffer Resources.
void TestShaderMaker::generateErrorRawBufferResources(std::vector<TestShaderWriter::RawBufferResource> & rbs)
{
    uint32_t stageCount = (uint32_t)mShaderTestLayout.stages.size();

    //  Multiple stages.
    TestShaderWriter::RawBufferResource erRB0;
    erRB0.variable = "erRB";
    erRB0.isUsedResource = true;


    TestShaderWriter::RawBufferResource erRB1;
    erRB1.variable = "erRB";
    erRB1.isUsedResource = true;

    //  Two buffers with the same name, in different stages with different registers.
    erRB0.isExplicitIndex = true;
    erRB0.regIndex = 0;
    erRB0.isExplicitSpace = true && mShaderTestLayout.allowExplicitSpaces;
    erRB0.regSpace = 0;

    mTSpacesIndexes[0].nextRegisterIndex++;

    erRB1.isExplicitIndex = true;
    erRB1.regIndex = 1;
    erRB1.isExplicitSpace = true && mShaderTestLayout.allowExplicitSpaces;
    erRB1.regSpace = 0;

    mTSpacesIndexes[0].nextRegisterIndex++;

    //  
    if (stageCount == 1)
    {
        //  We have one stage, in which case we have to go for straight redefinition.
        TestShaderWriter::ShaderResourcesData * sRD = getShaderStageResourceData(mShaderTestLayout.stages[0]);

        sRD->rbs.push_back(erRB0);
        sRD->rbs.push_back(erRB1);
    }
    else
    {
        for (uint32_t stageIndex = 0; stageIndex < stageCount; stageIndex++)
        {
            TestShaderWriter::ShaderResourcesData * sRD = getShaderStageResourceData(mShaderTestLayout.stages[stageIndex]);

            //  
            if (stageIndex % 2 == 0)
            {
                sRD->rbs.push_back(erRB0);
            }
            else
            {
                sRD->rbs.push_back(erRB1);
            }
        }
    }
}

//  Generate the Error Texture Resources.
void TestShaderMaker::generateErrorTextureResources(std::vector<TestShaderWriter::TextureResource> & texs)
{
    uint32_t stageCount = (uint32_t)mShaderTestLayout.stages.size();

    TestShaderWriter::TextureResource erTex0;
    erTex0.variable = "erTex";
    erTex0.isUsedResource = true;
    erTex0.textureType = ProgramReflection::Variable::Type::Float4;
    erTex0.textureResourceType = ProgramReflection::Resource::Dimensions::Texture2D;

    TestShaderWriter::TextureResource erTex1;
    erTex1.variable = "erTex";
    erTex1.isUsedResource = true;
    erTex1.textureType = ProgramReflection::Variable::Type::Float4;
    erTex1.textureResourceType = ProgramReflection::Resource::Dimensions::Texture2D;

    erTex0.isExplicitIndex = true;
    erTex0.regIndex = 0;
    erTex0.isExplicitSpace = true && mShaderTestLayout.allowExplicitSpaces;
    erTex0.regSpace = 0;
    mTSpacesIndexes[0].nextRegisterIndex++;

    erTex1.isExplicitIndex = true;
    erTex1.regIndex = 1;
    erTex1.isExplicitSpace = true && mShaderTestLayout.allowExplicitSpaces;
    erTex1.regSpace = 0;
    mTSpacesIndexes[0].nextRegisterIndex++;

    if (stageCount == 1)
    {
        //  We have one stage, in which case we have to go for straight redefinition.
        TestShaderWriter::ShaderResourcesData * sRD = getShaderStageResourceData(mShaderTestLayout.stages[0]);

        sRD->textures.push_back(erTex0);
        sRD->textures.push_back(erTex1);
    }
    else
    {
        for (uint32_t stageIndex = 0; stageIndex < stageCount; stageIndex++)
        {
            TestShaderWriter::ShaderResourcesData * sRD = getShaderStageResourceData(mShaderTestLayout.stages[stageIndex]);

            //  
            if (stageIndex % 2 == 0)
            {
                sRD->textures.push_back(erTex0);
            }
            else
            {
                sRD->textures.push_back(erTex1);
            }
        }
    }
}

//  Generate Error Sampler Resources.
void TestShaderMaker::generateErrorSamplersResources(std::vector<TestShaderWriter::SamplerResource> & samps)
{
    uint32_t stageCount = (uint32_t)mShaderTestLayout.stages.size();

    TestShaderWriter::SamplerResource erSampler0;
    erSampler0.variable = "erTex";
    erSampler0.isUsedResource = true;

    TestShaderWriter::SamplerResource erSampler1;
    erSampler1.variable = "erTex";
    erSampler1.isUsedResource = true;


    erSampler0.isExplicitIndex = true;
    erSampler0.regIndex = 0;
    erSampler0.isExplicitSpace = true && mShaderTestLayout.allowExplicitSpaces;
    erSampler0.regSpace = 0;
    mBSpacesIndexes[0].nextRegisterIndex++;

    erSampler1.isExplicitIndex = true;
    erSampler1.regIndex = 1;
    erSampler1.isExplicitSpace = true && mShaderTestLayout.allowExplicitSpaces;
    erSampler1.regSpace = 0;
    mSSpacesIndexes[0].nextRegisterIndex++;

    if (stageCount == 1)
    {
        //  We have one stage, in which case we have to go for straight redefinition.
        TestShaderWriter::ShaderResourcesData * sRD = getShaderStageResourceData(mShaderTestLayout.stages[0]);

        sRD->samplers.push_back(erSampler0);
        sRD->samplers.push_back(erSampler1);
    }
    else
    {
        for (uint32_t stageIndex = 0; stageIndex < stageCount; stageIndex++)
        {
            TestShaderWriter::ShaderResourcesData * sRD = getShaderStageResourceData(mShaderTestLayout.stages[stageIndex]);

            //  
            if (stageIndex % 2 == 0)
            {
                sRD->samplers.push_back(erSampler0);
            }
            else
            {
                sRD->samplers.push_back(erSampler1);
            }
        }
    }
}

//  Generate the Error UAV Resources.
void TestShaderMaker::generateErrorUAVResources(std::vector<TestShaderWriter::StructuredBufferResource> & rwSBs, std::vector<TestShaderWriter::TextureResource> & rwTexs)
{
    uint32_t stageCount = (uint32_t)mShaderTestLayout.stages.size();

    TestShaderWriter::StructuredBufferResource erSBR;
    erSBR.usesExternalStructVariables = false;
    erSBR.isArray = false;
    erSBR.variable = "erRWUAV";
    erSBR.bufferType = ProgramReflection::BufferReflection::StructuredType::Default;
    erSBR.sbType = ProgramReflection::Variable::Type::Float4;
    erSBR.shaderAccess = ProgramReflection::ShaderAccess::ReadWrite;

    erSBR.isExplicitIndex = true;
    erSBR.regIndex = mUSpacesIndexes[0].nextRegisterIndex;

    erSBR.isExplicitSpace = true && mShaderTestLayout.allowExplicitSpaces;
    erSBR.regSpace = 0;

    uint32_t structVariableCount = (rand() % (mShaderTestLayout.maxStructVariablesCount)) + 1;
    uint32_t selectVariable = rand() % structVariableCount;

    //  
    for (uint32_t svIndex = 0; svIndex < structVariableCount; svIndex++)
    {
        TestShaderWriter::StructVariable sv;

        if (svIndex == selectVariable)
        {
            sv.isUsed = true;
        }
        else
        {
            sv.isUsed = false;
        }

        sv.structVariable = "erSBR_sv_0_" + std::to_string(svIndex);
        sv.useStructVariable = "use_" + sv.structVariable;
        sv.variableType = ProgramReflection::Variable::Type::Float4;
        erSBR.externalStructVariables.push_back(sv);
    }


    TestShaderWriter::TextureResource erTex;
    erTex.variable = "erTex";
    erTex.isUsedResource = true;
    erTex.textureType = ProgramReflection::Variable::Type::Float4;
    erTex.textureResourceType = ProgramReflection::Resource::Dimensions::Texture2D;
    erTex.shaderAccess = ProgramReflection::ShaderAccess::ReadWrite;


    erTex.isExplicitIndex = true;
    erTex.regIndex = mUSpacesIndexes[0].nextRegisterIndex;

    erTex.isExplicitSpace = true && mShaderTestLayout.allowExplicitSpaces;
    erTex.regSpace = 0;

    mUSpacesIndexes[0].nextRegisterIndex++;

    for (uint32_t stageIndex = 0; stageIndex < stageCount; stageIndex++)
    {
        TestShaderWriter::ShaderResourcesData * sRD = getShaderStageResourceData(mShaderTestLayout.stages[stageIndex]);

        //  
        if (stageIndex % 2 == 0)
        {
            sRD->sbs.push_back(erSBR);
        }
        else
        {
            sRD->textures.push_back(erTex);
        }
    }

}

//  Generate the Constant Buffer Resources.
void TestShaderMaker::generateConstantBufferResources(std::vector<TestShaderWriter::ConstantBufferResource> & mUCBs)
{
    //  Generate the Constant Buffers.
    for (uint32_t cbIndex = 0; cbIndex < mShaderTestLayout.constantbuffers; cbIndex++)
    {
        //  
        TestShaderWriter::ConstantBufferResource cbResource;
        cbResource.variable = "cbR" + std::to_string(cbIndex);

        //  Check whether we use the resource.
        uint32_t useChance = rand() % 2;

        if (useChance)
        {
            cbResource.isUsedResource = true;
        }

        //  Chance for Explicit Index and Spaces.
        cbResource.isExplicitIndex = ((rand() % 2) == 0);
        cbResource.isExplicitSpace = ((rand() % 2) == 0) && mShaderTestLayout.allowExplicitSpaces;

        //  
        getRegisterIndexAndSpace(mShaderTestLayout.allowExplicitSpaces, cbResource.isExplicitIndex, cbResource.isExplicitSpace, cbResource.isArray, cbResource.isUnbounded, std::vector<uint32_t>(), mBSpacesIndexes, mBSpaces, mMaxBSpace, cbResource.regIndex, cbResource.regSpace);

        //  Struct Variable Count.
        uint32_t structVariableCount = (rand() % (mShaderTestLayout.maxStructVariablesCount)) + 1;

        //  Which variable to use.
        uint32_t selectVariable = rand() % structVariableCount;

        //  
        for (uint32_t svIndex = 0; svIndex < structVariableCount; svIndex++)
        {
            TestShaderWriter::StructVariable sv;

            if (svIndex == selectVariable)
            {
                sv.isUsed = true;
            }
            else
            {
                sv.isUsed = false;
            }

            sv.structVariable = "cbR" + std::to_string(cbIndex) + "_sv" + std::to_string(svIndex);
            sv.useStructVariable = "use_" + sv.structVariable;
            sv.variableType = ProgramReflection::Variable::Type::Float4;
            cbResource.structVariables.push_back(sv);
        }

        //  
        mUCBs.push_back(cbResource);
    }
}



//  Generate the Structured Buffer Resources.
void TestShaderMaker::generateStructuredBufferResources(std::vector<TestShaderWriter::StructuredBufferResource> & sbs, std::vector<TestShaderWriter::StructuredBufferResource> & rwSBs)
{
    //  Generate the Structured Buffers.
    for (uint32_t i = 0; i < mShaderTestLayout.structuredbuffers; i++)
    {
        //  
        TestShaderWriter::StructuredBufferResource sbResource;
        sbResource.usesExternalStructVariables = false;
        sbResource.isArray = false;
        sbResource.variable = "sbR" + std::to_string(i);
        sbResource.bufferType = ProgramReflection::BufferReflection::StructuredType::Default;
        sbResource.sbType = ProgramReflection::Variable::Type::Float4;


        //  The chance of it being used.
        uint32_t useChance = rand() % 2;
        if (useChance)
        {
            sbResource.isUsedResource = true;
        }


        //  Chance for Explicit Index and Spaces.
        sbResource.isExplicitIndex = ((rand() % 2) == 0);
        sbResource.isExplicitSpace = ((rand() % 2) == 0) && mShaderTestLayout.allowExplicitSpaces;


        getRegisterIndexAndSpace(mShaderTestLayout.allowExplicitSpaces, sbResource.isExplicitIndex, sbResource.isExplicitSpace, sbResource.isArray, sbResource.isUnbounded, sbResource.arrayDimensions, mTSpacesIndexes, mTSpaces, mMaxTSpace, sbResource.regIndex, sbResource.regSpace);


        //  
        sbs.push_back(sbResource);
    }


    //  Generate the RW Structured Buffers.
    for (uint32_t i = 0; i < mShaderTestLayout.rwStructuredBuffers; i++)
    {
        //  
        TestShaderWriter::StructuredBufferResource sbResource;
        sbResource.usesExternalStructVariables = false;
        sbResource.isArray = false;
        sbResource.variable = "rwSBR" + std::to_string(i);
        sbResource.bufferType = ProgramReflection::BufferReflection::StructuredType::Default;
        sbResource.sbType = ProgramReflection::Variable::Type::Float4;
        sbResource.shaderAccess = ProgramReflection::ShaderAccess::ReadWrite;

        //  The chance of it being used.
        uint32_t useChance = rand() % 2;
        if (useChance)
        {
            sbResource.isUsedResource = true;
        }


        //  Chance for Explicit Index and Spaces.
        sbResource.isExplicitIndex = ((i % 2) == 0);
        sbResource.isExplicitSpace = ((i % 2) == 0) && mShaderTestLayout.allowExplicitSpaces;

        getRegisterIndexAndSpace(mShaderTestLayout.allowExplicitSpaces, sbResource.isExplicitIndex, sbResource.isExplicitSpace, sbResource.isArray, sbResource.isUnbounded, sbResource.arrayDimensions, mUSpacesIndexes, mUSpaces, mMaxUSpace, sbResource.regIndex, sbResource.regSpace);

        //  
        rwSBs.push_back(sbResource);
    }

}



//  Generate the Raw Buffer Resources.
void TestShaderMaker::generateRawBufferResources(std::vector<TestShaderWriter::RawBufferResource> & rbs, std::vector<TestShaderWriter::RawBufferResource> & rwRbs)
{
    //  Generate the Raw Buffers.
    for (uint32_t i = 0; i < mShaderTestLayout.rawbuffers; i++)
    {
        //  
        TestShaderWriter::RawBufferResource rbResource;
        rbResource.variable = "rbR" + std::to_string(i);


        //  The chance of it being used.
        uint32_t useChance = rand() % 2;
        if (useChance)
        {
            rbResource.isUsedResource = true;
        }


        //  Chance for Explicit Index and Spaces.
        rbResource.isExplicitIndex = ((rand() % 2) == 0);
        rbResource.isExplicitSpace = ((rand() % 2) == 0) && mShaderTestLayout.allowExplicitSpaces;

        getRegisterIndexAndSpace(mShaderTestLayout.allowExplicitSpaces, rbResource.isExplicitIndex, rbResource.isExplicitSpace, rbResource.isArray, rbResource.isUnbounded, rbResource.arrayDimensions, mTSpacesIndexes, mTSpaces, mMaxTSpace, rbResource.regIndex, rbResource.regSpace);

        rbs.push_back(rbResource);
    }

    //
    for (uint32_t i = 0; i < mShaderTestLayout.rwRawBuffers; i++)
    {
        //  
        TestShaderWriter::RawBufferResource rbResource;
        rbResource.variable = "rwRBR" + std::to_string(i);
        rbResource.shaderAccess = ProgramReflection::ShaderAccess::ReadWrite;


        //  Chance for Explicit Index and Spaces.
        rbResource.isExplicitIndex = ((rand() % 2) == 0);
        rbResource.isExplicitSpace = ((rand() % 2) == 0) && mShaderTestLayout.allowExplicitSpaces;


        getRegisterIndexAndSpace(mShaderTestLayout.allowExplicitSpaces, rbResource.isExplicitIndex, rbResource.isExplicitSpace, rbResource.isArray, rbResource.isUnbounded, rbResource.arrayDimensions, mUSpacesIndexes, mUSpaces, mMaxUSpace, rbResource.regIndex, rbResource.regSpace);


        rwRbs.push_back(rbResource);
    }
}



//  Generate the Texture Resources.
void TestShaderMaker::generateTextureResources(std::vector<TestShaderWriter::TextureResource> & texs, std::vector<TestShaderWriter::TextureResource> & rwTexs)
{
    //  Generate the Textures.
    for (uint32_t i = 0; i < mShaderTestLayout.textures; i++)
    {
        TestShaderWriter::TextureResource textureResource;
        textureResource.textureResourceType = ProgramReflection::Resource::Dimensions::Texture2D;
        textureResource.textureType = ProgramReflection::Variable::Type::Float4;
        textureResource.variable = "textureR" + std::to_string(i);


        //  The chance of it being used.
        uint32_t useChance = rand() % 2;

        if (useChance)
        {
            textureResource.isUsedResource = true;
        }


        //  Chance for Explicit Index and Spaces.
        textureResource.isExplicitIndex = ((rand() % 2) == 0);
        textureResource.isExplicitSpace = ((rand() % 2) == 0) && mShaderTestLayout.allowExplicitSpaces;

        getRegisterIndexAndSpace(mShaderTestLayout.allowExplicitSpaces, textureResource.isExplicitIndex, textureResource.isExplicitSpace, textureResource.isArray, textureResource.isUnbounded, textureResource.arrayDimensions, mTSpacesIndexes, mTSpaces, mMaxTSpace, textureResource.regIndex, textureResource.regSpace);

        texs.push_back(textureResource);
    }


    //  Generate the Textures.
    for (uint32_t i = 0; i < mShaderTestLayout.rwTextures; i++)
    {
        TestShaderWriter::TextureResource textureResource;
        textureResource.textureResourceType = ProgramReflection::Resource::Dimensions::Texture2D;
        textureResource.textureType = ProgramReflection::Variable::Type::Float4;
        textureResource.variable = "rwTextureR" + std::to_string(i);
        textureResource.shaderAccess = ProgramReflection::ShaderAccess::ReadWrite;


        //  The chance of it being used.
        uint32_t useChance = rand() % 2;

        if (useChance)
        {
            textureResource.isUsedResource = true;
        }

        //  Chance for Explicit Index and Spaces.
        textureResource.isExplicitIndex = ((rand() % 2) == 0);
        textureResource.isExplicitSpace = ((rand() % 2) == 0) && mShaderTestLayout.allowExplicitSpaces;

        getRegisterIndexAndSpace(mShaderTestLayout.allowExplicitSpaces, textureResource.isExplicitIndex, textureResource.isExplicitSpace, textureResource.isArray, textureResource.isUnbounded, textureResource.arrayDimensions, mUSpacesIndexes, mUSpaces, mMaxUSpace, textureResource.regIndex, textureResource.regSpace);

        rwTexs.push_back(textureResource);
    }

}


//  Generate the Samplers Resources.
void TestShaderMaker::generateSamplersResources(std::vector<TestShaderWriter::SamplerResource> & samps)
{
    //  Generate the Samplers.
    for (uint32_t i = 0; i < mShaderTestLayout.samplers; i++)
    {
        //  Sampler Resource.
        TestShaderWriter::SamplerResource samplerResource;
        samplerResource.variable = "samplerR" + std::to_string(i);
        samplerResource.isUsedResource = ((rand() % 2) == 0);

        //  Chance for Explicit Index and Spaces.
        samplerResource.isExplicitIndex = ((rand() % 2) == 0);
        samplerResource.isExplicitSpace = ((rand() % 2) == 0) && mShaderTestLayout.allowExplicitSpaces;

        //  
        getRegisterIndexAndSpace(mShaderTestLayout.allowExplicitSpaces, samplerResource.isExplicitIndex, samplerResource.isExplicitSpace, samplerResource.isArray, samplerResource.isUnbounded, samplerResource.arrayDimensions, mSSpacesIndexes, mSSpaces, mMaxSSpace, samplerResource.regIndex, samplerResource.regSpace);

        //  Push back the Sampler Resource.
        samps.push_back(samplerResource);
    }
}



//  Get the Shader Resource Data.
TestShaderWriter::ShaderResourcesData * TestShaderMaker::getShaderStageResourceData(const std::string & stage)
{
    if (stage == "vs")
    {
        return mShaderMaker.getVSResourceData();
    }
    else if (stage == "hs")
    {
        return mShaderMaker.getHSResourceData();
    }
    else if (stage == "ds")
    {
        return mShaderMaker.getDSResourceData();
    }
    else if (stage == "gs")
    {
        return mShaderMaker.getGSResourceData();
    }
    else if (stage == "ps")
    {
        return mShaderMaker.getPSResourceData();
    }
    else if (stage == "cs")
    {
        return mShaderMaker.getCSResourceData();
    }
    else
    {
        return nullptr;
    }


}

//  Get the Register Index And Space.
void TestShaderMaker::getRegisterIndexAndSpace(bool isUsingExplicitSpace, bool isExplicitIndex, bool isExplicitSpace, bool isArray, bool isUnboundedArray, std::vector<uint32_t> arrayDimensions, std::map<uint32_t, RegisterCount> & spaceRegisterIndexes, std::vector<uint32_t> & spacesArray, uint32_t & maxSpace, uint32_t & newIndex, uint32_t & newSpace)
{
    if (isUsingExplicitSpace)
    {
        if (isExplicitIndex)
        {
            uint32_t spaceIndex = rand() % spacesArray.size();

            newIndex = spaceRegisterIndexes[spacesArray[spaceIndex]].nextRegisterIndex;

            if (isArray)
            {
                if (isUnboundedArray)
                {
                    spaceRegisterIndexes[spaceIndex].isMaxed = true;
                    spacesArray[spaceIndex] = mSSpaces[spacesArray.size() - 1];
                    spacesArray.pop_back();
                    spacesArray.push_back(maxSpace);
                    maxSpace++;
                }
                else
                {
                    spaceRegisterIndexes[spacesArray[spaceIndex]].nextRegisterIndex = arrayDimensions[0] + spaceRegisterIndexes[spacesArray[spaceIndex]].nextRegisterIndex + 1;
                }
            }
            else
            {
                spaceRegisterIndexes[spacesArray[spaceIndex]].nextRegisterIndex++;
            }

        }
    }
    else
    {
        if (isExplicitIndex)
        {
            newSpace = 0;
            newIndex = spaceRegisterIndexes[0].nextRegisterIndex;
            spaceRegisterIndexes[0].nextRegisterIndex = spaceRegisterIndexes[0].nextRegisterIndex + 1;
        }
    }

}

//  Generate the Randomized Stages.
std::vector<uint32_t> TestShaderMaker::generateRandomizedStages()
{
    std::vector<uint32_t> selectedStages;
    uint32_t stageCount = 0;

    //  
    if (mShaderTestLayout.stages.size() <= 1)
    {
        stageCount = 1;
    }
    else
    {
        stageCount = (rand() % mShaderTestLayout.stages.size()) + 1;
    }

    //  
    std::vector<uint32_t> allStagesIndices;
    for (uint32_t stageIndex = 0; stageIndex < mShaderTestLayout.stages.size(); stageIndex++)
    {
        allStagesIndices.push_back(stageIndex);
    }

    //  
    for (uint32_t stageIndex = 0; stageIndex < stageCount; stageIndex++)
    {
        uint32_t selectStage = rand() % allStagesIndices.size();
        selectedStages.push_back(allStagesIndices[selectStage]);
        allStagesIndices[selectStage] = allStagesIndices[allStagesIndices.size() - 1];
        allStagesIndices.pop_back();
    }

    //  
    return selectedStages;
}

//  Distribute the Constant Buffer Resources to the Shader Stages.
void TestShaderMaker::distributeConstantBuffers(std::vector<TestShaderWriter::ConstantBufferResource> & mUCBs)
{
    //  Iterate over the Constant Buffer Indices.
    for (uint32_t cbIndex = 0; cbIndex < mUCBs.size(); cbIndex++)
    {
        std::vector<uint32_t> selectedStages = generateRandomizedStages();

        uint32_t declareExplicitStage = 0;

        if (selectedStages.size() <= 1)
        {
            declareExplicitStage = 0;
        }
        else
        {
            declareExplicitStage = (rand() % selectedStages.size());
        }

        for (uint32_t stageIndex = 0; stageIndex < selectedStages.size(); stageIndex++)
        {
            TestShaderWriter::ConstantBufferResource currentCB(mUCBs[cbIndex]);

            if (stageIndex == declareExplicitStage)
            {
                currentCB.isExplicitIndex = true && currentCB.isExplicitIndex;
            }
            else
            {
                currentCB.isExplicitIndex = false;
                currentCB.isExplicitSpace = false;
            }

            //  
            TestShaderWriter::ShaderResourcesData * sRD = getShaderStageResourceData(mShaderTestLayout.stages[selectedStages[stageIndex]]);

            if (sRD)
            {
                sRD->mUCBs.push_back(currentCB);
            }
        }
    }
}


//  Distribute the Structured Buffers.
void TestShaderMaker::distributeStructuredBuffers(std::vector<TestShaderWriter::StructuredBufferResource> & sbs, std::vector<TestShaderWriter::StructuredBufferResource> & rwSBs)
{
    //
    for (uint32_t currentStructuredBufferIndex = 0; currentStructuredBufferIndex < sbs.size(); currentStructuredBufferIndex++)
    {
        std::vector<uint32_t> selectedStages = generateRandomizedStages();

        uint32_t declareExplicitStage = 0;

        if (selectedStages.size() <= 1)
        {
            declareExplicitStage = 0;
        }
        else
        {
            declareExplicitStage = (rand() % selectedStages.size());
        }

        for (uint32_t stageIndex = 0; stageIndex < selectedStages.size(); stageIndex++)
        {
            TestShaderWriter::StructuredBufferResource currentSB(sbs[currentStructuredBufferIndex]);

            if (stageIndex == declareExplicitStage)
            {
                currentSB.isExplicitIndex = true && currentSB.isExplicitIndex;
            }
            else
            {
                currentSB.isExplicitIndex = false;
                currentSB.isExplicitSpace = false;
            }

            //  
            TestShaderWriter::ShaderResourcesData * sRD = getShaderStageResourceData(mShaderTestLayout.stages[selectedStages[stageIndex]]);

            if (sRD)
            {
                sRD->sbs.push_back(currentSB);
            }
        }

    }


    //  Distribute the Read Write Structured Buffers.
    for (uint32_t currentStructuredBufferIndex = 0; currentStructuredBufferIndex < rwSBs.size(); currentStructuredBufferIndex++)
    {
        std::vector<uint32_t> selectedStages = generateRandomizedStages();

        uint32_t declareExplicitStage = 0;

        if (selectedStages.size() <= 1)
        {
            declareExplicitStage = 0;
        }
        else
        {
            declareExplicitStage = (rand() % selectedStages.size());
        }

        for (uint32_t stageIndex = 0; stageIndex < selectedStages.size(); stageIndex++)
        {
            TestShaderWriter::StructuredBufferResource currentSB(rwSBs[currentStructuredBufferIndex]);

            if (stageIndex == declareExplicitStage)
            {
                currentSB.isExplicitIndex = true && currentSB.isExplicitIndex;
            }
            else
            {
                currentSB.isExplicitIndex = false;
                currentSB.isExplicitSpace = false;
            }

            //  
            TestShaderWriter::ShaderResourcesData * sRD = getShaderStageResourceData(mShaderTestLayout.stages[selectedStages[stageIndex]]);

            if (sRD)
            {
                sRD->sbs.push_back(currentSB);
            }
        }


    }


}

//  Distribute the Samplers.
void TestShaderMaker::distributeSamplers(std::vector<TestShaderWriter::SamplerResource> & samplerResources)
{
    //  
    for (uint32_t currentSamplerIndex = 0; currentSamplerIndex < samplerResources.size(); currentSamplerIndex++)
    {
        std::vector<uint32_t> selectedStages = generateRandomizedStages();

        uint32_t declareExplicitStage = 0;

        if (selectedStages.size() <= 1)
        {
            declareExplicitStage = 0;
        }
        else
        {
            declareExplicitStage = (rand() % selectedStages.size());
        }

        for (uint32_t stageIndex = 0; stageIndex < selectedStages.size(); stageIndex++)
        {
            TestShaderWriter::SamplerResource currentSampler(samplerResources[currentSamplerIndex]);

            if (stageIndex == declareExplicitStage)
            {
                currentSampler.isExplicitIndex = true && currentSampler.isExplicitIndex;
            }
            else
            {
                currentSampler.isExplicitIndex = false;
                currentSampler.isExplicitSpace = false;
            }

            //  
            TestShaderWriter::ShaderResourcesData * sRD = getShaderStageResourceData(mShaderTestLayout.stages[selectedStages[stageIndex]]);

            if (sRD)
            {
                sRD->samplers.push_back(currentSampler);

                if (currentSampler.isUsedResource)
                {
                    for (auto & currentTexture : sRD->textures)
                    {
                        if (currentTexture.isUsedResource && currentTexture.samplerVariable == "")
                        {
                            uint32_t samplerChance = rand() % 2;

                            if (samplerChance)
                            {
                                currentTexture.isSampled = true;
                                currentTexture.samplerVariable = currentSampler.variable;
                            }

                        }
                    }
                }
            }

        }
    }
}

//  Distribute the Textures.
void TestShaderMaker::distributeTextures(std::vector<TestShaderWriter::TextureResource> & texs, std::vector<TestShaderWriter::TextureResource> & rwTexs)
{
    for (uint32_t currentTextureIndex = 0; currentTextureIndex < texs.size(); currentTextureIndex++)
    {
        std::vector<uint32_t> selectedStages = generateRandomizedStages();

        uint32_t declareExplicitStage = 0;

        if (selectedStages.size() <= 1)
        {
            declareExplicitStage = 0;
        }
        else
        {
            declareExplicitStage = (rand() % selectedStages.size());
        }

        for (uint32_t stageIndex = 0; stageIndex < selectedStages.size(); stageIndex++)
        {
            TestShaderWriter::TextureResource currentTexture(texs[currentTextureIndex]);

            if (stageIndex == declareExplicitStage)
            {
                currentTexture.isExplicitIndex = true && currentTexture.isExplicitIndex;
            }
            else
            {
                currentTexture.isExplicitIndex = false;
                currentTexture.isExplicitSpace = false;
            }

            //  
            TestShaderWriter::ShaderResourcesData * sRD = getShaderStageResourceData(mShaderTestLayout.stages[selectedStages[stageIndex]]);

            if (sRD)
            {
                sRD->textures.push_back(currentTexture);
            }
        }
    }

    //  Distribute the Read Write Textures.
    for (uint32_t currentTextureIndex = 0; currentTextureIndex < rwTexs.size(); currentTextureIndex++)
    {

        std::vector<uint32_t> selectedStages = generateRandomizedStages();

        uint32_t declareExplicitStage = 0;

        if (selectedStages.size() <= 1)
        {
            declareExplicitStage = 0;
        }
        else
        {
            declareExplicitStage = (rand() % selectedStages.size());
        }

        for (uint32_t stageIndex = 0; stageIndex < selectedStages.size(); stageIndex++)
        {
            TestShaderWriter::TextureResource currentTexture(rwTexs[currentTextureIndex]);

            if (stageIndex == declareExplicitStage)
            {
                currentTexture.isExplicitIndex = true && currentTexture.isExplicitIndex;
            }
            else
            {
                currentTexture.isExplicitIndex = false;
                currentTexture.isExplicitSpace = false;
            }

            //  
            TestShaderWriter::ShaderResourcesData * sRD = getShaderStageResourceData(mShaderTestLayout.stages[selectedStages[stageIndex]]);

            if (sRD)
            {
                sRD->textures.push_back(currentTexture);
            }
        }
    }
}


//  Distribute the Raw Buffers.
void TestShaderMaker::distributeRawBuffers(std::vector<TestShaderWriter::RawBufferResource> & rBs, std::vector<TestShaderWriter::RawBufferResource> & rwRBs)
{
    //  Distribute the Raw Buffer Resources.
    for (uint32_t currentRBIndex = 0; currentRBIndex < rBs.size(); currentRBIndex++)
    {
        std::vector<uint32_t> selectedStages = generateRandomizedStages();

        uint32_t declareExplicitStage = 0;

        if (selectedStages.size() <= 1)
        {
            declareExplicitStage = 0;
        }
        else
        {
            declareExplicitStage = (rand() % selectedStages.size());
        }

        for (uint32_t stageIndex = 0; stageIndex < selectedStages.size(); stageIndex++)
        {
            TestShaderWriter::RawBufferResource currentTexture(rBs[currentRBIndex]);

            if (stageIndex == declareExplicitStage)
            {
                currentTexture.isExplicitIndex = true && currentTexture.isExplicitIndex;
            }
            else
            {
                currentTexture.isExplicitIndex = false;
                currentTexture.isExplicitSpace = false;
            }

            //  
            TestShaderWriter::ShaderResourcesData * sRD = getShaderStageResourceData(mShaderTestLayout.stages[selectedStages[stageIndex]]);

            if (sRD)
            {
                sRD->rbs.push_back(currentTexture);
            }
        }
    }

    //  Distribute the RW Raw Buffer Resources.
    for (uint32_t currentRBIndex = 0; currentRBIndex < rwRBs.size(); currentRBIndex++)
    {

        std::vector<uint32_t> selectedStages = generateRandomizedStages();

        uint32_t declareExplicitStage = 0;

        if (selectedStages.size() <= 1)
        {
            declareExplicitStage = 0;
        }
        else
        {
            declareExplicitStage = (rand() % selectedStages.size());
        }

        for (uint32_t stageIndex = 0; stageIndex < selectedStages.size(); stageIndex++)
        {
            TestShaderWriter::RawBufferResource currentTexture(rwRBs[currentRBIndex]);

            if (stageIndex == declareExplicitStage)
            {
                currentTexture.isExplicitIndex = true && currentTexture.isExplicitIndex;
            }
            else
            {
                currentTexture.isExplicitIndex = false;
                currentTexture.isExplicitSpace = false;
            }

            //  
            TestShaderWriter::ShaderResourcesData * sRD = getShaderStageResourceData(mShaderTestLayout.stages[selectedStages[stageIndex]]);

            if (sRD)
            {
                sRD->rbs.push_back(currentTexture);
            }
        }
    }
}


//  Verify All the Resources.
bool TestShaderMaker::verifyAllResources(const TestShaderWriter::ShaderResourcesData & rsData, ProgramReflection::SharedConstPtr graphicsReflection, std::string & errorMessage)
{
    //  Program Reflection.

    //  Verify Constant Buffer Resources.
    const ProgramReflection::BufferMap& cbBufferMap = graphicsReflection->getBufferMap(ProgramReflection::BufferReflection::Type::Constant);

    for (const auto & firstCurrentCB : cbBufferMap)
    {
        for (const auto & secondCurrentCB : cbBufferMap)
        {
            if (firstCurrentCB.second->getName() != secondCurrentCB.second->getName())
            {
                if (firstCurrentCB.second->getRegisterSpace() == secondCurrentCB.second->getRegisterSpace())
                {
                    //  Need to check for full overlap.
                    bool overlap = firstCurrentCB.second->getRegisterIndex() == secondCurrentCB.second->getRegisterIndex() && secondCurrentCB.second->getShaderAccess() == firstCurrentCB.second->getShaderAccess();

                    if (overlap)
                    {
                        //  Construct the error messsage.
                        errorMessage = "Error : " + firstCurrentCB.second->getName() + " conflicts, with ( Index : " + std::to_string(firstCurrentCB.second->getRegisterIndex()) + ", Space : " + std::to_string(firstCurrentCB.second->getRegisterSpace()) + ")  , ";
                        errorMessage = errorMessage + secondCurrentCB.second->getName() + " ( Index : " + std::to_string(secondCurrentCB.second->getRegisterIndex()) + ", Space : " + std::to_string(secondCurrentCB.second->getRegisterSpace()) + ").";

                        return false;
                    }
                }
            }
        }
    }

    //  
    for (const auto & currentCBR : rsData.mUCBs)
    {
        ProgramReflection::BufferReflection::SharedConstPtr buffRef = graphicsReflection->getBufferDesc(currentCBR.variable, ProgramReflection::BufferReflection::Type::Constant);

        //  
        if (buffRef)
        {
            bool verify = true;
            verify = (buffRef->getName() == currentCBR.variable);

            if (currentCBR.isExplicitIndex)
            {
                verify = (buffRef->getRegisterIndex() == currentCBR.regIndex);
            }

            if (currentCBR.isExplicitSpace)
            {
                verify = (buffRef->getRegisterSpace() == currentCBR.regSpace);
            }

            if (!verify)
            {
                errorMessage = "Constant Buffer : " + currentCBR.variable + " expected (index, space) does not match with reflection (index, space).";
                return false;
            }

            //  
            for (uint32_t i = 0; i < currentCBR.structVariables.size(); i++)
            {
                const ProgramReflection::Variable * cV = buffRef->getVariableData(currentCBR.structVariables[i].structVariable);

                if (!cV)
                {
                    errorMessage = "Constant Buffer : " + currentCBR.variable + " expected to contain variable : " + currentCBR.structVariables[i].structVariable;
                    return false;
                }
            }
        }
        else
        {
            errorMessage = "Constant Buffer : " + currentCBR.variable + " is not found in reflection!";
            return false;
        }
    }

    //  Verify Structured Buffer Resources.
    const ProgramReflection::BufferMap& sbBufferMap = graphicsReflection->getBufferMap(ProgramReflection::BufferReflection::Type::Structured);

    for (const auto & firstCurrentSB : sbBufferMap)
    {
        for (const auto & secondCurrentSB : sbBufferMap)
        {
            if (firstCurrentSB.second->getName() != secondCurrentSB.second->getName())
            {
                if (firstCurrentSB.second->getRegisterSpace() == secondCurrentSB.second->getRegisterSpace())
                {
                    //  Need to check for full overlap.
                    bool overlap = firstCurrentSB.second->getRegisterIndex() == secondCurrentSB.second->getRegisterIndex() && firstCurrentSB.second->getShaderAccess() == secondCurrentSB.second->getShaderAccess();

                    //  
                    if (overlap)
                    {
                        //  Construct the error message.
                        errorMessage = "Error : " + firstCurrentSB.second->getName() + " conflicts, with ( Index : " + std::to_string(firstCurrentSB.second->getRegisterIndex()) + ", Space : " + std::to_string(firstCurrentSB.second->getRegisterSpace()) + ")  , ";
                        errorMessage = errorMessage + secondCurrentSB.second->getName() + " ( Index : " + std::to_string(secondCurrentSB.second->getRegisterIndex()) + ", Space : " + std::to_string(secondCurrentSB.second->getRegisterSpace()) + ").";

                        return false;
                    }
                }
            }
        }
    }


    //  
    for (const auto & currentSBR : rsData.sbs)
    {
        ProgramReflection::BufferReflection::SharedConstPtr buffRef = graphicsReflection->getBufferDesc(currentSBR.variable, ProgramReflection::BufferReflection::Type::Structured);

        //  
        if (buffRef)
        {
            bool verify = true;
            verify = verify && (buffRef->getName() == currentSBR.variable);

            if (currentSBR.isExplicitIndex)
            {
                verify = verify && (buffRef->getRegisterIndex() == currentSBR.regIndex);
            }

            if (currentSBR.isExplicitSpace)
            {
                verify = verify && (buffRef->getRegisterSpace() == currentSBR.regSpace);
            }

            if (!verify)
            {
                errorMessage = "Structured Buffer : " + currentSBR.variable + " expected (index, space) does not match with reflection (index, space).";
                return false;
            }


            if (currentSBR.usesExternalStructVariables)
            {
                // 
                for (uint32_t i = 0; i < currentSBR.externalStructVariables.size(); i++)
                {
                    const ProgramReflection::Variable * cV = buffRef->getVariableData(currentSBR.externalStructVariables[i].structVariable);

                    if (!cV)
                    {
                        errorMessage = "Structured Buffer : " + currentSBR.variable + " expected to contain variable : " + currentSBR.externalStructVariables[i].structVariable;
                        return false;
                    }
                }
            }
        }
        else
        {
            errorMessage = "Structured Buffer : " + currentSBR.variable + " is not found in reflection!";
            return false;
        }
    }


    //  Verify the Raw Buffers
    for (const auto & currentRBs : rsData.rbs)
    {
        bool rbVerify = verifyCommonResourceMatch(currentRBs, ProgramReflection::Resource::ResourceType::RawBuffer, graphicsReflection, errorMessage);

        if (!rbVerify)
        {
            errorMessage = "Error : Failed to verify Texture " + currentRBs.variable;
            return false;
        }
    }


    //  Verify the Samplers.
    for (const auto & currentSampler : rsData.samplers)
    {
        bool sVerify = verifyCommonResourceMatch(currentSampler, ProgramReflection::Resource::ResourceType::Sampler, graphicsReflection, errorMessage);

        if (!sVerify)
        {
            errorMessage = "Error : Failed to verify Sampler " + currentSampler.variable;
            return false;
        }
    }

    //  Verify the Textures.
    for (const auto & currentTexture : rsData.textures)
    {
        bool sVerify = verifyCommonResourceMatch(currentTexture, ProgramReflection::Resource::ResourceType::Texture, graphicsReflection, errorMessage);

        if (!sVerify)
        {
            errorMessage = "Error : Failed to verify Texture " + currentTexture.variable;
            return false;
        }
    }

    return true;
}


//  Verify Common Resource.
bool TestShaderMaker::verifyCommonResourceMatch(const TestShaderWriter::ShaderResource & srData, ProgramReflection::Resource::ResourceType resourceType, ProgramReflection::SharedConstPtr graphicsReflection, std::string & errorMessage)
{
    //  
    const ProgramReflection::Resource * currentResourceRef = graphicsReflection->getResourceDesc(srData.variable);

    bool verify = true;

    if (currentResourceRef)
    {
        verify = verify && (currentResourceRef->type == resourceType);

        if (srData.isExplicitIndex)
        {
            verify = verify && (currentResourceRef->regIndex == srData.regIndex);
        }

        if (srData.isExplicitSpace)
        {
            verify = verify && (currentResourceRef->regSpace == srData.regSpace);
        }

        if (srData.isArray)
        {
            verify = verify && (currentResourceRef->arraySize == srData.arrayDimensions[0]);
        }
    }
    else
    {
        verify = false;
    }

    if (!verify)
    {
        return false;
    }
    else
    {
        return true;
    }

}
