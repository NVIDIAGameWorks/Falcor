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
#include "CudaModule.h"
#include <nvrtc_helper.h>
#include <vector_types.h>

namespace Falcor {
namespace Cuda {
    const CudaModule::CudaPtxTypeDesc cudaPtxTypesList[] = {    {".b8", 1}, {".b16", 2}, {".b32", 4}, {".b64", 8}, 
                                                                {".s8", 1}, {".s16", 2}, {".s32", 4}, {".s64", 8},
                                                                {".u8", 1}, {".u16", 2}, {".u32", 4}, {".u64", 8},
                                                                {".f16", 2}, {".f32", 4}, {".f64", 8}
                                                            };

    CudaModule::CudaKernelArgInfo getKernelArgInfo(std::string typeName, size_t numElements, size_t alignment)
    {
        CudaModule::CudaKernelArgInfo info;
        info.typeString = typeName;
        info.arrayNumElements = numElements;

        const size_t arraySize = arraysize(cudaPtxTypesList);
        for(int i=0; i<arraySize; ++i)
        {
            if( cudaPtxTypesList[i].name == typeName)
            {
                
                info.typeSize = cudaPtxTypesList[i].size;   
                info.totalSize = ((info.typeSize*numElements)/alignment)*alignment;

                return info;
            }
        }

        assert(!"Type not recognised");

        return info;
    }


	CudaModule::CudaModule(const std::string &filename) : mFileName(filename)
    {
		if(!mIncludePathVectorInitOK)
        {
			for (auto& dir : getDataDirectoriesList())
            {
				std::string optionLine = std::string("-I") + dir;
				mIncludePathVectorStr.push_back(optionLine);
			}

			for (auto& dir : mIncludePathVectorStr)
            {
				mIncludePathVector.push_back(dir.c_str());
			}

            mIncludePathVectorInitOK = true;
		}

        recompile();
	}

	void CudaModule::recompile(uint8_t maxRegisterCount)
	{
		std::string fullpath;
		if (Falcor::findFileInDataDirectories(mFileName.c_str(), fullpath) == false)
		{
			Logger::log(Logger::Level::Fatal, std::string("Can't find CUDA file ") + mFileName);
			return;
		}

		compileToPTX((char*)fullpath.c_str(), (int)mIncludePathVector.size(), mIncludePathVector.data(), &mpPtxString, &mPtxStringSize);


		parseBuildKernelsInfo();

		if (maxRegisterCount == 0)
		{
			checkCudaErrors(cuModuleLoadDataEx(&mpModule, mpPtxString, 0, 0, 0));
		}
		else
		{
			CUjit_option options[] = { CU_JIT_MAX_REGISTERS };
			void* optionvals[] = { (void*)maxRegisterCount };

			checkCudaErrors(cuModuleLoadDataEx(&mpModule, mpPtxString, 1, options, optionvals));
		}
	}
    CudaModule::~CudaModule()
    {
	}

	CudaModule::SharedPtr CudaModule::create(const std::string &filename)
    {
		CudaModule::SharedPtr cudaMod(new CudaModule(filename));
		return cudaMod;
	}

    void CudaModule::setGlobalVariableBlob(const std::string &varName, void* buffer, size_t size)
    {
        //Retrieve global constant variable in the kernel module;
        size_t varSize = 0;
        CUdeviceptr vertBuffDevVarPtr = getGlobalVariablePtr(varName, varSize);

        if(vertBuffDevVarPtr == 0ull || varSize == 0)
        {
            Logger::log(Logger::Level::Error, std::string("Can't find kernel global variable") + varName);
            return;
        }

        if(size > varSize)
        {
            Logger::log(Logger::Level::Error, std::string("Kernel global variable is too small to accomodate the host counterpart: ") + varName);
            return;
        }

        // Copy the variable global pointer of the kernel module
        checkFalcorCudaErrors(cuMemcpyHtoD(vertBuffDevVarPtr, &buffer, size));
    }

	void CudaModule::setGlobalTexture(const std::string &varName, const CUtexObject& texObject, const CUaddress_mode_enum addressMode, const CUfilter_mode filterMode)
    {
        CUtexref texRef = getGlobalTexRef(varName);

        // Get texture desc
        CUDA_RESOURCE_DESC texDesc;
        checkFalcorCudaErrors(cuTexObjectGetResourceDesc(&texDesc, texObject));
        if(texDesc.resType == CU_RESOURCE_TYPE_ARRAY)
        {
            checkFalcorCudaErrors(cuTexRefSetArray(texRef, texDesc.res.array.hArray, CU_TRSA_OVERRIDE_FORMAT));
        }
        else if(texDesc.resType == CU_RESOURCE_TYPE_MIPMAPPED_ARRAY)
        {
            checkFalcorCudaErrors(cuTexRefSetMipmappedArray(texRef, texDesc.res.mipmap.hMipmappedArray, CU_TRSA_OVERRIDE_FORMAT));
        }
        else
        {
            CUDA_ARRAY_DESCRIPTOR arrayDescriptor;
            assert(texDesc.resType == CU_RESOURCE_TYPE_PITCH2D);
            arrayDescriptor.Width = texDesc.res.pitch2D.width;
            arrayDescriptor.Height = texDesc.res.pitch2D.height;
            arrayDescriptor.Format = texDesc.res.pitch2D.format;
            arrayDescriptor.NumChannels = texDesc.res.pitch2D.numChannels;

            // Set texture
            checkFalcorCudaErrors(cuTexRefSetAddress2D(texRef, &arrayDescriptor, texDesc.res.pitch2D.devPtr, texDesc.res.pitch2D.pitchInBytes));
        }
        // Set address mode
        checkFalcorCudaErrors(cuTexRefSetAddressMode(texRef, 0, addressMode));
        checkFalcorCudaErrors(cuTexRefSetAddressMode(texRef, 1, addressMode));

        // Set linear filtering by default
		checkFalcorCudaErrors(cuTexRefSetFilterMode(texRef, filterMode));
    }

    void CudaModule::compileToPTX(char *filename, int argc, const char **argv,
		char **ptxResult, size_t *ptxResultSize)
	{
        nvrtcResult res = NVRTC_ERROR_COMPILATION;
        nvrtcProgram prog = nullptr;
        while(res != NVRTC_SUCCESS)
        {
            std::ifstream inputFile(filename, std::ios::in | std::ios::binary |
                std::ios::ate);

            if(!inputFile.is_open())
            {
                Logger::log(Logger::Level::Error, std::string("Can't open CUDA file for reading") + filename);
                continue;
            }

            std::streampos pos = inputFile.tellg();
            size_t inputSize = (size_t)pos;
            char * memBlock = new char[inputSize + 1];

            inputFile.seekg(0, std::ios::beg);
            inputFile.read(memBlock, inputSize);
            inputFile.close();
            memBlock[inputSize] = '\x0';

            // compile
            NVRTC_SAFE_CALL("nvrtcCreateProgram", nvrtcCreateProgram(&prog, memBlock,
                filename, 0, NULL, NULL));
            res = nvrtcCompileProgram(prog, argc, argv);

            // dump log
            size_t logSize;
            NVRTC_SAFE_CALL("nvrtcGetProgramLogSize", nvrtcGetProgramLogSize(prog, &logSize));
            if(logSize > 1)
            {
                // Check if there are errors
                char *log = (char *)malloc(sizeof(char) * logSize + 1);
                NVRTC_SAFE_CALL("nvrtcGetProgramLog", nvrtcGetProgramLog(prog, log));
                log[logSize] = '\x0';
                if(res != NVRTC_SUCCESS)
                {
                    Logger::log(Logger::Level::Error, std::string("Can't compile CUDA kernel ") + filename + "\n" + log);
                }
                else
                {
                    Logger::log(Logger::Level::Warning, std::string("CUDA kernel has warnings ") + filename + "\n" + log);
                }
                free(log);
            }
        }

		NVRTC_SAFE_CALL("nvrtcCompileProgram", res);
		// fetch PTX
		size_t ptxSize;
		NVRTC_SAFE_CALL("nvrtcGetPTXSize", nvrtcGetPTXSize(prog, &ptxSize));
		char *ptx = (char *)malloc(sizeof(char) * ptxSize);
		NVRTC_SAFE_CALL("nvrtcGetPTX", nvrtcGetPTX(prog, ptx));
		NVRTC_SAFE_CALL("nvrtcDestroyProgram", nvrtcDestroyProgram(&prog));
		*ptxResult = ptx;
		*ptxResultSize = ptxSize;
	}

    void CudaModule::parseBuildKernelsInfo()
    {
#ifdef FALCOR_CUDA_RUNTIME_ERROR_CHECKING
        std::string ptxString = std::string(mpPtxString);

        mKernelNameVector.clear();
        mKernelArgsInfoVector.clear();

        size_t kernelOffset = ptxString.find(".entry", 0);
        while (kernelOffset != std::string::npos)
        {

            size_t kernelNameStartOffset = kernelOffset + 7;
            size_t kernelNameEndOffset = ptxString.find("(", kernelOffset);

            std::string kernelName = ptxString.substr(kernelNameStartOffset, kernelNameEndOffset - kernelNameStartOffset);
            mKernelNameVector.push_back(kernelName);
            mKernelArgsInfoVector.push_back(std::vector<CudaKernelArgInfo>());

            size_t kernelParamsEndOffset = ptxString.find(")", kernelNameEndOffset);

            size_t kernelParamOffset = ptxString.find(".param", kernelNameEndOffset);
            while ((kernelParamOffset != std::string::npos) && (kernelParamOffset < kernelParamsEndOffset))
            {
                size_t argDeclarationEndOffset = min(ptxString.find(",", kernelParamOffset + 6), kernelParamsEndOffset);

                ////

                size_t argAlignment = 1;
                size_t argAlignStartOffset = ptxString.find(".align", kernelParamOffset + 6);
                if ((argAlignStartOffset != std::string::npos) && (argAlignStartOffset < argDeclarationEndOffset))
                {
                    argAlignStartOffset += 7;
                    size_t argAlignEndOffset = ptxString.find(" ", argAlignStartOffset);

                    std::string argAlignString = ptxString.substr(argAlignStartOffset, argAlignEndOffset - argAlignStartOffset);
                    argAlignment = stoi(argAlignString);
                }
                else
                {
                    argAlignStartOffset = kernelParamOffset + 6;  //For followup search
                }

                ////

                size_t argTypeStartOffset = ptxString.find(".", argAlignStartOffset);
                size_t argTypeEndOffset = ptxString.find(" ", argTypeStartOffset);
                std::string argTypeString = ptxString.substr(argTypeStartOffset, argTypeEndOffset - argTypeStartOffset);

                ////
                size_t argArraySize = 1;
                size_t argArraySizeStartOffset = ptxString.find("[", argTypeEndOffset);
                if ((argArraySizeStartOffset != std::string::npos) && (argArraySizeStartOffset < argDeclarationEndOffset))
                {
                    size_t argArraySizeEndOffset = ptxString.find("]", argArraySizeStartOffset);

                    std::string argArraySizeString = ptxString.substr(argArraySizeStartOffset + 1, argArraySizeEndOffset - argArraySizeStartOffset - 1);
                    argArraySize = stoi(argArraySizeString);
                }

                CudaKernelArgInfo argInfo = getKernelArgInfo(argTypeString, argArraySize, argAlignment);
                mKernelArgsInfoVector.back().push_back(argInfo);

                kernelParamOffset = ptxString.find(".param", kernelParamOffset + 6);
            }

            kernelOffset = ptxString.find(".entry", kernelOffset + 6);
        }
        //

        //For debug only
        /*for (int k = 0; k < mKernelNameVector.size(); k++){
            std::cout << "\n[" << mKernelNameVector[k] << "]\n";

            std::string typeName;
            for (int a = 0; a < mKernelArgsInfoVector[k].size(); a++){
                std::cout << mKernelArgsInfoVector[k][a].typeString << "  " << mKernelArgsInfoVector[k][a].totalSize << "\n";
            }
        }*/
#endif
    }
}
}