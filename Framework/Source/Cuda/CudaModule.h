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

#include <Falcor.h>

#include "CudaContext.h"


#include <string>
#include <vector>
#include <algorithm>
#include <cuda.h>
#include <cudaGL.h> //Driver API
#include <vector_types.h>
#include <helper_cuda_drvapi.h>

#ifndef NDEBUG
# define FALCOR_CUDA_RUNTIME_ERROR_CHECKING     //New experimental error checking
#endif

namespace Falcor {
namespace Cuda {


	/** Kernel abstraction for CUDA. 
		Encapsulates kernel loading at runtime and execution.
		Uses the CUDA driver API.
	*/
	class CudaModule : public std::enable_shared_from_this<CudaModule>
	{
    public:
        using SharedPtr = std::shared_ptr<CudaModule>;
        using SharedConstPtr = std::shared_ptr<const CudaModule>;

        struct CudaPtxTypeDesc {
            const std::string name;
            size_t size;
        };

		~CudaModule();

		/** Construct a new CudaModule from a filename.
		*/
		static CudaModule::SharedPtr create(const std::string &filename);

		/** Launch a CUDA kernel contained in the module.
		*/
		template<typename ... Types>
		bool launchKernel(const std::string &kernelName, dim3 blockSize, dim3 gridSize, const Types& ... params)
        {
#ifdef FALCOR_CUDA_RUNTIME_ERROR_CHECKING
            //Prepare runtime error checking
            std::vector<std::string>::iterator it = std::find( mKernelNameVector.begin(), mKernelNameVector.end(), kernelName );

            if( it==mKernelNameVector.end() )
                Logger::log(Logger::Level::Error, "Kernel not present in Cuda module");
            assert( it!=mKernelNameVector.end() );      //TODO: Fused assert + Logger::log function

            mRuntimeCurrentKernel = std::distance(mKernelNameVector.begin(), it);
#endif

			//Build arguments list
			mArgsVector.clear();
			buildArgsList(mArgsVector, params...);
			void **arr = (void **)mArgsVector.data();

			CUfunction kernel_addr;
			checkFalcorCudaErrors(cuModuleGetFunction(&kernel_addr, mpModule, kernelName.c_str()));

			checkFalcorCudaErrors(cuLaunchKernel(kernel_addr,
						gridSize.x, gridSize.y, gridSize.z,			/* grid dim */
						blockSize.x, blockSize.y, blockSize.z,		/* block dim */
						0, 0,										/* shared mem, stream */
						arr,										/* arguments */
						0));

#ifdef FALCOR_CUDA_RUNTIME_ERROR_CHECKING
			checkFalcorCudaErrors(cuCtxSynchronize());
#endif

			return true;
		}

		/** Return the CUDA module structure.
		*/
		CUmodule &get()    { return mpModule; }

		/** Return a (GPU) pointer to a global __constant__ or __device__ variable declared in the module.
		*/
		CUdeviceptr getGlobalVariablePtr(const std::string &varName, size_t &varSize)
        {
			CUdeviceptr ptr = 0;
			checkFalcorCudaErrors( cuModuleGetGlobal(&ptr, &varSize, mpModule, varName.c_str()) );
			return ptr;
		}
        
		/** Return a (GPU) pointer to a global texture.
		*/
		CUtexref getGlobalTexRef(const std::string &varName)
        {
			CUtexref texRef = nullptr;
			checkFalcorCudaErrors( cuModuleGetTexRef(&texRef, mpModule, varName.c_str()) );
			return texRef;
		}
        
		/** Shortcuts for setting variables.
		*/
        void setGlobalVariableBlob(const std::string &varName, void* buffer, size_t size);

        template<typename T>
        void setGlobalVariable(const std::string &varName, const T& value)
        {
            setGlobalVariableBlob(varName, &value, sizeof(value));
        }

        template<>
        void setGlobalVariable<CUdeviceptr>(const std::string &varName, const CUdeviceptr& value)
        {
            setGlobalVariableBlob(varName, (void*)value, sizeof(value));
        }

		void setGlobalTexture(const std::string &varName, const CUtexObject& value, const CUaddress_mode_enum addressMode = CU_TR_ADDRESS_MODE_CLAMP, const CUfilter_mode filterMode = CU_TR_FILTER_MODE_LINEAR);

        void recompile(uint8_t maxRegisterCount = 0);


        char* getPtxString()
        {
            return mpPtxString;
        }

        struct CudaKernelArgInfo {
            std::string         typeString;
            size_t              typeSize;
            size_t              arrayNumElements;
            size_t              totalSize;
        };

    protected:
        CudaModule() { }

        CudaModule(const std::string &filename);

        static void compileToPTX(char *filename, int argc, const char **argv,
            char **ptxResult, size_t *ptxResultSize);

        //Parse PTX for building kernel info vectors used for runtime error checking
        void parseBuildKernelsInfo();

        //Build list of arguments using variadic parameters and recursion
        void buildArgsList(std::vector<void*> &argVector) { }

        template<typename T, typename ... Types>
        void buildArgsList(std::vector<void*> &argVector, const T& first, const Types& ... rest)
        {
#ifdef FALCOR_CUDA_RUNTIME_ERROR_CHECKING
            //Error checking
            size_t curArgument = argVector.size();
            if( curArgument >= mKernelArgsInfoVector[mRuntimeCurrentKernel].size() )
                Logger::log(Logger::Level::Error, "Too many arguments passed to CUDA kernel");
            assert( curArgument < mKernelArgsInfoVector[mRuntimeCurrentKernel].size() );

            if( sizeof(T) != mKernelArgsInfoVector[mRuntimeCurrentKernel][curArgument].totalSize )
                Logger::log(Logger::Level::Error, "Wrong argument: Size of launchKernel argument doesn't match kernel code");
            assert( sizeof(T) == mKernelArgsInfoVector[mRuntimeCurrentKernel][curArgument].totalSize );
#endif
            //Recursion
            argVector.push_back((void *)&(first));
            buildArgsList(argVector, rest...);
        }

        std::string					        mFileName;
        CUmodule					        mpModule = nullptr;

        std::vector<void*>		            mArgsVector;
#ifdef FALCOR_CUDA_RUNTIME_ERROR_CHECKING
        std::vector< std::string >		                    mKernelNameVector;
        std::vector< std::vector<CudaKernelArgInfo> >	    mKernelArgsInfoVector;
        size_t                              mRuntimeCurrentKernel = 0;
#endif
        bool						        mIncludePathVectorInitOK = false;
        std::vector<std::string>	        mIncludePathVectorStr;
        std::vector<const char*>	        mIncludePathVector;

        char*                               mpPtxString;
        size_t                              mPtxStringSize;


       
	};

    extern const CudaModule::CudaPtxTypeDesc cudaPtxTypesList[];


    CudaModule::CudaKernelArgInfo getKernelArgInfo(std::string typeName, size_t numElements);

}
}