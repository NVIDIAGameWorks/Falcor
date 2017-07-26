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
#include "CudaContext.h"
#include <helper_cuda_drvapi.h>

namespace Falcor {
    //TODO: find header where this is declared
    bool getEnvironemntVariable(const std::string& varName, std::string& value);
    void addDataDirectory(const std::string& dataDir);
    const std::string& getExecutableDirectory();

namespace Cuda {

    void CudaContext::__checkFalcorCudaErrors(CUresult err, const char *file, const int line) 
    {
        if (CUDA_SUCCESS != err)
        {
            std::string str = std::string("checkFalcorCudaErrors() Driver API error = ") + std::to_string((int)err) + std::string(" \"")
                + std::string(getCudaDrvErrorString(err))
                + std::string("\" from file <") + std::string(file) + std::string(">, line ") + std::to_string(line) + std::string(".\n");
            Falcor::Logger::log(Falcor::Logger::Level::Error, str);

            exit(EXIT_FAILURE);
        }
    }

    Falcor::Cuda::CudaContext& CudaContext::get()
    {
        static CudaContext instance;
        return instance;
    }

    void CudaContext::init(bool useGLDevice, int preferedDevice)
    {
        if(mpCudaContext)
            return;

		const int kMaxDevices = 32;
			
		assert(preferedDevice<kMaxDevices);
		assert(mpCudaContext == nullptr);

		CUdevice device[kMaxDevices];
		checkFalcorCudaErrors( cuInit(0) );
	    
        //CUdevice cuDevice = findCudaDeviceDRV(argc, (const char **)argv);

		if(useGLDevice){
			unsigned int cudaDeviceCount;
			checkFalcorCudaErrors( cuGLGetDevices(&cudaDeviceCount, device, kMaxDevices, CU_GL_DEVICE_LIST_ALL) );

            mpCudaDevice = device[preferedDevice];
		}else{
			cuDeviceGet(device, preferedDevice);
			
            mpCudaDevice = device[0];
		}

        // get compute capabilities and the device name
        int major = 0, minor = 0;
        char deviceName[256];
        checkCudaErrors(cuDeviceComputeCapability(&major, &minor, mpCudaDevice));
        checkCudaErrors(cuDeviceGetName(deviceName, 256, mpCudaDevice));
        Logger::log(Logger::Level::Info, "GPU Device has SM %d.%d compute capability:" + std::to_string(major) + "." +std::to_string(minor));

        // Retain an existing context
        checkCudaErrors(cuDevicePrimaryCtxRetain(&mpCudaContext, mpCudaDevice));
        CUcontext context = 0;
        checkCudaErrors(cuCtxGetCurrent(&context));
        if(context != mpCudaContext)
        {
            checkCudaErrors(cuCtxSetCurrent(mpCudaContext));
        }
		//Add CUDA possible header locations paths//
		const std::string cudaPathEnvVar = "CUDA_PATH_V7_5";
        std::string cudaPath;
		if (Falcor::getEnvironemntVariable(cudaPathEnvVar, cudaPath)) {
			Falcor::addDataDirectory(cudaPath + std::string("/include/"));
        }

		//Falcor::AddDataDirectory(std::string(std::getenv("FALCOR_HOME")) + std::string("/Externals/Cuda/include/"));

		///Externals/Cuda/include/
		Falcor::addDataDirectory(std::string("./Data/include/"));
		Falcor::addDataDirectory(std::string(getExecutableDirectory()) + std::string("/Data/include/"));
	}
}
}