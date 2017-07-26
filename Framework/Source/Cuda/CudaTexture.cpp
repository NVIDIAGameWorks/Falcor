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
#include "CudaTexture.h"
#include "CudaContext.h"

namespace Falcor {
namespace Cuda {

	CudaTexture::CudaTexture(CUgraphicsResource &cuGraphicsResource) 
    {
		mpCUGraphicsResource = cuGraphicsResource;

        // Get the first mip for surface writes
        checkFalcorCudaErrors(cuGraphicsSubResourceGetMappedArray(&mpCUArray, cuGraphicsResource, 0, 0));
        memset(&mCUSurfRessource, 0, sizeof(CUDA_RESOURCE_DESC));
        mCUSurfRessource.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
        mCUSurfRessource.res.array.hArray = mpCUArray;

        // Get the whole array for mip-mapped texture filtering
		checkFalcorCudaErrors(cuGraphicsResourceGetMappedMipmappedArray(&mpCUMipArray, cuGraphicsResource));
        memset(&mCUTexRessource, 0, sizeof(CUDA_RESOURCE_DESC));
        mCUTexRessource.resType = CUresourcetype::CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
        mCUTexRessource.res.mipmap.hMipmappedArray = mpCUMipArray;
	}

	CudaTexture::~CudaTexture()
    {
		if (mpCUGraphicsResource != nullptr){
            checkFalcorCudaErrors(cuGraphicsUnmapResources(1, &mpCUGraphicsResource, 0));

            for(auto& it : mCUTexObjMap)
				checkFalcorCudaErrors(cuTexObjectDestroy( it.second ));

			if (mIsCUSurfObjCreated)
				checkFalcorCudaErrors(cuSurfObjectDestroy(mCUSurfObj));
		}
	}

	CudaTexture::SharedPtr CudaTexture::create(CUgraphicsResource &cuGraphicsResource)
    {
		CudaTexture::SharedPtr cudaTex(new CudaTexture(cuGraphicsResource));
		return cudaTex;
	}

	CUsurfObject &CudaTexture::getSurfaceObject()
    {
		if (!mIsCUSurfObjCreated)
        {
			checkFalcorCudaErrors(cuSurfObjectCreate(&mCUSurfObj, &mCUSurfRessource));
			mIsCUSurfObjCreated = true;
		}

		return mCUSurfObj;
	}

	CUtexObject &CudaTexture::getTextureObject(bool accessUseNormalizedCoords, bool linearFiltering, Sampler::AddressMode addressMode)
    {
		uint32_t hashVal = hashCUTexObject(accessUseNormalizedCoords, linearFiltering, addressMode);
		
		if ( mCUTexObjMap.find(hashVal) == mCUTexObjMap.end() )
        {
			CUDA_TEXTURE_DESC		cu_texDescr;
			memset(&cu_texDescr, 0, sizeof(CUDA_TEXTURE_DESC));

			cu_texDescr.flags = accessUseNormalizedCoords ? CU_TRSF_NORMALIZED_COORDINATES : 0;

            cu_texDescr.filterMode = linearFiltering ? CU_TR_FILTER_MODE_LINEAR : CU_TR_FILTER_MODE_POINT;  //cudaFilterModeLinear
            cu_texDescr.mipmapFilterMode = linearFiltering ? CU_TR_FILTER_MODE_LINEAR : CU_TR_FILTER_MODE_POINT;  //cudaFilterModeLinear

			CUaddress_mode addressmode = CU_TR_ADDRESS_MODE_CLAMP;
			switch (addressMode){
			case Sampler::AddressMode::Clamp:
				addressmode = CU_TR_ADDRESS_MODE_CLAMP;
				break;
			case Sampler::AddressMode::Wrap:
				addressmode = CU_TR_ADDRESS_MODE_WRAP;
				break;
			case Sampler::AddressMode::Mirror:
				addressmode = CU_TR_ADDRESS_MODE_MIRROR;
				break;
			case Sampler::AddressMode::Border:
				addressmode = CU_TR_ADDRESS_MODE_BORDER;
				break;

			default:
				assert(!"MirrorClampToEdge address mode unsupported !");
				break;
			}

			cu_texDescr.addressMode[0] = addressmode;
			cu_texDescr.addressMode[1] = addressmode;
			cu_texDescr.addressMode[2] = addressmode;

            cu_texDescr.mipmapLevelBias = 0;
            cu_texDescr.minMipmapLevelClamp = 0;
            cu_texDescr.maxMipmapLevelClamp = 100;

			CUtexObject	cuTexObj;
			checkFalcorCudaErrors(cuTexObjectCreate(&cuTexObj, &mCUTexRessource, &cu_texDescr, NULL));

			mCUTexObjMap[hashVal] = cuTexObj;
		}

		return mCUTexObjMap[hashVal];
	}
}
}