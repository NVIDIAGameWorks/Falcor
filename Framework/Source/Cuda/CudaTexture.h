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

#include <cuda.h>
#include <cudaGL.h> //Driver API

#include <unordered_map>

namespace Falcor {
namespace Cuda {


	/** Texture abstraction in CUDA. Encapsulate a Cuda Array.
		Can be created either from a registered OpenGL texture, or from scratch.
		Uses the CUDA driver API.
	*/
	class CudaTexture : public std::enable_shared_from_this<CudaTexture>
	{
    public:
        using SharedPtr = std::shared_ptr<CudaTexture>;
        using SharedConstPtr = std::shared_ptr<const CudaTexture>;

	private:
		CudaTexture()
        {   }

		/** Private constructor from a registered OpenGL texture.
		*/
		CudaTexture(CUgraphicsResource &cuGraphicsResource);

	protected:
		CUgraphicsResource		mpCUGraphicsResource	= nullptr;
		CUarray		            mpCUArray				= nullptr;
		CUmipmappedArray		mpCUMipArray			= nullptr;

		CUDA_RESOURCE_DESC		mCUSurfRessource, mCUTexRessource;

		std::unordered_map< uint32_t, CUtexObject >		mCUTexObjMap;

		bool					mIsCUSurfObjCreated = false;
		CUsurfObject			mCUSurfObj;
		
		
		uint32_t hashCUTexObject(bool accessUseNormalizedCoords, bool linearFiltering, Sampler::AddressMode addressMode)
        {
			//TODO: make sure that AddressMode bits will never overlap the to MSBs...
			return (((uint32_t)accessUseNormalizedCoords) << 31U) | (((uint32_t)linearFiltering) << 30U) | ((uint32_t)addressMode);
		}
	public:
		~CudaTexture();

		/** Construct a new CudaTexture from a registered OpenGL texture.
		*/
		static CudaTexture::SharedPtr create(CUgraphicsResource &cuGraphicsResource);

		/** create a CUDA surface object for the texture (bindless surface) if necessary, and returns it.
		*/
		CUsurfObject &getSurfaceObject();

		/** create a CUDA texture object for the texture (bindless texture) if necessary, and returns it.
			CUDA texture objects encapsulate both the texture and the sampler to use with it.
			Sampler is created based on provided parameters.
		*/
		CUtexObject &getTextureObject(bool accessUseNormalizedCoords = true, bool linearFiltering = false, Sampler::AddressMode addressMode = Sampler::AddressMode::Clamp);
	};
}
}