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

#include "CudaInterop.h"
#include "CudaContext.h"

#include <helper_cuda_drvapi.h>

namespace Falcor {
namespace Cuda {
    Falcor::Cuda::CudaInterop& CudaInterop::get()
    {
        static CudaInterop instance;
        return instance;
    }

    std::shared_ptr<Cuda::CudaTexture> CudaInterop::getMappedCudaTexture(const Falcor::Texture::SharedConstPtr& tex)
    {
		if (mTextureMap.find((size_t)tex.get()) == mTextureMap.end())
        {
			CuInteropMapVal<Cuda::CudaTexture> interopVal;
			GLenum glSizedFormat = getGlSizedFormat(tex->getFormat());
			GLenum glTarget = getGlTextureTarget((int)(tex->getType()));

			checkFalcorCudaErrors(
				cuGraphicsGLRegisterImage(&(interopVal.cudaGraphicsResource), tex->getApiHandle(), glTarget, CU_GRAPHICS_REGISTER_FLAGS_NONE));  //

			mTextureMap[(size_t)tex.get()] = interopVal;
		}

        auto& res = mTextureMap[(size_t)tex.get()];
        std::shared_ptr<Cuda::CudaTexture> resource; 
		if (res.pFalcorCudaGraphicsResource.expired())
        {
            checkFalcorCudaErrors(cuGraphicsMapResources(1, &res.cudaGraphicsResource, 0));
            resource = Cuda::CudaTexture::create(res.cudaGraphicsResource);
			res.pFalcorCudaGraphicsResource = resource;
		}
        else
        {
            resource = res.pFalcorCudaGraphicsResource.lock();
        }

		return resource;
	}

    void CudaInterop::getMappedCudaTextures(std::initializer_list<std::pair<Falcor::Texture::SharedConstPtr, Cuda::CudaTexture::SharedPtr*>> texturePairs)
    {
        std::vector<std::pair<Falcor::Texture::SharedConstPtr, Cuda::CudaTexture::SharedPtr*>> texToMap;
        std::vector<CUgraphicsResource> resourcesToMap;
        texToMap.reserve(texturePairs.size());

        // Register all new textures, collect all unmapped
        for (auto& texPair : texturePairs)
        {
            if(mTextureMap.find((size_t)texPair.first.get()) == mTextureMap.end())
            {
                CuInteropMapVal<Cuda::CudaTexture> interopVal;
                GLenum glSizedFormat = getGlSizedFormat(texPair.first->getFormat());
                GLenum glTarget = getGlTextureTarget((int)(texPair.first->getType()));

                checkFalcorCudaErrors(
                    cuGraphicsGLRegisterImage(&interopVal.cudaGraphicsResource, texPair.first->getApiHandle(), glTarget, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST));  //

                mTextureMap[(size_t)texPair.first.get()] = interopVal;
            }

            auto& resource = mTextureMap[(size_t)texPair.first.get()];
            if(resource.pFalcorCudaGraphicsResource.expired())
            {
                texToMap.push_back(texPair);
                resourcesToMap.push_back(resource.cudaGraphicsResource);
            }
            else
            {
                *texPair.second = resource.pFalcorCudaGraphicsResource.lock();
            }
        }

        // Map all unmapped textures
        if(!resourcesToMap.empty())
        {
            checkFalcorCudaErrors(cuGraphicsMapResources((uint32_t)resourcesToMap.size(), resourcesToMap.data(), 0));
        }

        // Allocate resulting textures, assign the mapped resources
        for(size_t i=0;i<texToMap.size();++i)
        {
            auto& texPair = texToMap[i];
            auto& mappedResource = resourcesToMap[i];
            auto& resource = mTextureMap[(size_t)texPair.first.get()];
            resource.cudaGraphicsResource = mappedResource;
            assert(resource.pFalcorCudaGraphicsResource.expired());
            if(resource.pFalcorCudaGraphicsResource.expired())
            {
                *texPair.second = Cuda::CudaTexture::create(resource.cudaGraphicsResource);
                resource.pFalcorCudaGraphicsResource = *texPair.second;
            }
        }
    }

    void CudaInterop::unregisterCudaTexture(const Falcor::Texture* tex)
    {
        if(mTextureMap.empty())
            return;

        size_t ptr = (size_t)tex;
        auto it = mTextureMap.find(ptr);
        if(it == mTextureMap.end())
        {
            return;
        }

        if(!it->second.pFalcorCudaGraphicsResource.expired())
        {
            Logger::log(Logger::Level::Error, "The resource to be unregistered is still mapped.");
        }

        checkFalcorCudaErrors(cuGraphicsUnregisterResource(it->second.cudaGraphicsResource));
        mTextureMap.erase(it);
    }

    std::shared_ptr<Cuda::CudaBuffer> CudaInterop::getMappedCudaBuffer(const Falcor::Buffer::SharedConstPtr& buff)
    {
		if (mBufferMap.find((size_t)buff.get()) == mBufferMap.end())
        {
			CuInteropMapVal<Cuda::CudaBuffer> interopVal;
			checkFalcorCudaErrors(cuGraphicsGLRegisterBuffer(&(interopVal.cudaGraphicsResource), buff->getApiHandle(), CU_GRAPHICS_REGISTER_FLAGS_NONE));
            mBufferMap.insert(std::make_pair((size_t)buff.get(), interopVal));
        }

        auto& res = mBufferMap[(size_t)buff.get()];
        Cuda::CudaBuffer::SharedPtr resource;
		if (res.pFalcorCudaGraphicsResource.expired()) 
        {
            resource = Cuda::CudaBuffer::create(res.cudaGraphicsResource);
			res.pFalcorCudaGraphicsResource = resource;
		}
        else
        {
            resource = res.pFalcorCudaGraphicsResource.lock();
        }

		return resource;
	}

    void CudaInterop::unregisterCudaBuffer(const Falcor::Buffer* buff)
    {
        if(mBufferMap.empty())
            return;

        size_t ptr = (size_t)buff;
        auto it = mBufferMap.find(ptr);
        if(it == mBufferMap.end())
        {
            return;
        }

        if(!it->second.pFalcorCudaGraphicsResource.expired())
        {
            Logger::log(Logger::Level::Error, "The resource to be unregistered is still mapped.");
        }

        checkFalcorCudaErrors(cuGraphicsUnregisterResource(it->second.cudaGraphicsResource));
        mBufferMap.erase(it);
    }

    void CudaInterop::uninit()
    {
        for(auto& it : mTextureMap)
            cuGraphicsUnregisterResource(it.second.cudaGraphicsResource);
        mTextureMap.clear();

        for(auto& it : mBufferMap)
            cuGraphicsUnregisterResource(it.second.cudaGraphicsResource);
        mBufferMap.clear();
    }
}
}
