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
#include "SimpleCuda.h"
#include <Cuda/CudaInterop.h>
#include <Cuda/CudaContext.h>

#include <iostream>

#include <nvrtc_helper.h>
#include <vector_types.h>

extern "C"
void initCuda(unsigned int texID);


void SimpleCuda::initUI()
{
    Gui::setGlobalHelpMessage("Sample application to load and display a model.\nUse the UI to switch between wireframe and solid mode.");
}


CUmodule module;
void SimpleCuda::onLoad()
{
    mpCamera = Camera::create();
    mModelViewCameraController.attachCamera(mpCamera);

    initUI();

	loadModelFromFile("ogre/bs_smile.obj");

	mpProgram = Program::createFromFile("", "SimpleCuda.fs");

	Sampler::Desc samplerDesc;
	samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Point);
	mpLinearSampler = Sampler::create(samplerDesc);

	Falcor::ResourceFormat fbFormat[1] = { Falcor::ResourceFormat::RGBA8Unorm };
    mDisplayFbo = FboHelper::create2DWithDepth(mpDefaultFBO->getWidth(), mpDefaultFBO->getHeight(), fbFormat, Falcor::ResourceFormat::D24UnormS8);

	mDisplayTexture = mDisplayFbo->getColorTexture(0);

	////
    uint32_t numBuff = mpModel->getBufferCount();
    uint32_t numMesh = mpModel->getMeshCount();

    for(uint32_t i = 0; i < numMesh; ++i)
    {
        uint32_t numVert = mpModel->getMesh(i)->getVertexCount();

		//Why multiple vertex buffers ? For non interleaved ?
        uint32_t numVertBuff = mpModel->getMesh(i)->getVao()->getVertexBuffersCount();

		VertexBufferInfo vertBuffInfo;
		vertBuffInfo.numVertices = numVert;
		vertBuffInfo.stride = mpModel->getMesh(i)->getVao()->getVertexBufferStride(0);


        uint32_t numVertAttribs = mpModel->getMesh(i)->getVao()->getVertexBufferLayout(0)->getElementCount();
        for(uint32_t a = 0; a < numVertAttribs; a++)
        {
			std::string elemName = mpModel->getMesh(i)->getVao()->getVertexBufferLayout(0)->getElementName(a);
			//std::cout << elemName.c_str() << "\n";

			//POSITION : ??
			//NAME : texCoords
			//TEXCOORD : position, but components are wrong
			if (elemName == "TEXCOORD")
            {

				vertBuffInfo.attribFormat = mpModel->getMesh(i)->getVao()->getVertexBufferLayout(0)->getElementFormat(a);
				vertBuffInfo.attribSize = getFormatBytesPerBlock(vertBuffInfo.attribFormat);
                vertBuffInfo.attribOffset = mpModel->getMesh(i)->getVao()->getVertexBufferLayout(0)->getElementOffset(a);

				//break;
			}
		}

		if (vertBuffInfo.attribSize > 0)
        {
			mVertexBuffers.push_back(mpModel->getMesh(i)->getVao()->getVertexBuffer(0));
			mVertexBuffersInfo.push_back(vertBuffInfo);
		}
	}
	////

	//Init CUDA context
	Cuda::CudaContext::get().init(true);

	//Load CUDA source kernel file as a module ready for execution
	mCudaModule         = Cuda::CudaModule::create("SimpleCudaKernels.cu");
    mCudaModuleVertex   = Cuda::CudaModule::create("SimpleCudaKernelVertex.cu");

}

void SimpleCuda::onFrameRender()
{
	mModelViewCameraController.update();

    uint32_t width = mpDefaultFBO->getWidth();
    uint32_t height = mpDefaultFBO->getHeight();

	const glm::vec4 clearColor(0.35f, 0.75f, 0.35f, 1);
	mpDefaultFBO->clear(clearColor, 1.0f, 0, FboAttachmentType::All);

	///////Transform vertices positions using CUDA///////

	for (int i = 0; i < mVertexBuffers.size(); ++i)
    {
		Buffer::SharedConstPtr buff = mVertexBuffers[i];

		std::shared_ptr<Falcor::Cuda::CudaBuffer> cudaBuff = Cuda::CudaInterop::get().getMappedCudaBuffer(buff);

		/////////////
		uint32_t numVertices    = mVertexBuffersInfo[i].numVertices;
		uint32_t stride			= mVertexBuffersInfo[i].stride;
		size_t attribSize		= mVertexBuffersInfo[i].attribSize;
		uint32_t attribOffset	= mVertexBuffersInfo[i].attribOffset;
		uint8_t* pVertBuff	= (uint8_t *)cudaBuff->getDevicePtr();

		dim3 blockSize(32, 1, 1);
		dim3 gridSize(((uint32_t)numVertices + blockSize.x - 1) / blockSize.x, 1, 1);

		static float scale = 0.9999f;
		static int time = 0;


		//Retrieve global constant variable in the kernel module;
		size_t varSize=0;
		CUdeviceptr vertBuffDevVarPtr = mCudaModuleVertex->getGlobalVariablePtr("d_vertPosBuff", varSize);

		//Copy address of the vertex buffer (in device memory) into the global pointer of the kernel module
		checkFalcorCudaErrors( cuMemcpyHtoD(vertBuffDevVarPtr, &pVertBuff, varSize) );

		//Launch the kernel
		mCudaModuleVertex->launchKernel("kernelProcessVertices", blockSize, gridSize, 
						scale, numVertices, stride, attribSize, attribOffset);

		time++;
		if (time == 10000){
			time = 0;
			scale = 1.0f / scale; //Yes, this will not end well :)
		}
	}

	mpRenderContext->pushFbo( mDisplayFbo );
	{
		const glm::vec4 clearColor(0.35f, 0.35f, 0.75f, 1);
		mDisplayFbo->clear(clearColor, 1.0f, 0, FboAttachmentType::All);
		//m_displayFramebuffer->GetFbo()->clear(clearColor, 1.0f, 0, FboTarget::Depth);

		//Works if the depth buffer is copied from default one
		//m_pRenderContext->BlitFbo(m_pDefaultFBO, m_displayFramebuffer->GetFbo(),
		//	glm::ivec4(0, 0, width, height), glm::ivec4(0, 0, width, height), false, FboTarget::Depth);

		
        mpCamera->setDepthRange(mNearZ, mFarZ);

		// Set render state
		RasterizerState::Desc solidDesc;
		solidDesc.setCullMode(RasterizerState::CullMode::None);
		mpRenderContext->setRasterizerState( RasterizerState::create(solidDesc) );

		DepthStencilState::Desc desc;
		desc.setDepthTest(true);
		desc.setDepthFunc(DepthStencilState::Func::LessEqual);
        mpRenderContext->setDepthStencilState(DepthStencilState::create(desc), 0);

		//m_DirLight.setIntoUniformBuffer(m_pPerFrameCB, "gDirLight");
		//m_PointLight.setIntoUniformBuffer(m_pPerFrameCB, "gPointLight");
		//m_pModel->GetMesh(0)->getVao()->GetVertexBuffer(0)->

		mpModel->bindSamplerToMaterials(mpLinearSampler); //m_pPointSampler

        mModelRenderer.render(mpRenderContext.get(), mpProgram.get(), mpModel, mpCamera.get());

	}
	mpRenderContext->popFbo();


	//Post-process framebuffer texture using CUDA//
	dim3 blockSize(16, 16, 1);
	dim3 gridSize(((uint32_t)width + blockSize.x - 1) / blockSize.x, ((uint32_t)height + blockSize.y - 1) / blockSize.y, 1);
	
	//Map texture in CUDA
	std::shared_ptr<Falcor::Cuda::CudaTexture> cudaTex = Cuda::CudaInterop::get().getMappedCudaTexture(mDisplayTexture);

	mCudaModule->launchKernel("kernelProcessTexture", blockSize, gridSize, 
						(cudaTextureObject_t&)cudaTex->getTextureObject(),
						(cudaSurfaceObject_t&)cudaTex->getSurfaceObject(),
						width, height, 
                        glm::u8vec3(0,0,128)
                        //glm::u8vec4(0,0,128, 0)   //Test runtime error checking
                        //,42                       //Test runtime error checking
                        );

	////////////////////////////////////////


	mpRenderContext->blitFbo(mDisplayFbo.get(), mpDefaultFBO.get(),
		glm::ivec4(0, 0, width, height),
		glm::ivec4(0, 0, width, height));



	renderText(getGlobalSampleMessage(true), glm::vec2(10, 10));

}

void SimpleCuda::onShutdown()
{

}

bool SimpleCuda::onKeyEvent(const KeyboardEvent& keyEvent)
{
    bool bHandled = mModelViewCameraController.onKeyEvent(keyEvent);
	if (bHandled == false)
	{
        if(keyEvent.type == KeyboardEvent::Type::KeyPressed)
		{
			switch (keyEvent.key)
			{
			case KeyboardEvent::Key::R:
				resetCamera();
				break;
			default:
				bHandled = false;
			}
		}
	}
	return bHandled;
}

bool SimpleCuda::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mModelViewCameraController.onMouseEvent(mouseEvent);
}

void SimpleCuda::onDataReload()
{

}

void SimpleCuda::onResizeSwapChain()
{
    RenderContext::Viewport vp;
    vp.height = (float)mpDefaultFBO->getHeight();
    vp.width = (float)mpDefaultFBO->getWidth();
    mpRenderContext->setViewport(0, vp);
    mpCamera->setFovY(float(M_PI / 8));
    mpCamera->setAspectRatio(vp.width / vp.height);
    mpCamera->setDepthRange(0, 1000);

	// create a lower-res default FBO in case if we need to scale the results
    if((mDisplayFbo == nullptr) || (mDisplayFbo->getWidth() != mpDefaultFBO->getWidth()) || (mDisplayFbo->getHeight() != mpDefaultFBO->getHeight()))
	{
        mDisplayFbo = nullptr;
		Falcor::ResourceFormat fbFormat[1] = { Falcor::ResourceFormat::RGBA8Unorm };
        mDisplayFbo = FboHelper::create2DWithDepth(mpDefaultFBO->getWidth(), mpDefaultFBO->getHeight(), fbFormat, Falcor::ResourceFormat::D24UnormS8);
	}
}

void SimpleCuda::loadModelFromFile(const std::string& Filename)
{
	CpuTimer timer;
    timer.update();

	uint32_t flags = Model::CompressTextures;
	//Flags |= Model::GenerateTangentSpace;

	mpModel = Model::createFromFile(Filename, flags);

    if(mpModel == nullptr)
	{
		msgBox("Could not load model");
		return;
	}
	//SetModelUIElements();
	resetCamera();

	/*float Radius = m_pModel->getRadius();
	m_PointLight.setWorldPosition(glm::vec3(0, Radius*1.25f, 0));*/

    timer.update();

	//SetModelString(false, Timer.GetElapsedTime());
}


void SimpleCuda::resetCamera()
{
	if (mpModel)
	{
		// update the camera position
        float radius = mpModel->getRadius();
        const glm::vec3& modelCenter = mpModel->getCenter();
		glm::vec3 camPos = modelCenter;
		camPos.z += radius * 2;

        mpCamera->setPosition(camPos);
        mpCamera->setTarget(modelCenter);
        mpCamera->setUpVector(glm::vec3(0, 1, 0));

		// Update the controllers
		mModelViewCameraController.setModelParams(modelCenter, radius, 10);
		/*m_pFirstPersonCameraController->SetCameraSpeed(Radius*0.25f);
		m_p6DoFCameraController->SetCameraSpeed(Radius*0.25f);
		*/

		mNearZ = std::max(0.1f, mpModel->getRadius() / 750.0f);
		mFarZ = radius * 10;
		
	}
}

//int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
int main(char** argv, int argc)
{
    SimpleCuda simpleCuda;
    SampleConfig config;
	config.windowDesc.swapChainDesc.width = 1280;
	config.windowDesc.swapChainDesc.height = 720;
	
	config.windowDesc.title = "Falcor Simple Cuda";
    simpleCuda.run(config);
}
