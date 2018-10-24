# Controls what config to build samples with. Valid values are "Debug" and "Release"
SAMPLE_CONFIG:=Release

All : ForwardRenderer AllCore AllEffects AllUtils
AllCore : ComputeShader MultiPassPostProcess ShaderToy SimpleDeferred StereoRendering
AllEffects : AmbientOcclusion SkyBoxRenderer HashedAlpha HDRToneMapping Shadows
AllUtils : ModelViewer SceneEditor

# A sample demonstrating Falcor's effects library
ForwardRenderer : $(SAMPLE_CONFIG)
	$(eval DIR=Samples/ForwardRenderer/)
	@$(CC) $(CXXFLAGS) $(DIR)ForwardRenderer.cpp -o $(DIR)ForwardRenderer.o
	@$(CC) $(CXXFLAGS) $(DIR)ForwardRendererControls.cpp -o $(DIR)ForwardRendererControls.o
	@$(CC) $(CXXFLAGS) $(DIR)ForwardRendererSceneRenderer.cpp -o $(DIR)ForwardRendererSceneRenderer.o
	@$(CC) -o $(OUT_DIR)ForwardRenderer $(DIR)ForwardRenderer.o $(DIR)ForwardRendererControls.o $(DIR)ForwardRendererSceneRenderer.o $(ADDITIONAL_LIB_DIRS) $(LIBS) $(RELATIVE_RPATH)
	$(call MoveFalcorData,$(OUT_DIR))
	$(call MoveProjectData,$(DIR), $(OUT_DIR))
	@echo Built $@

# Core Samples

ComputeShader : $(SAMPLE_CONFIG)
	$(call CompileSample,Samples/Core/ComputeShader/,ComputeShader.cpp,ComputeShader)

MultiPassPostProcess : $(SAMPLE_CONFIG)
	$(call CompileSample,Samples/Core/MultiPassPostProcess/,MultiPassPostProcess.cpp,MultiPassPostProcess)

ShaderToy : $(SAMPLE_CONFIG)
	$(call CompileSample,Samples/Core/ShaderToy/,ShaderToy.cpp,ShaderToy)

SimpleDeferred : $(SAMPLE_CONFIG)
	$(call CompileSample,Samples/Core/SimpleDeferred/,SimpleDeferred.cpp,SimpleDeferred)

StereoRendering : $(SAMPLE_CONFIG)
	$(call CompileSample,Samples/Core/StereoRendering/,StereoRendering.cpp,StereoRendering)

# Effect Samples

AmbientOcclusion : $(SAMPLE_CONFIG)
	$(call CompileSample,Samples/Effects/AmbientOcclusion/,AmbientOcclusion.cpp,AmbientOcclusion)

SkyBoxRenderer : $(SAMPLE_CONFIG)
	$(call CompileSample,Samples/Effects/SkyBoxRenderer/,SkyBoxRenderer.cpp,SkyBoxRenderer)

HashedAlpha : $(SAMPLE_CONFIG)
	$(call CompileSample,Samples/Effects/HashedAlpha/,HashedAlpha.cpp,HashedAlpha)

HDRToneMapping : $(SAMPLE_CONFIG)
	$(call CompileSample,Samples/Effects/HDRToneMapping/,HDRToneMapping.cpp,HDRToneMapping)

Shadows : $(SAMPLE_CONFIG)
	$(call CompileSample,Samples/Effects/Shadows/,Shadows.cpp,Shadows)

# Utilities

ModelViewer : $(SAMPLE_CONFIG)
	$(call CompileSample,Samples/Utils/ModelViewer/,ModelViewer.cpp,ModelViewer)

SceneEditor : $(SAMPLE_CONFIG)
	$(call CompileSample,Samples/Utils/SceneEditor/,SceneEditorApp.cpp,SceneEditor)

CC:=g++

INCLUDES = \
-I "Framework" \
-I "Framework/Source" \
-I "Framework/Externals/GLM" \
-I "Framework/Externals/OpenVR/headers" \
-I "Framework/Externals/RapidJson/include" \
-I "Framework/Externals/pybind11/include" \
-I "$(VULKAN_SDK)/include" \
$(shell pkg-config --cflags assimp gtk+-3.0 glfw3 python3) \
$(shell pkg-config --cflags libavcodec libavdevice libavformat libswscale libavutil)

ADDITIONAL_LIB_DIRS = -L "Bin/" \
-L "Framework/Externals/OpenVR/lib" \
-L "Framework/Externals/Slang/bin/linux-x86_64/release" \
-L "$(VULKAN_SDK)/lib"

LIBS = -lfalcor -lfalcorshared \
-lfreeimage -lslang -lslang-glslang -lopenvr_api \
$(shell pkg-config --libs assimp gtk+-3.0 glfw3 x11 python3) \
$(shell pkg-config --libs libavcodec libavdevice libavformat libswscale libavutil) \
-lvulkan -lstdc++fs -lpthread -lrt -lm -ldl -lz

# Compiler Flags
DEBUG_FLAGS:=-O0 -g -Wno-unused-variable
RELEASE_FLAGS:=-O2
DISABLED_WARNINGS:=-Wno-reorder -Wno-switch -Wno-unknown-pragmas -Wno-unused-function -Wno-sign-compare -Wno-nonnull-compare -Wno-attributes -Wno-address \
-Wno-unused-but-set-variable -Wno-misleading-indentation -Wno-strict-aliasing -Wno-maybe-uninitialized
# Disabling "unused-but-set-variable and misleading-indentation" ignores warnings when compiling imgui, not Falcor
COMMON_FLAGS=-c -Wall -Werror -std=c++17 -m64 $(DISABLED_WARNINGS)

# Defines
DEBUG_DEFINES:=-D "_DEBUG"
RELEASE_DEFINES:=
COMMON_DEFINES:=-D "FALCOR_VK" -D "GLM_FORCE_DEPTH_ZERO_TO_ONE" -D "_PROJECT_DIR_=\"Framework/Source\""

# Base source directory
SOURCE_DIR:=Framework/Source/

# All directories containing source code relative from the base Source folder. The "/" in the first line is to include the base Source directory
RELATIVE_DIRS:=/ \
API/ API/LowLevel/ API/Vulkan/ API/Vulkan/LowLevel/ \
Effects/AmbientOcclusion/ Effects/FXAA/ Effects/NormalMap/ Effects/ParticleSystem/ Effects/Shadows/ Effects/SkyBox/ Effects/TAA/ Effects/ToneMapping/ Effects/Utils/ \
Graphics/ Graphics/Camera/ Graphics/Material/ Graphics/Model/ Graphics/RenderGraph/ Graphics/Model/Loaders/ Graphics/Paths/ Graphics/Program/ Graphics/Scene/  Graphics/Scene/Editor/ \
Utils/ Utils/Math/ Utils/Scripting/ Utils/Picking/ Utils/PatternGenerators/ Utils/Psychophysics/ Utils/Platform/ Utils/Platform/Linux/ Utils/Video/ RenderPasses/ \
VR/ VR/OpenVR/ \
../Externals/dear_imgui/ ../Externals/dear_imgui_addons/imguinodegrapheditor/

# RELATIVE_DIRS, but now with paths relative to Makefile
SOURCE_DIRS = $(addprefix $(SOURCE_DIR), $(RELATIVE_DIRS))

# All source files enumerated with paths relative to Makefile (base repo)
# TODO: Fix VKGSO.
# Filter out VKGraphicsStateObject from rest of config because it currently cannot be compiled with optimizations.
ALL_SOURCE_FILES = $(filter-out %FalcorSharedObjects.cpp %VKGraphicsStateObject.cpp,$(wildcard $(addsuffix *.cpp,$(SOURCE_DIRS))))

# All expected .o files with the same path as their corresponding .cpp.
ALL_OBJ_FILES = $(patsubst %.cpp,%.o,$(ALL_SOURCE_FILES))

OUT_DIR:=Bin/

RELATIVE_RPATH:="-Wl,-rpath,"'$$'"ORIGIN/"

# Args: (1) Relative Directory, (2) Cpp filename, (3) Executable name
define CompileSample
	$(eval O_FILE=$(patsubst %.cpp,%.o,$(2)))
	@echo $(2)
	@$(CC) $(CXXFLAGS) $(1)$(2) -o $(1)$(O_FILE)
	@echo Linking $(3)
	@$(CC) -o $(OUT_DIR)$(3) $(1)$(O_FILE) $(ADDITIONAL_LIB_DIRS) $(LIBS) $(RELATIVE_RPATH)
	$(call MoveFalcorData,$(OUT_DIR))
	$(call MoveProjectData,$(1), $(OUT_DIR))
	@echo Built $(3)
endef

# Moves Falcor Data folder, ShadingUtils and Slang files to target directory. Contents of ShadingUtils will be placed inside a Data folder at destination
# Args: (1) Destination directory
define MoveFalcorData
	$(call MoveProjectData,Framework/Source/,$(1))
	@cp -r Framework/Source/ShadingUtils/* $(1)/Data/
	@cp Framework/Externals/Slang/bin/linux-x86_64/release/*.so $(1)
endef

# Copies the "Data" folder inside the directory specified by Source path to the Destination path
# Args: (1) Source, (2) Destination
define MoveProjectData
	@cp -r $(1)Data $(2) 2>/dev/null || :
endef

# Builds Falcor library in Release
Release : PreBuild ReleaseConfig $(OUT_DIR)libfalcorshared.so $(OUT_DIR)libfalcor.a

# Builds Falcor library in Debug
Debug : PreBuild DebugConfig $(OUT_DIR)libfalcorshared.so $(OUT_DIR)libfalcor.a

$(OUT_DIR)libfalcorshared.so : $(OUT_DIR)libfalcor.a
	$(call CompileSharedLibrary,Framework/FalcorSharedObjects/,FalcorSharedObjects.cpp,libfalcorshared.so)

define CompileSharedLibrary
	$(eval O_FILE=$(patsubst %.cpp,%.o,$(2)))
	@echo $(2)
	@$(CC) -fpic $(CXXFLAGS) $(1)$(2) -o $(1)$(O_FILE) -D BUILDING_SHARED_DLL
	@echo Linking $(3)
	@$(CC) -shared -o $(OUT_DIR)$(3) $(1)$(O_FILE)
endef

# Creates the lib
$(OUT_DIR)libfalcor.a : $(ALL_OBJ_FILES) $(SOURCE_DIR)API/Vulkan/VKGraphicsStateObject.o
	@mkdir -p $(dir $(OUT_DIR))
	@echo Creating $@
	@ar rcs $@ $^

$(ALL_OBJ_FILES) : %.o : %.cpp
	@echo $^
	@$(CC) $(CXXFLAGS) $^ -o $@

# TODO: Figure out how optimizations break this file
$(SOURCE_DIR)API/Vulkan/VKGraphicsStateObject.o :
	@echo $^
	@$(CC) $(INCLUDES) $(DEBUG_FLAGS) $(DEBUG_DEFINES) $(COMMON_FLAGS) $(COMMON_DEFINES) $(SOURCE_DIR)API/Vulkan/VKGraphicsStateObject.cpp -o $@

.PHONY : DebugConfig
DebugConfig :
	$(eval CXXFLAGS=$(INCLUDES) $(DEBUG_FLAGS) $(DEBUG_DEFINES) $(COMMON_FLAGS) $(COMMON_DEFINES))

.PHONY : ReleaseConfig
ReleaseConfig :
	$(eval CXXFLAGS=$(INCLUDES) $(RELEASE_FLAGS) $(RELEASE_DEFINES) $(COMMON_FLAGS) $(COMMON_DEFINES))

.PHONY : PreBuild
PreBuild :
	$(shell ./update_dependencies.sh >&2)

.PHONY : clean
clean :
	@find . -name "*.o" -type f -delete
	@find ./Framework/Externals/dear_imgui/ -name '*.o' -delete
	@rm -rf "Bin/"
	@rm -f Falcor.a
