CC = g++

INCLUDES = \
-I "Framework" \
-I "Framework/Source" \
-I "Framework/Externals/GLM" \
-I "Framework/Externals/GLFW/include" \
-I "Framework/Externals/OpenVR/headers" \
-I "Framework/Externals/RapidJson/include" \
-I "Framework/Externals/FFMPEG/include" \
-I "$(VULKAN_SDK)/include" \
-I "/usr/include" \
$(shell pkg-config --cflags assimp gtk+-3.0)

# No pkg-config metadata: glm freeimage
# glm needs to be a newer version than what's on Linux repos, so pull from packman
# Using older version of RapidJson, pull from packman
# Need newer version of ffmpeg than whats on ubuntu
# Compiling GLFW with Wayland backend

# -I "Framework/Externals/FreeImage/include" assume in usr/include
# -I "Framework/Externals/ASSIMP/include"
# -I "$(FALCOR_PYBIND11_PATH)/include"
# -I "$(FALCOR_PYTHON_PATH)/include"

ADDITIONAL_LIB_DIRS = -L "Bin/" \
-L "Framework/Externals/OpenVR/lib" \
-L "Framework/Externals/FFMPEG/lib" \
-L "Framework/Externals/GLFW/lib" \
-L "Framework/Externals/Slang/bin/linux-x86_64/release" \
-L "$(VULKAN_SDK)/lib"

LIBS = -lfalcor \
-lfreeimage -lslang -lslang-glslang -lopenvr_api \
$(shell pkg-config --libs assimp gtk+-3.0 ) -lglfw3 \
-lvulkan -lxkbcommon -lwayland-cursor -lwayland-egl -lwayland-client  \
-lstdc++fs -lrt -lm -ldl -lz -lpthread
# ffmpeg stuff: -lavcodec -lavdevice -lavformat -lswscale -lavutil -lopus

# Compiler Flags
DEBUG_FLAGS:=-O0 -g -Wno-unused-variable
RELEASE_FLAGS:=-O3
DISABLED_WARNINGS:=-Wno-unknown-pragmas -Wno-reorder -Wno-attributes -Wno-unused-function -Wno-switch -Wno-sign-compare -Wno-address -Wno-strict-aliasing -Wno-nonnull-compare \
-Wno-unused-but-set-variable -Wno-misleading-indentation
# Disabling "unused-but-set-variable and misleading-indenttion" ignores warnings when compiling imgui, not Falcor
COMMON_FLAGS=-c -Wall -Werror -std=c++17 -m64 $(DISABLED_WARNINGS)

# Defines
DEBUG_DEFINES:=-D "_DEBUG"
RELEASE_DEFINES:=
COMMON_DEFINES:=-D "FALCOR_VK" -D "GLM_FORCE_DEPTH_ZERO_TO_ONE"

# Base source directory
SOURCE_DIR:=Framework/Source/

# All directories containing source code relative from the base Source folder. The "/" in the first line is to include the base Source directory
RELATIVE_DIRS:= \
/ \
API/ API/LowLevel/ API/Vulkan/ API/Vulkan/LowLevel/ \
Effects/AmbientOcclusion/ Effects/NormalMap/ Effects/ParticleSystem/ Effects/Shadows/ Effects/SkyBox/ Effects/TAA/ Effects/ToneMapping/ Effects/Utils/ \
Graphics/ Graphics/Camera/ Graphics/Material/ Graphics/Model/ Graphics/Model/Loaders/ Graphics/Paths/ Graphics/Scene/  Graphics/Scene/Editor/ \
Utils/ Utils/Math/ Utils/Picking/ Utils/Psychophysics/ Utils/Platform/ Utils/Platform/Linux/ \
VR/ VR/OpenVR/ \
../Externals/dear_imgui/
# Utils/Video/

# 1,1    2,5    6,13    14,21    22,28    29,31 , 32
# SOURCE_DIRS = $(addprefix $(SOURCE_DIR), $(wordlist 1,30,$(RELATIVE_DIRS)))
# RELATIVE_DIRS, but now with paths relative to Makefile
SOURCE_DIRS = $(addprefix $(SOURCE_DIR), $(RELATIVE_DIRS))

# All source files enumerated with paths relative to Makefile (base repo)
ALL_SOURCE_FILES = $(wildcard $(addsuffix *.cpp,$(SOURCE_DIRS)))

# All expected .o files with the same path as their corresponding .cpp. Output redirected to actual output folder during compilation recipe
ALL_OBJ_FILES = $(patsubst %.cpp,%.o,$(ALL_SOURCE_FILES))

OUT_DIR:=Bin/

FeatureDemo : DebugVK
	$(eval DIR=Samples/FeatureDemo/)
	@$(CC) $(COMPILE_ARGS) $(DIR)FeatureDemo.cpp -o $(DIR)FeatureDemo.o
	@$(CC) $(COMPILE_ARGS) $(DIR)FeatureDemoControls.cpp -o $(DIR)FeatureDemoControls.o
	@$(CC) $(COMPILE_ARGS) $(DIR)FeatureDemoSceneRenderer.cpp -o $(DIR)FeatureDemoSceneRenderer.o
	@$(CC) -o $(OUT_DIR)FeatureDemo $(DIR)FeatureDemo.o $(DIR)FeatureDemoControls.o $(DIR)FeatureDemoSceneRenderer.o $(ADDITIONAL_LIB_DIRS) $(LIBS)
	@echo Built $@

Shadows : DebugVK
	$(eval DIR=Samples/Effects/Shadows/)
	@$(CC) $(COMPILE_ARGS) $(DIR)Shadows.cpp -o $(DIR)Shadows.o
	@$(CC) -o $(OUT_DIR)Shadows $(DIR)Shadows.o $(ADDITIONAL_LIB_DIRS) $(LIBS)
	@echo Built $@

ShaderToy : DebugVK
	$(eval DIR=Samples/Core/ShaderToy/)
	@$(CC) $(COMPILE_ARGS) $(DIR)ShaderToy.cpp -o $(DIR)ShaderToy.o
	@$(CC) -o $(OUT_DIR)ShaderToy $(DIR)ShaderToy.o $(ADDITIONAL_LIB_DIRS) $(LIBS)
	@echo Built $@

MultiPassPostProcess : DebugVK
	$(eval DIR=Samples/Core/MultiPassPostProcess/)
	@$(CC) $(COMPILE_ARGS) $(DIR)MultiPassPostProcess.cpp -o $(DIR)MultiPassPostProcess.o
	@$(CC) -o $(OUT_DIR)MultiPassPostProcess $(DIR)MultiPassPostProcess.o $(ADDITIONAL_LIB_DIRS) $(LIBS)
	@echo Built $@

ModelViewer : DebugVK
	$(eval DIR=Samples/Utils/ModelViewer/)
	@$(CC) $(COMPILE_ARGS) $(DIR)ModelViewer.cpp -o $(DIR)ModelViewer.o
	@$(CC) -o $(OUT_DIR)ModelViewer $(DIR)ModelViewer.o $(ADDITIONAL_LIB_DIRS) $(LIBS)
	@echo Built $@

ProjectTemplate : DebugVK
	$(eval DIR=Samples/Core/ProjectTemplate/)
	@$(CC) $(COMPILE_ARGS) $(DIR)ProjectTemplate.cpp -o $(DIR)ProjectTemplate.o
	@$(CC) -o $(OUT_DIR)ProjectTemplate $(DIR)ProjectTemplate.o $(ADDITIONAL_LIB_DIRS) $(LIBS)
	@echo Built $@

ReleaseVK : ReleaseConfig $(OUT_DIR)libfalcor.a

DebugVK : DebugConfig $(OUT_DIR)libfalcor.a

$(OUT_DIR)libfalcor.a : $(ALL_OBJ_FILES)
	@mkdir -p $(dir $(OUT_DIR))
	@echo Creating $@
	@ar rcs $@ $^

$(ALL_OBJ_FILES) : %.o : %.cpp
	@echo $^ $@
	@$(CC) $(COMPILE_ARGS) $^ -o $@

.PHONY : DebugConfig
DebugConfig :
	$(eval COMPILE_ARGS=$(INCLUDES) $(DEBUG_FLAGS) $(DEBUG_DEFINES) $(COMMON_FLAGS) $(COMMON_DEFINES))

.PHONY : ReleaseConfig
ReleaseConfig :
	$(eval COMPILE_ARGS=$(INCLUDES) $(RELEASE_FLAGS) $(RELEASE_DEFINES) $(COMMON_FLAGS) $(COMMON_DEFINES))

.PHONY : clean
clean :
	@find . -name "*.o" -type f -delete
#	@rm -rf "Bin/"
#	@rm -f Falcor.a
