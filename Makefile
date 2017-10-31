CC = g++

INCLUDES = \
-I "Framework/Source/" \
-I "Framework/Externals/GLM" \
-I "Framework/Externals/GLFW/include" \
-I "Framework/Externals/AntTweakBar/include" \
-I "Framework/Externals/FreeImage" \
-I "Framework/Externals/ASSIMP/Include" \
-I "Framework/Externals/FFMPEG/include" \
-I "Framework/Externals/OculusSDK/LibOVR/Include" \
-I "Framework/Externals/OculusSDK/LibOVRKernel/Src" \
-I "Framework/Externals/openvr/headers" \
-I "Framework/Externals/VulkanSDK/Include" \
-I "Framework/Externals/RapidJson/include" \
-I "Framework/" \
-I "$(FALCOR_PYBIND11_PATH)/include" \
-I "$(FALCOR_PYTHON_PATH)/include" \
-I "Framework/Externals/nvapi" \
-I "$(VK_SDK_PATH)/Include"

# Compiler Flags
DEBUG_FLAGS:=-O0
RELEASE_FLAGS:=-O3
DISABLED_WARNINGS:=-Wno-unknown-pragmas -Wno-reorder -Wno-attributes -Wno-unused-function -Wno-switch -Wno-sign-compare -Wno-address -Wno-strict-aliasing
COMMON_FLAGS:=-c -Wall -Werror -std=c++14 -m64 $(DISABLED_WARNINGS)

# Defines
DEBUG_DEFINES:=-D "_DEBUG"
RELEASE_DEFINES:=
COMMON_DEFINES:=-D "FALCOR_VK" -D "GLM_FORCE_DEPTH_ZERO_TO_ONE"
# Windows defines  -D "WIN32" -D "_LIB" -D "_UNICODE" -D "UNICODE" 

# Base source directory
SOURCE_DIR:=Framework/Source/

# All directories containing source code relative from the base Source folder. The "/" in the first line is to include the base Source directory
#RELATIVE_DIRS := TestCode/
RELATIVE_DIRS:=\
/ \
API/ API/Vulkan/ API/Vulkan/LowLevel/ \
Effects/AmbientOcclusion/ Effects/NormalMap/ Effects/ParticleSystem/ Effects/Shadows/ Effects/SkyBox/ Effects/TAA/ Effects/ToneMapping/ Effects/Utils/ \
Graphics/ Graphics/Camera/ Graphics/Material/ Graphics/Model/ Graphics/Model/Loaders/ Graphics/Paths/ Graphics/Scene/  Graphics/Scene/Editor/ \
Utils/ Utils/Math/ Utils/Picking/ Utils/Psychophysics/ Utils/Video/  \
VR/ VR/OpenVR/

# 1,1    2,4    5,12    13, 20    21, 25     26,27

# RELATIVE_DIRS, but now with paths relative to Makefile
SOURCE_DIRS = $(addprefix $(SOURCE_DIR), $(wordlist 21,25,$(RELATIVE_DIRS)))
#SOURCE_DIRS = $(addprefix $(SOURCE_DIR), $(RELATIVE_DIRS))

# All source files enumerated with paths relative to Makefile (base repo)
ALL_SOURCE_FILES = $(wildcard $(addsuffix *.cpp,$(SOURCE_DIRS)))

# All expected .o files with the same path as their corresponding .cpp. Output redirected to actual output folder during compilation recipe
ALL_OBJ_FILES = $(patsubst %.cpp,%.o,$(ALL_SOURCE_FILES))

BASE_OUT_DIR:=Bin/gcc/
DEBUG_OUT_DIR=$(BASE_OUT_DIR)DebugVK/
RELEASE_OUT_DIR=$(BASE_OUT_DIR)ReleaseVK/

# OS specific versions of shell commands
ifeq ($(OS),Windows_NT)
	CreateDir = @if not exist "$(1)" mkdir "$(1)"
	RemoveDir = @rmdir /s /q "$(1)"
else
	CreateDir = @mkdir -p "$(1)"
# Can Linux's rm work if directory has a trailing '/'?
	RemoveDir = @rm -rf "$(1)"
endif

ReleaseVK : ReleaseConfig $(ALL_OBJ_FILES)
DebugVK : DebugConfig $(ALL_OBJ_FILES)

# Compiles a single file without linking
# Targets are the same path as the cpp, only with a .o extension.
# Path will be manipulated to point to proper output folder within this recipe
# TODO: Leaving the targets the same as cpp path but output elsewhere will break incremental building
#       If I leave the output in the same directory as the cpp, remember to copy the .o to the output folder on recipe completion
$(ALL_OBJ_FILES) : %.o : %.cpp
#	$(eval OUT_FILE=$(OUT_DIR)$(notdir $@))
	$(eval OUT_FILE=$@)
	@echo $^ $(OUT_FILE)
	@$(CC) $(INCLUDES) $(CONFIG_ARGS) $(COMMON_FLAGS) $(COMMON_DEFINES) $^ -o $(OUT_FILE)

.PHONY : DebugConfig
DebugConfig :
	$(eval OUT_DIR=$(DEBUG_OUT_DIR))
	$(call CreateDir, $(dir $(DEBUG_OUT_DIR)))
	$(eval CONFIG_ARGS=$(DEBUG_FLAGS) $(DEBUG_DEFINES))
	@echo Compiling to $(OUT_DIR)

.PHONY : ReleaseConfig
ReleaseConfig :
	$(eval OUT_DIR=$(RELEASE_OUT_DIR))
	$(call CreateDir, $(dir $(RELEASE_OUT_DIR)))
	$(eval CONFIG_ARGS=$(RELEASE_FLAGS) $(RELEASE_DEFINES))
#	@echo Compiling $(ALL_SOURCE_FILES) files to $(OUT_DIR)

.PHONY : Clean
Clean :
	find . -name "*.o" -type f -delete
#	@echo Cleaning $(BASE_OUT_DIR)
#	$(call RemoveDir,$(BASE_OUT_DIR))

.PHONY : CleanDebug
CleanDebug :
	@echo Cleaning $(DEBUG_OUT_DIR)
	$(call RemoveDir,$(DEBUG_OUT_DIR))

.PHONY : CleanRelease
CleanRelease :
	@echo Cleaning $(RELEASE_OUT_DIR)
	$(call RemoveDir,$(RELEASE_OUT_DIR))
