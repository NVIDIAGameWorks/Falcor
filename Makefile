CC = g++

INCLUDES = \
-I"Framework/Source/" \
-I"Framework/Externals/GLM" \
-I"Framework/Externals/GLFW/include" \
-I"Framework/Externals/AntTweakBar/include" \
-I"Framework/Externals/FreeImage" \
-I"Framework/Externals/assimp/include" \
-I"Framework/Externals/FFMpeg/Include" \
-I"Framework/Externals/OculusSDK/LibOVR/Include" \
-I"Framework/Externals/OculusSDK/LibOVRKernel/Src" \
-I"Framework/Externals/openvr/headers" \
-I"Framework/Externals/VulkanSDK/Include" \
-I"Framework/" \
-I"$(FALCOR_PYBIND11_PATH)/include" \
-I"$(FALCOR_PYTHON_PATH)/include" \
-I"Framework/Externals/nvapi" \
-I"$(VK_SDK_PATH)/Include"

# Compiler Flags
CL_DEBUG_FLAGS = /ZI /Od /RTC1 /MDd
CL_RELEASE_FLAGS = /Gy /Zi /O2 /Oi /MD
CL_COMMON_FLAGS = /c /MP /GS /W3 /WX /Zc:wchar_t /Gm- /Zc:inline /fp:precise /errorReport:prompt /Zc:forScope /Gd /EHsc /nologo

DEBUG_FLAGS =
RELEASE_FLAGS =
COMMON_FLAGS =

# Defines
DEBUG_DEFINES = /D "_DEBUG"
RELEASE_DEFINES = /D "NDEBUG"
D3D12_DEFINES = /D "FALCOR_D3D12"
VK_DEFINES = /D "FALCOR_VK"
COMMON_DEFINES = /D "WIN32" /D "_LIB" /D "_UNICODE" /D "UNICODE" /D "GLM_FORCE_DEPTH_ZERO_TO_ONE"

# Arguments to set output
SET_OUTDIR = /Fo"$(OUTDIR)/" /Fd"$(OUTDIR)/Falcor.pdb"

# Base source directory
SOURCE_DIR = Framework/Source/

# All directories containing source code, relative from the base Source folder. The first line "/" is to include the base Source directory
RELATIVE_DIRS = TestCode/
# RELATIVE_DIRS = / \
# API/ API/Vulkan/ API/Vulkan/LowLevel/ \
# Effects/AmbientOcclusion/ Effects/NormalMap/ Effects/ParticleSystem/ Effects/Shadows/ Effects/SkyBox/ Effects/TAA/ Effects/ToneMapping/ Effects/Utils/ \
# Graphics/ Graphics/Camera/ Graphics/Material/ Graphics/Model/ Graphics/Model/Loaders/ Graphics/Paths/ Graphics/Scene/  Graphics/Scene/Editor/ \
# Utils/ Utils/Math/ Utils/Picking/ Utils/Psychophysics/ Utils/Video/  \
# VR/ VR/OpenVR/ \
# TestCode/

SOURCE_FOLDERS = $(addprefix $(SOURCE_DIR)/, $(RELATIVE_DIRS))

# All source files relative to Makefile (base repo)
ALL_SOURCE_FILES = $(wildcard $(addsuffix *.cpp,$(SOURCE_FOLDERS)))

# All source files relative to base source folder (Framework/Source/)
ALL_SOURCE_FILES_RELATIVE = $(patsubst $(SOURCE_DIR)/%,%,$(ALL_SOURCE_FILES))

ALL_OBJ_FILES_RELATIVE = $(patsubst %.cpp,%.o,$(ALL_SOURCE_FILES_RELATIVE))

#DEBUG_OUT_DIR = Bin/Int/x64/DebugVK
#RELEASE_OUT_DIR = Bin/Int/x64/ReleaseVK
DEBUG_OUT_DIR = Bin/gcc/DebugVK
RELEASE_OUT_DIR = Bin/gcc/ReleaseVK

ALL_DEBUG_OBJ_FILES=$(addprefix $(DEBUG_OUT_DIR)/, $(ALL_OBJ_FILES_RELATIVE))
ALL_RELEASE_OBJ_FILES=$(addprefix $(RELEASE_OUT_DIR)/, $(ALL_OBJ_FILES_RELATIVE))

# Function to create output directory based on OS
ifeq ($(OS),Windows_NT)
	CreateDir = @if not exist $(1) mkdir $(1)
else
	CreateDir = @mkdir -p $(1)
endif

ReleaseVK : $(ALL_RELEASE_OBJ_FILES)
	@echo $^

# Compiles a file without linking
# Targets are expected OBJ filenames relative to base Makefile directory
$(ALL_RELEASE_OBJ_FILES) : $(RELEASE_OUT_DIR)/%.o : $(SOURCE_DIR)/%.cpp
	@echo $^ $@
	$(call CreateDir, "$(dir $@)")
#	if not exist "$(dir $@)" mkdir "$(dir $@)"
#	g++ -c $^ -o $@
