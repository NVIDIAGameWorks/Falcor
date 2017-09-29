CC = cl

INCLUDES = \
/I"Framework/Source/" \
/I"Framework/Externals/GLM" \
/I"Framework/Externals/GLFW/include" \
/I"Framework/Externals/AntTweakBar/include" \
/I"Framework/Externals/FreeImage" \
/I"Framework/Externals/assimp/include" \
/I"Framework/Externals/FFMpeg/Include" \
/I"Framework/Externals/OculusSDK/LibOVR/Include" \
/I"Framework/Externals/OculusSDK/LibOVRKernel/Src" \
/I"Framework/Externals/openvr/headers" \
/I"Framework/Externals/VulkanSDK/Include" \
/I"Framework/" \
/I"C:/pybind11/include" \
/I"C:/Python35/include" \
/I"Framework/Externals/nvapi"

# Compiler Flags
DEBUG_FLAGS = /ZI /Od /RTC1 /MDd
RELEASE_FLAGS = /Gy /Zi /O2 /Oi /MD
COMMON_FLAGS = /c /MP /GS /W3 /WX /Zc:wchar_t /Gm- /Zc:inline /fp:precise /errorReport:prompt /Zc:forScope /Gd /EHsc /nologo

# Defines
DEBUG_DEFINES = /D "_DEBUG"
RELEASE_DEFINES = /D "NDEBUG"
D3D12_DEFINES = /D "FALCOR_D3D12"
VK_DEFINES = /D "FALCOR_VK"
COMMON_DEFINES = /D "WIN32" /D "_LIB" /D "_UNICODE" /D "UNICODE" /D "GLM_FORCE_DEPTH_ZERO_TO_ONE"

# Output directory
OUTDIR = Bin/Int/x64/ReleaseD3D12

# Arguments to set output
SETOUTDIR = /Fo"$(OUTDIR)/" /Fd"$(OUTDIR)/Falcor.pdb"

# Base source directory
SOURCE_DIR = Framework/Source

D3D12_FILES = $(SOURCE_DIR)/API/D3D/*.cpp $(SOURCE_DIR)/API/D3D/D3D12/*.cpp $(SOURCE_DIR)/API/D3D/D3D12/LowLevel/*.cpp
VK_FILES = $(SOURCE_DIR)/API/Vulkan/*.cpp $(SOURCE_DIR)/API/Vulkan/LowLevel/*.cpp
COMMON_FILES = $(SOURCE_DIR)/*.cpp \
$(SOURCE_DIR)/Effects/AmbientOcclusion/*.cpp $(SOURCE_DIR)/Effects/NormalMap/*.cpp $(SOURCE_DIR)/Effects/ParticleSystem/*.cpp $(SOURCE_DIR)/Effects/Shadows/*.cpp $(SOURCE_DIR)/Effects/SkyBox/*.cpp $(SOURCE_DIR)/Effects/TAA/*.cpp $(SOURCE_DIR)/Effects/ToneMapping/*.cpp $(SOURCE_DIR)/Effects/Utils/*.cpp\
$(SOURCE_DIR)/Graphics/*.cpp $(SOURCE_DIR)/Graphics/Camera/*.cpp $(SOURCE_DIR)/Graphics/Material/*.cpp $(SOURCE_DIR)/Graphics/Model/*.cpp $(SOURCE_DIR)/Graphics/Model/Loaders/*.cpp $(SOURCE_DIR)/Graphics/Paths/*.cpp $(SOURCE_DIR)/Graphics/Scene/*.cpp  $(SOURCE_DIR)/Graphics/Scene/Editor/*.cpp\
$(SOURCE_DIR)/Utils/*.cpp $(SOURCE_DIR)/Utils/Math/*.cpp $(SOURCE_DIR)/Utils/Picking/*.cpp $(SOURCE_DIR)/Utils/Psychophysics/*.cpp $(SOURCE_DIR)/Utils/Video/*.cpp  \
$(SOURCE_DIR)/VR/*.cpp $(SOURCE_DIR)/VR/OpenVR/*.cpp

released3d12 :
	$(CC) $(INCLUDES) $(DEFINES) $(SETOUTDIR) $(COMMON_FLAGS) $(RELEASE_FLAGS) $(COMMON_DEFINES) $(D3D12_DEFINES) $(RELEASE_DEFINES) $(D3D12_FILES) $(COMMON_FILES)

# Target for testing variable values
var :
	echo $(SETOUTDIR)

# This can't be run in the makefile, the environment changes don't persist. Leaving it here for reference
env :
	"%VS140COMNTOOLS%../../VC/vcvarsall.bat" x64