if(NOT FALCOR_HAS_CUDA)
    return()
endif()

add_falcor_executable(CudaInterop)

target_sources(CudaInterop PRIVATE
    CopySurface.cu
    CopySurface.h
    CudaInterop.cpp
    CudaInterop.h
    FalcorCUDA.cpp
    FalcorCUDA.h
)

set_target_properties(CudaInterop PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_source_group(CudaInterop "Samples")
