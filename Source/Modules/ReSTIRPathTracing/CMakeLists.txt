add_library(ReSTIRPathTracing STATIC)

file(GLOB_RECURSE PAIRED_REUSE_PATTERN_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/PairedReusePattern/*.*"
)

target_sources(ReSTIRPathTracing PRIVATE
    ReSTIRPathTracing.cpp
    ReSTIRPathTracing.h
    PathReservoir.slang
    ReflectTypes.cs.slang
    ReSTIRPT.slang
    ReSTIRPathTracingParams.slang
    LoadShadingDataDynamic.slang
    SpatialResampling.cs.slang
    SpatialPathRetrace.cs.slang
    SpatialPrecomputeShiftBuffer.cs.slang
    TemporalResampling.cs.slang
    TemporalPathRetrace.cs.slang
    GenerateRetraceWorkload.cs.slang
    HybridShiftTraceRays.slang
    PathTracer.slang
    Shift.slang
    RetraceWorkQueue.slang
    RetraceScheduleDefinition.slangh
    FetchDenoiserData.cs.slang
    FillSampleID.cs.slang
    ComputeDuplicationMap.cs.slang
    ResamplingCommon.slang
    SpatialTraceShiftedPrimaryHits.cs.slang
    TemporalTraceShiftedPrimaryHits.cs.slang
    TemporalRobustCreateShiftVBuffer.cs.slang
    TemporalRobustPathRetrace.cs.slang
    TemporalRobustProduceWorkload.cs.slang
    TemporalRobustResampling.cs.slang
    TemporalPopulateFloatingReservoir.cs.slang
    ComputeReuseMotionVector.cs.slang
    TemporalReservoirSplatting.cs.slang
    PairedNeighborTransform.slang
    ${PAIRED_REUSE_PATTERN_SOURCES}
)

foreach(file IN LISTS PAIRED_REUSE_PATTERN_SOURCES)
    file(RELATIVE_PATH rel_path "${CMAKE_CURRENT_SOURCE_DIR}" "${file}")
    get_filename_component(dir "${rel_path}" DIRECTORY)
    string(REPLACE "/" "\\" group "${dir}")
    source_group("${group}" FILES "${file}")
endforeach()


target_copy_shaders(ReSTIRPathTracing ReSTIRPathTracing)

# Apply source grouping before adding generated files
target_source_group(ReSTIRPathTracing "Modules/")

function(target_copy_LUTs target output_dir)
    get_target_property(target_source_dir ${target} SOURCE_DIR)
    if(NOT target_source_dir)
        set(target_source_dir "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
    set(shader_output_dir "${FALCOR_SHADER_OUTPUT_DIRECTORY}/${output_dir}")
    
    # Find all PNG files in PairedReusePattern directory
    file(GLOB_RECURSE PAIRED_REUSE_PATTERN_IMAGES
        "${target_source_dir}/PairedReusePattern/*.png"
        "${target_source_dir}/PairedReusePattern/*.PNG"
    )
    
    # Generate corresponding output file paths
    set(OUTPUT_FILES "")
    foreach(src_file ${PAIRED_REUSE_PATTERN_IMAGES})
        file(RELATIVE_PATH rel_path "${target_source_dir}" "${src_file}")
        set(dst_file "${shader_output_dir}/${rel_path}")
        list(APPEND OUTPUT_FILES "${dst_file}")
    endforeach()
    
    # Create custom command that only runs when source files are newer than outputs
    add_custom_command(
        OUTPUT ${OUTPUT_FILES}
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${shader_output_dir}"
        COMMAND "${CMAKE_COMMAND}"
            -D SRC:PATH=${target_source_dir}
            -D DST:PATH=${shader_output_dir}
            -P "${CMAKE_CURRENT_LIST_DIR}/copy_luts.cmake"
        DEPENDS ${PAIRED_REUSE_PATTERN_IMAGES}
        COMMENT "${target}: Copying LUTs"
        VERBATIM
    )
    
    # Add the output files as sources to the target so it tracks them
    target_sources(${target} PRIVATE ${OUTPUT_FILES})
    set_source_files_properties(${OUTPUT_FILES} PROPERTIES 
        GENERATED TRUE 
        HEADER_FILE_ONLY TRUE
        SKIP_AUTOGEN TRUE
    )
    # Put generated files in a separate source group to avoid path issues
    source_group("Generated/LUTs" FILES ${OUTPUT_FILES})
endfunction() 

target_copy_LUTs(ReSTIRPathTracing ReSTIRPathTracing)

target_link_libraries(ReSTIRPathTracing
    PUBLIC
        Falcor
)

target_include_directories(ReSTIRPathTracing
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_target_properties(ReSTIRPathTracing
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        LIBRARY_OUTPUT_DIRECTORY ${FALCOR_RUNTIME_OUTPUT_DIRECTORY}
)

validate_headers(ReSTIRPathTracing)
