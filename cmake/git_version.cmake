find_package(Git QUIET)

set(CURRENT_LIST_DIR ${CMAKE_CURRENT_LIST_DIR})

if(NOT DEFINED PRE_CONFIGURE_FILE)
    set(PRE_CONFIGURE_FILE ${CMAKE_CURRENT_LIST_DIR}/git_version.h.in)
endif()

if (NOT DEFINED POST_CONFIGURE_FILE)
    set(POST_CONFIGURE_FILE ${CMAKE_CURRENT_BINARY_DIR}/git_version/git_version.h)
endif()

if(NOT DEFINED STATE_FILE)
    set(STATE_FILE ${CMAKE_CURRENT_BINARY_DIR}/git_version_state)
endif()


function(write_state hash)
    file(WRITE ${STATE_FILE} ${hash})
endfunction()

function(read_state hash)
    if(EXISTS ${STATE_FILE})
        file(READ ${STATE_FILE} content)
        set(${hash} ${content} PARENT_SCOPE)
    endif()
endfunction()

macro(run_git)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" ${ARGV}
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
        RESULT_VARIABLE exit_code
        OUTPUT_VARIABLE output
        ERROR_VARIABLE stderr
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endmacro()

function(git_version_check)

    set(GIT_AVAILABLE "0")
    set(GIT_COMMIT "unknown")
    set(GIT_BRANCH "unknown")
    set(GIT_DIRTY "0")

    if(Git_FOUND)
        set(GIT_AVAILABLE "1")

        # Get commit
        run_git(log -1 --format=%h)
        if(exit_code EQUAL 0)
            set(GIT_COMMIT ${output})
        endif()

        # Get branch
        run_git(symbolic-ref --short -q HEAD)
        if(exit_code EQUAL 0)
            set(GIT_BRANCH ${output})
        endif()

        # Check for uncommitted changes
        run_git(status --porcelain -uno)
        if(exit_code EQUAL 0)
            if(NOT "${output}" STREQUAL "")
                set(GIT_DIRTY "1")
            endif()
        endif()
    endif()

    string(SHA256 hash "${GIT_AVAILABLE},${GIT_COMMIT},${GIT_BRANCH},${GIT_DIRTY}")
    read_state(old_hash)

    # Only update git_version.h and state cache if anything changed.
    if(NOT EXISTS ${POST_CONFIGURE_FILE}
        OR NOT DEFINED old_hash
        OR NOT "${hash}" STREQUAL "${old_hash}")
        write_state(${hash})
        get_filename_component(PARENT_DIR ${POST_CONFIGURE_FILE} DIRECTORY)
        file(MAKE_DIRECTORY ${PARENT_DIR})
        configure_file(${PRE_CONFIGURE_FILE} ${POST_CONFIGURE_FILE} @ONLY)
    endif()

endfunction()

function(git_version_setup)
    add_custom_target(git_version_run_check COMMAND ${CMAKE_COMMAND}
        -DGIT_VERSION_RUN_CHECK=1
        -DPRE_CONFIGURE_FILE=${PRE_CONFIGURE_FILE}
        -DPOST_CONFIGURE_FILE=${POST_CONFIGURE_FILE}
        -DSTATE_FILE=${STATE_FILE}
        -P ${CURRENT_LIST_DIR}/git_version.cmake
        BYPRODUCTS ${POST_CONFIGURE_FILE} ${STATE_FILE}
    )
    set_target_properties(git_version_run_check PROPERTIES FOLDER "Misc")

    add_library(git_version INTERFACE)
    target_include_directories(git_version INTERFACE ${CMAKE_BINARY_DIR}/git_version)
    add_dependencies(git_version git_version_run_check)

    git_version_check()
endfunction()

# Used to run from external cmake process.
if(GIT_VERSION_RUN_CHECK)
    git_version_check()
endif()
