# Ninja can fail when embedding manifests just after building an executable or library.
# mt.exe may not be able to open the file because AV software has already locked it.
# msbuild presumably has some retry mechanism in place to deal with this, but when
# building with Ninja this problem can still emerge.
# To fix this, we set CMAKE_MT to execute a wrapper batch script which does re-run
# mt.exe in case it fails the first time.

if(WIN32 AND CMAKE_GENERATOR MATCHES "Ninja")
    message(STATUS "Setting up mt-retry workaround.")
    set(CMAKE_MT_ORIGINAL ${CMAKE_MT} CACHE FILEPATH "" FORCE)
    # Write out a wrapper batch script to re-run mt.exe when it fails.
    configure_file(${CMAKE_CURRENT_LIST_DIR}/mt-retry.bat.in ${CMAKE_CURRENT_BINARY_DIR}/mt-retry.bat)
    # Set CMAKE_MT to execute the wrapper batch script.
    set(CMAKE_MT ${CMAKE_CURRENT_BINARY_DIR}/mt-retry.bat CACHE FILEPATH "" FORCE)
endif()
