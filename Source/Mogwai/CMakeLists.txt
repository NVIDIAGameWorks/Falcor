add_falcor_executable(Mogwai)

target_sources(Mogwai PRIVATE
    AppData.cpp
    AppData.h
    Mogwai.cpp
    Mogwai.h
    MogwaiScripting.cpp
    MogwaiSettings.cpp
    MogwaiSettings.h

    Extensions/Capture/CaptureTrigger.cpp
    Extensions/Capture/CaptureTrigger.h
    Extensions/Capture/FrameCapture.cpp
    Extensions/Capture/FrameCapture.h
    Extensions/Capture/VideoCapture.cpp
    Extensions/Capture/VideoCapture.h
    Extensions/Profiler/TimingCapture.cpp
    Extensions/Profiler/TimingCapture.h
)

target_copy_data_folder(Mogwai)

target_source_group(Mogwai "/")
