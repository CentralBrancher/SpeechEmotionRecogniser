cmake_minimum_required(VERSION 3.20)
project(SpeechEmotionRecogniser VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Executable ----
add_executable(SpeechEmotionRecogniser
    src/main.cpp
    src/Audio/AudioLoader.cpp
    src/Inference/EmotionRecognizer.cpp
)

# ---- Include paths ----
target_include_directories(SpeechEmotionRecogniser PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/third_party
    ${PROJECT_SOURCE_DIR}/third_party/onnxruntime/include
)

# ---- Link ONNX Runtime lib ----
target_link_libraries(SpeechEmotionRecogniser PRIVATE
    "${PROJECT_SOURCE_DIR}/third_party/onnxruntime/lib/onnxruntime.lib"
)

# ---- Copy ONNX Runtime DLL to output folder ----
add_custom_command(TARGET SpeechEmotionRecogniser POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PROJECT_SOURCE_DIR}/third_party/onnxruntime/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:SpeechEmotionRecogniser>
)

# ---- Windows compiler warnings ----
if (MSVC)
    target_compile_options(SpeechEmotionRecogniser PRIVATE /W4)
endif()
