cmake_minimum_required(VERSION 3.20)

project(SpeechEmotionRecogniser
    VERSION 0.1
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Executable ----
add_executable(SpeechEmotionRecogniser
    src/main.cpp
    src/Audio/AudioLoader.cpp
)

# ---- Include paths ----
target_include_directories(SpeechEmotionRecogniser
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/third_party
)

# ---- Windows specifics ----
if (MSVC)
    target_compile_options(SpeechEmotionRecogniser PRIVATE /W4)
endif()
