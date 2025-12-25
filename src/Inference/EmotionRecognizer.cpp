#include "Inference/EmotionRecognizer.h"
#include <stdexcept>

EmotionRecognizer::EmotionRecognizer(const std::string& modelPath) {
    // Create ONNX environment (runtime, NOT static)
    env = std::make_unique<Ort::Env>(
        ORT_LOGGING_LEVEL_WARNING,
        "SpeechEmotionRecogniser"
    );

    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    #ifdef _WIN32
        std::wstring wpath(modelPath.begin(), modelPath.end());
        session = std::make_unique<Ort::Session>(*env, wpath.c_str(), opts);
    #else
        session = std::make_unique<Ort::Session>(*env, modelPath.c_str(), opts);
    #endif
}

std::vector<float> EmotionRecognizer::Predict(const std::vector<float>& waveform) {
    if (!session)
        throw std::runtime_error("ONNX session not initialized");

    Ort::AllocatorWithDefaultOptions allocator;

    auto inputName  = session->GetInputNameAllocated(0, allocator);
    auto outputName = session->GetOutputNameAllocated(0, allocator);

    std::vector<int64_t> inputShape = {
        1,
        static_cast<int64_t>(waveform.size())
    };

    Ort::MemoryInfo memInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value inputTensor =
        Ort::Value::CreateTensor<float>(
            memInfo,
            const_cast<float*>(waveform.data()),
            waveform.size(),
            inputShape.data(),
            inputShape.size()
        );

    const char* inputNames[]  = { inputName.get() };
    const char* outputNames[] = { outputName.get() };

    auto outputs = session->Run(
        Ort::RunOptions{ nullptr },
        inputNames,
        &inputTensor,
        1,
        outputNames,
        1
    );

    float* logits =
        outputs[0].GetTensorMutableData<float>();

    size_t count =
        outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();

    return { logits, logits + count };
}