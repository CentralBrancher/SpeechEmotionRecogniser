#define ORTCHAR_T char

#include "Inference/EmotionRecognizer.h"
#include <onnxruntime_cxx_api.h>
#include <stdexcept>
#include <vector>
#include <memory>

// Static ONNX Runtime environment (shared across sessions)
static Ort::Env ortEnv(ORT_LOGGING_LEVEL_WARNING, "ser");

EmotionRecognizer::EmotionRecognizer(const std::string& modelPath) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Create session
    #ifdef _WIN32
    session = new Ort::Session(ortEnv, std::wstring(modelPath.begin(), modelPath.end()).c_str(), opts);
    #else
    session = new Ort::Session(ortEnv, modelPath.c_str(), opts);
    #endif
}

EmotionRecognizer::~EmotionRecognizer() {
    if (session) {
        delete static_cast<Ort::Session*>(session);
        session = nullptr;
    }
}

std::vector<float> EmotionRecognizer::Predict(const std::vector<float>& waveform) {
    if (!session) throw std::runtime_error("ONNX session not initialized");

    auto* sess = static_cast<Ort::Session*>(session);
    Ort::AllocatorWithDefaultOptions allocator;

    // Get input/output names
    auto inputName = sess->GetInputNameAllocated(0, allocator);
    auto outputName = sess->GetOutputNameAllocated(0, allocator);

    // Input tensor shape: [1, waveform_length]
    std::vector<int64_t> inputShape = {1, static_cast<int64_t>(waveform.size())};

    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    // Create input tensor
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo,
        const_cast<float*>(waveform.data()),
        waveform.size(),
        inputShape.data(),
        inputShape.size()
    );

    // Run inference
    const char* input_names[] = { inputName.get() };
    const char* output_names[] = { outputName.get() };

    auto outputTensors = sess->Run(
        Ort::RunOptions{nullptr},
        input_names,
        &inputTensor,
        1,
        output_names,
        1
    );

    // Extract logits
    float* logits = outputTensors[0].GetTensorMutableData<float>();
    size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

    // Convert to std::vector<float>
    std::vector<float> output(logits, logits + count);

    return output;
}
