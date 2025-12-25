#include "Inference/EmotionRecognizer.h"
#include <onnxruntime_cxx_api.h>

static Ort::Env ortEnv(ORT_LOGGING_LEVEL_WARNING, "ser");

EmotionRecognizer::EmotionRecognizer(const std::string& modelPath) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    session = new Ort::Session(ortEnv, modelPath.c_str(), opts);
}

std::vector<float> EmotionRecognizer::Predict(const std::vector<float>& waveform) {
    Ort::AllocatorWithDefaultOptions allocator;
    auto* sess = static_cast<Ort::Session*>(session);

    const char* inputName = sess->GetInputName(0, allocator);
    const char* outputName = sess->GetOutputName(0, allocator);

    std::vector<int64_t> shape = {1, static_cast<int64_t>(waveform.size())};

    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo,
        const_cast<float*>(waveform.data()),
        waveform.size(),
        shape.data(),
        shape.size()
    );

    auto outputs = sess->Run(
        Ort::RunOptions{nullptr},
        &inputName,
        &inputTensor,
        1,
        &outputName,
        1
    );

    float* logits = outputs[0].GetTensorMutableData<float>();
    size_t count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();

    return std::vector<float>(logits, logits + count);
}
