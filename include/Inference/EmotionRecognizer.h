#pragma once
#include <vector>
#include <string>
#include <memory>

#include <onnxruntime_cxx_api.h>

class EmotionRecognizer {
public:
    explicit EmotionRecognizer(const std::string& modelPath);
    ~EmotionRecognizer() = default;

    std::vector<float> Predict(const std::vector<float>& waveform);

private:
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
};
