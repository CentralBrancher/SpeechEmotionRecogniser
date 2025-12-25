#pragma once
#include <vector>
#include <string>

class EmotionRecognizer {
public:
    explicit EmotionRecognizer(const std::string& modelPath);
    ~EmotionRecognizer();  // Destructor to clean up ONNX session

    std::vector<float> Predict(const std::vector<float>& waveform);

private:
    void* session;  // Raw pointer to Ort::Session
};
