#pragma once
#include <vector>
#include <string>

class EmotionRecognizer {
public:
    explicit EmotionRecognizer(const std::string& modelPath);
    std::vector<float> Predict(const std::vector<float>& waveform);

private:
    void* env;
    void* session;
};
