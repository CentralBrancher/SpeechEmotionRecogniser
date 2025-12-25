#include <iostream>
#include <iomanip>
#include <string>
#include "Audio/AudioLoader.h"
#include "Inference/EmotionRecognizer.h"
#include "Domain/EmotionLabels.h"
#include <cmath>
#include <algorithm>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_wav>\n";
        return 1;
    }

    std::string audioPath = argv[1];

    try {
        // Load audio
        AudioBuffer audio = AudioLoader::LoadWav(audioPath);
        std::cout << "Audio loaded successfully!\n";
        std::cout << "Sample rate: " << audio.sampleRate << "\n";
        std::cout << "Frames: " << audio.samples.size() << "\n";

        // Load model
        EmotionRecognizer recognizer("models/model_int8.onnx");

        // Run prediction
        std::vector<float> logits = recognizer.Predict(audio.samples);

        // Apply softmax
        float maxLogit = *std::max_element(logits.begin(), logits.end());
        std::vector<float> probs(logits.size());
        float sumExp = 0.0f;
        for (size_t i = 0; i < logits.size(); ++i) {
            probs[i] = std::exp(logits[i] - maxLogit);
            sumExp += probs[i];
        }
        for (float& p : probs) p /= sumExp;

        // Find top prediction
        auto maxIt = std::max_element(probs.begin(), probs.end());
        size_t idx = std::distance(probs.begin(), maxIt);

        std::cout << "\nPrediction: " << EMOTION_LABELS[idx] << "\n";
        std::cout << "Confidence: " << std::fixed << std::setprecision(3) << *maxIt << "\n";

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
