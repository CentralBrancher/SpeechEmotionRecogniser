#include <iostream>
#include <iomanip>
#include "Audio/AudioLoader.h"
#include "Inference/EmotionRecognizer.h"
#include "Domain/EmotionLabels.h"
#include <algorithm>
#include <cmath>

int main() {
    std::string filePath = "../data/samples/anger.wav"; // adjust path if needed

    try {
        // 1️⃣ Load audio
        AudioBuffer buffer = AudioLoader::LoadWav(filePath);
        std::cout << "Audio loaded successfully!\n";
        std::cout << "Sample rate: " << buffer.sampleRate << "\n";
        std::cout << "Frames: " << buffer.samples.size() << "\n";

        // 2️⃣ Load ONNX model
        EmotionRecognizer recognizer("../models/model_fp16.onnx"); // adjust path if needed
        std::cout << "EmotionRecognizer model loaded successfully!\n";

        // 3️⃣ Predict
        std::vector<float> logits = recognizer.Predict(buffer.samples);

        // 4️⃣ Softmax
        float maxLogit = *std::max_element(logits.begin(), logits.end());
        std::vector<float> probs(logits.size());
        float sumExp = 0.0f;
        for (size_t i = 0; i < logits.size(); ++i) {
            probs[i] = std::exp(logits[i] - maxLogit);
            sumExp += probs[i];
        }
        for (float& p : probs) p /= sumExp;

        // 5️⃣ Top prediction
        auto maxIt = std::max_element(probs.begin(), probs.end());
        size_t idx = std::distance(probs.begin(), maxIt);

        std::cout << "\nPrediction: " << EMOTION_LABELS[idx] << "\n";
        std::cout << "Confidence: " << std::fixed << std::setprecision(3) << *maxIt << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
