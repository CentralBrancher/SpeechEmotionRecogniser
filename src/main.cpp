#include "Audio/AudioLoader.h"
#include "Inference/EmotionRecognizer.h"
#include "Domain/EmotionLabels.h"

#include <iostream>
#include <cmath>
#include <algorithm>

static std::vector<float> Softmax(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size());
    float maxLogit = *std::max_element(logits.begin(), logits.end());

    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - maxLogit);
        sum += probs[i];
    }

    for (auto& p : probs) p /= sum;
    return probs;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: SpeechEmotionRecogniser <audio.wav>\n";
        return 1;
    }

    auto audio = AudioLoader::LoadWav(argv[1]);

    EmotionRecognizer recogniser("models/model_int8.onnx");
    auto logits = recogniser.Predict(audio.samples);
    auto probs = Softmax(logits);

    auto maxIt = std::max_element(probs.begin(), probs.end());
    int idx = static_cast<int>(std::distance(probs.begin(), maxIt));

    std::cout << "Prediction: " << EMOTION_LABELS[idx] << "\n";
    std::cout << "Confidence: " << *maxIt << "\n";

    return 0;
}
