#include <iostream>
#include "Audio/AudioLoader.h"

int main() {
    std::string filePath = "../data/samples/anger.wav"; // adjust path if needed

    try {
        AudioBuffer buffer = AudioLoader::LoadWav(filePath);

        std::cout << "Audio loaded successfully!\n";
        std::cout << "Sample rate: " << buffer.sampleRate << "\n";
        std::cout << "Frames: " << buffer.samples.size() << "\n";
    } 
    catch (const std::exception& e) {
        std::cerr << "Error loading audio: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
