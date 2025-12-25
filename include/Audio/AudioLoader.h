#pragma once
#include <vector>
#include <string>

struct AudioBuffer {
    std::vector<float> samples;
    int sampleRate;
};

class AudioLoader {
public:
    static AudioBuffer LoadWav(const std::string& path);
};
