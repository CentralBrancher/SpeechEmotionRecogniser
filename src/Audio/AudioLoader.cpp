#define DR_WAV_IMPLEMENTATION
#include <dr_wav/dr_wav.h>

#include "Audio/AudioLoader.h"
#include <stdexcept>

AudioBuffer AudioLoader::LoadWav(const std::string& path) {
    drwav wav;
    if (!drwav_init_file(&wav, path.c_str(), nullptr)) {
        throw std::runtime_error("Failed to load WAV file");
    }

    std::vector<float> samples(wav.totalPCMFrameCount * wav.channels);
    drwav_read_pcm_frames_f32(&wav, wav.totalPCMFrameCount, samples.data());

    drwav_uninit(&wav);

    // Convert to mono if needed
    if (wav.channels > 1) {
        std::vector<float> mono(wav.totalPCMFrameCount);
        for (uint64_t i = 0; i < wav.totalPCMFrameCount; ++i) {
            float sum = 0.0f;
            for (uint32_t ch = 0; ch < wav.channels; ++ch) {
                sum += samples[i * wav.channels + ch];
            }
            mono[i] = sum / wav.channels;
        }
        samples = std::move(mono);
    }

    return { samples, static_cast<int>(wav.sampleRate) };
}
