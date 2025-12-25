# SpeechEmotionRecogniser

A **C++17 command-line application** for speech emotion recognition using a fine-tuned **Wav2Vec2 ONNX model**.
The tool loads a WAV file, runs inference via **ONNX Runtime**, and outputs the predicted emotion and confidence.

---

## Features

* WAV audio loading (mono, 16 kHz) using **dr_wav**
* Speech emotion classification with **Wav2Vec2**
* Fast CPU inference via **ONNX Runtime**
* Clean CLI output:

  ```text
  Prediction: Anger
  Confidence: 0.823
  ```

---

## Project Structure

```text
SpeechEmotionRecogniser/
├── CMakeLists.txt
├── include/
│   ├── Audio/
│   │   └── AudioLoader.h
│   ├── Domain/
│   │   └── EmotionLabels.h
│   └── Inference/
│       └── EmotionRecognizer.h
├── src/
│   ├── main.cpp
│   ├── Audio/
│   │   └── AudioLoader.cpp
│   └── Inference/
│       └── EmotionRecognizer.cpp
├── models/
│   └── model_fp16.onnx
├── data/
│   └── samples/
│       └── anger.wav
├── third_party/
│   ├── dr_wav/
│   │   └── dr_wav.h
│   └── onnxruntime/
│       ├── include/
│       └── lib/
└── build/
```

---

## Model

This project uses:

**`prithivMLmods/Speech-Emotion-Classification-ONNX`**
A Wav2Vec2-based speech emotion classifier fine-tuned on emotional speech data.

### Supported Emotions

| Index | Label     |
| ----: | --------- |
|     0 | Anger     |
|     1 | Calm      |
|     2 | Disgust   |
|     3 | Fear      |
|     4 | Happy     |
|     5 | Neutral   |
|     6 | Sad       |
|     7 | Surprised |

Defined in:

```cpp
include/Domain/EmotionLabels.h
```

---

## Important: Model Format

**Use the FP16 model**:

```
model_fp16.onnx
```

**Do NOT use**:

```
model_int8.onnx
```

### Why?

The default Windows ONNX Runtime CPU build **does not support `ConvInteger`** used in INT8 models.
Using `model_int8.onnx` will cause runtime errors.

---

## Dependencies

### Required

* **CMake ≥ 3.20**
* **ONNX Runtime (Windows x64)**
* **dr_wav** (single-header WAV loader)

### Included via `third_party/`

* `dr_wav/dr_wav.h`

### ONNX Runtime (Required)

You **must** provide ONNX Runtime yourself. This project does **not** fetch it automatically.

You have two supported options:

**Option A (Recommended): Vendor ONNX Runtime in `third_party/`**

```
third_party/onnxruntime/
├── include/   (onnxruntime_cxx_api.h, etc.)
└── lib/       (onnxruntime.lib)
```

This is the layout expected by `CMakeLists.txt`.

**Option B: System-wide ONNX Runtime install**
If ONNX Runtime is installed elsewhere, you must:

* Add its `include/` directory to `target_include_directories`
* Link against `onnxruntime.lib` in `target_link_libraries`

If ONNX Runtime headers or libraries are missing, the build will fail with errors such as:

```
cannot open source file "onnxruntime_cxx_api.h"
```

---

## Build Instructions (Windows)

### Clone the repository

```bash
git clone <your-repo-url>
cd SpeechEmotionRecogniser
```

### Create build directory

```bash
cmake -S . -B build
```

### Build (Release)

```bash
cmake --build build --config Release
```

---

## Running the Program

From the `build` directory:

```bash
Release\SpeechEmotionRecogniser.exe ..\data\samples\anger.wav
```

### Example Output

```text
Audio loaded successfully!
Sample rate: 16000
Frames: 100533

Prediction: Anger
Confidence: 0.812
```

---

## Audio Requirements

* WAV format
* Mono (auto-converted if stereo)
* 16,000 Hz sample rate (required by Wav2Vec2)
* Float PCM internally

Handled automatically by `AudioLoader`.

---

## Implementation Overview

### Audio Loading

* Uses **dr_wav**
* Converts stereo → mono
* Outputs `std::vector<float>` + sample rate

### Inference

* ONNX Runtime C++ API
* Single shared `Ort::Env`
* Dynamic waveform length
* Outputs raw logits

### Post-processing

* Softmax applied in C++
* Highest probability selected

---

## Design Notes

* INT8 quantized models are **not supported** by default ORT CPU builds
* FP32 Wav2Vec2 runs near real-time on modern CPUs
* Designed for clarity and debuggability over premature optimization

---

## Credits

* **Hugging Face** – Wav2Vec2
* **prithivMLmods** – Speech Emotion ONNX model
* **ONNX Runtime Team**
* **dr_wav** by David Reid
