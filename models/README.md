# Speech Emotion Classification ONNX Model

Model: prithivMLmods/Speech-Emotion-Classification-ONNX  
Architecture: Wav2Vec2ForSequenceClassification  
Sample Rate: 16 kHz  
Channels: Mono  

## Input
Name: input_values  
Shape: [1, N]  
Type: float32  
Description: Raw waveform normalized to [-1, 1]

## Output
Shape: [1, 8]  
Type: float32  
Description: Emotion logits (apply softmax)

## Labels
0 Anger  
1 Calm  
2 Disgust  
3 Fear  
4 Happy  
5 Neutral  
6 Sad  
7 Surprised
