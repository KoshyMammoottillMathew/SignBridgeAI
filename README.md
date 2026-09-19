# SignBridge AI

**Real-Time ASL Fingerspelling-to-Text Recognition Using Deep Learning**

SignBridge AI is an accessibility-focused deep-learning project that recognizes continuous **American Sign Language (ASL) fingerspelling** from video and converts it into text.

> **Scope:** SignBridge AI recognizes ASL fingerspelling. It is not a full ASL language-translation system.

## System Pipeline

```text
Video / Webcam
      ↓
Frame Sampling
      ↓
MediaPipe Hand Detection
      ↓
21 Hand Landmarks × (x, y, z)
      ↓
63 Features per Frame
      ↓
Wrist-Centering + Scale Normalization
      ↓
Temporal Sequence (T × 63)
      ↓
Temporal 1D CNN
      ↓
Bidirectional GRU
      ↓
Dense Character Logits
      ↓
CTC Decoding
      ↓
Predicted Text
```

## Model Architecture

The final model combines Temporal Conv1D layers for local motion patterns, batch normalization and dropout, a residual temporal convolution block, two Bidirectional GRU layers for longer sequence context, a 58-output character layer (57 character classes + CTC blank), and Connectionist Temporal Classification (CTC) for sequence learning without frame-level character alignment.

The model contains approximately **2.34 million parameters**.

## Dataset

The project uses **FSBoard**, a large-scale ASL fingerspelling dataset collected from smartphone video. Dataset videos and processed landmark caches are **not included in this repository**. Users should obtain the dataset through its official distribution channel and comply with its licensing and access requirements.

For the documented project stage, 1,306 clips were processed into compact MediaPipe landmark sequences:

- Training: 1,122
- Validation: 140
- Test: 44
- Compressed landmark data: 11.62 MB

## Preprocessing

Each sampled frame is processed with MediaPipe Hand Landmarker. The 21 detected hand landmarks provide x, y and z coordinates, producing **63 numerical features per frame**.

Landmarks are normalized by wrist-centering and scaling using the wrist-to-middle-finger MCP distance. The resulting temporal sequence is supplied to the neural network without temporal downsampling.

## Evaluation

Character Error Rate (CER) is the primary evaluation metric because SignBridge AI performs sequence transcription rather than single-label classification.

Documented results from the later training/evaluation stage:

- Best validation CER: **16.32%** at Epoch 7
- Early stopping: after Epoch 11
- Later cached FSBoard test-split CER: **15.96%**

These figures describe the documented experimental setup and should not be interpreted as a direct comparison with results obtained using different FSBoard subsets or protocols.

## SignBridge Live

The project includes an interactive **Gradio** prototype for webcam-based fingerspelling recognition. A rolling temporal buffer supplies sequences of hand landmarks to the trained model for interactive prediction.

The current implementation should be considered a **near-real-time prototype**; a formal end-to-end latency/FPS benchmark has not been performed.

## Repository Structure

```text
SignBridgeAI/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── SignBridge_AI_Training_and_Live_Demo.ipynb
├── src/
├── demo/
├── assets/
└── docs/
```

## Main Technologies

Python · TensorFlow/Keras · MediaPipe · OpenCV · NumPy · Pandas · Gradio · Google Colab

## Security and Credentials

Credentials such as the Kaggle API token should be supplied through **Google Colab Secrets or environment variables**. API keys, access tokens, dataset caches, trained-state folders, temporary videos and other private credentials/data should never be committed.

## Limitations

Current limitations include use of a subset of FSBoard, single-hand landmark tracking, no handedness canonicalization, no face/body landmarks, difficulty with some repeated-character sequences, and domain shift between FSBoard videos and live webcam input.

## Future Work

Potential improvements include training on more data and signers, handedness normalization, stronger decoding/language-model integration, improved repeated-character handling, data augmentation, multimodal landmarks, mobile deployment, text-to-speech output and broader sign-language recognition.

## Author

**Koshy Mammoottill Mathew**

Developed as part of the **Certificate Course in Data Science and Artificial Intelligence**.

## Acknowledgement

This project uses the FSBoard ASL fingerspelling dataset and open-source machine-learning and computer-vision libraries. Refer to the project report and original FSBoard publication for academic background and dataset attribution.
