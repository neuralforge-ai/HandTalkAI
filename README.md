# HandTalkAI: ASL Alphabet Recognition

HandTalkAI recognizes American Sign Language (ASL) alphabet letters (A to Z) from a single photo of a hand. The project compares three models and shows why a model that scores 99% on its own dataset can still fail on a new person, and how a landmark-based model reduces that problem.

**Live demo (Streamlit):** `https://handtalkai-vgfxmycq2qbs9emf3wbbpa.streamlit.app/`
**Flask interface:** see `notebooks/Flask_App.ipynb`

---

## 1. The problem

Image models trained on the Kaggle ASL Alphabet dataset reach about 99% accuracy on that dataset. But the dataset has one signer, one background and one lighting setup. When tested on photos of different people, the models predicted wrongly.

**Cause: domain gap.** The models learned skin tone, background and lighting, not only hand shape.

## 2. The approach

| # | Model | Input | Idea |
|---|-------|-------|------|
| A | Baseline CNN | 64x64 full image | 3 Conv+Pool blocks, Dense, Dropout, Softmax |
| C | MobileNetV2 (transfer learning) | 128x128 hand crop (MediaPipe) | Pretrained network, BatchNorm, Dropout, augmentation, two-phase training (frozen, then fine-tuned) |
| E | **Landmark ANN (the fix)** | 63 numbers (21 hand points x 3) | MediaPipe finds 21 points on the hand; they are normalized for position and size, so only hand *shape* reaches the network |

### Why landmarks help
Landmarks are normalized so the wrist is the origin and the hand size is scaled to 1. Skin colour, background, lighting and camera distance never reach the model.

```
Photo -> MediaPipe (21 points) -> normalize -> 63 numbers -> ANN -> letter
```

### ANN architecture
`Input(63) -> Dense(256) + BatchNorm + Dropout(0.3) -> Dense(128) + Dropout(0.3) -> Dense(64) -> Dense(26, softmax)`

## 3. Data

- **Kaggle ASL Alphabet** (`grassknoted/asl-alphabet`): up to 1,000 images per letter.
- **Own photos:** 3 people (Person_1, Person_2, Person_3), about 242 photos covering A to Z. Each shot was saved in several versions (with label, hand only, square enhanced).
- Photos where no hand was detected are skipped automatically.

## 4. Results

Accuracy on the dataset itself is **not** the real test. The real test is a person the model has never seen.

| Model | Dataset accuracy | New person accuracy |
|-------|------------------|---------------------|
| Baseline CNN | 99.5% | _fill from Part D_ |
| MobileNetV2 (fine-tuned) | 98.1% | _fill from Part D_ |
| Landmark ANN, Kaggle only | n/a | 44.5% |
| Landmark ANN, Kaggle + other people | n/a | 64.6% |

Adding photos from other signers raised accuracy on an unseen person from 44.5% to 64.6%. This is leave-one-person-out evaluation: the test person is never used in training.

> Replace the placeholders and check every number against your notebook output before submitting.

## 5. Repository structure

```
handtalkai/
├── app.py                    # Streamlit app
├── requirements.txt
├── packages.txt
├── handtalk_ann.npz          # trained ANN weights
├── hand_landmarker.task      # MediaPipe hand model
├── README.md
├── notebooks/
│   ├── File_1_Image_Models.ipynb     # Parts A to C (CNN, MediaPipe crop, MobileNetV2)
│   ├── File_2_Landmark_ANN.ipynb     # Part E (landmark ANN)
│   └── Flask_App.ipynb               # Flask web interface
└── docs/
    └── Project_Report.pdf
```

## 6. How to run

### Notebooks (Google Colab)
1. Open a notebook in Google Colab (set the runtime to GPU for File 1).
2. Run the cells from top to bottom.
3. A Kaggle account is needed to download the dataset. **Never upload `kaggle.json` to GitHub.**
4. Download your trained models and ZIP backups, because Colab storage resets.

### Flask interface
Open `notebooks/Flask_App.ipynb` in Colab, run the cells in order, then open the app on port 5000. Upload a photo or use the camera to see the predicted letter, its confidence and the top 3 guesses.

### Streamlit app (local)
```bash
pip install -r requirements.txt
streamlit run app.py
```
The app needs `handtalk_ann.npz` and `hand_landmarker.task` in the same folder as `app.py`.

## 7. Limitations

- **J and Z need motion.** A single photo cannot capture them, so expect weak results (video would be needed).
- **Look-alike letters** such as M, N, S, T and E are often confused, because the fingers are hidden in a fist.
- **Few signers.** Only 3 people, and each shot exists in several copies, so there are about 26 truly independent shots per person.
- **Hand detection can fail** on tight crops, unusual angles or very blurry photos; those photos are skipped.
- **Camera angle and tilt** differ between people and can shift the landmarks.

## 8. Future work

- Collect photos from many more signers, angles and lighting conditions.
- LSTM or other temporal model on video to handle J and Z.
- Per-user calibration ("calibrate for me") in the Streamlit app.
- Real-time recognition from a webcam stream.

## 9. Tech stack

Python, TensorFlow/Keras, MediaPipe, OpenCV, NumPy, scikit-learn, Matplotlib, Flask, Streamlit, Google Colab.

## 10. Acknowledgements

- Dataset: [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) by grassknoted (Kaggle).
- Hand landmarks: [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) by Google.

## 11. Author

`Amna Khalid` | `BS (AI)` | `SZABIST`
