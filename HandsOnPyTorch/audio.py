# import packages and fetch data

import os

import matplotlib
import matplotlib.pyplot as plt
import requests
import torch
from transformers import pipeline
import numpy as np
import torchaudio
import pygame
from CTCDecoder import GreedyCTCDecoder
import soundfile as sf


def get_audio():
    SPEECH_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"  # noqa: E501
    SPEECH_FILE = "_assets/speech.wav"

    if not os.path.exists(SPEECH_FILE):
        os.makedirs("_assets", exist_ok=True)
        with open(SPEECH_FILE, "wb") as file:
            file.write(requests.get(SPEECH_URL).content)

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    print("Sample Rate:", bundle.sample_rate)
    print("Labels:", bundle.get_labels())

    return SPEECH_FILE, bundle

def get_audio_from_text(speech_text, device):
    tts_pipeline = pipeline("text-to-speech", model="facebook/mms-tts-rus", device=device)
    speech = tts_pipeline(speech_text)

    model_config = tts_pipeline.model.config
    sample_rate = model_config.sampling_rate if hasattr(model_config, 'sampling_rate') else None
    if sample_rate is not None:
        print("Sample Rate from model config:", sample_rate)
    else:
        print("Sample Rate is not available in the model config.")

    if "audio" in speech:
        print(speech)
        audio = speech["audio"]  # Adjust this based on the output format
    else:
        raise KeyError("No 'audio' key found in the pipeline output.")

    # Convert audio data to a NumPy array if needed
    audio_np = np.array(audio)

    # Ensure the audio is in the correct shape
    if audio_np.ndim == 1:  # If it's mono
        audio_np = audio_np[np.newaxis, :]  # Add a new axis for the channel

    SPEECH_FILE = "_assets/speech.wav"

    # Save the audio to a .wav file
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    print("Sample Rate:", bundle.sample_rate)
    print("Labels:", bundle.get_labels())

    #sf.write(SPEECH_FILE, audio, samplerate=sample_rate)
    sf.write(SPEECH_FILE, audio_np.T, samplerate=bundle.sample_rate)


    return SPEECH_FILE, bundle

def play_audio(play_file):

    pygame.mixer.init()
    pygame.mixer.music.load(play_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def features_audio(bundle, speech_file):
    waveform, sample_rate = torchaudio.load(speech_file, format="wav")
    waveform = waveform.to(device)

    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

    model = bundle.get_model().to(device)
    print(model.__class__)
    with torch.inference_mode():
        features, _ = model.extract_features(waveform)

    return features, model

def visualize_emission(emission):

    plt.imshow(emission[0].cpu().T)
    plt.title("Classification result")
    plt.xlabel("Frame (time-axis)")
    plt.ylabel("Class")
    plt.show()
    #print("Class labels:", bundle.get_labels())

def visualize_features(features):
    fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
    for i, feats in enumerate(features):
        ax[i].imshow(feats[0].cpu())
        ax[i].set_title(f"Feature from transformer layer {i + 1}")
        ax[i].set_xlabel("Feature dimension")
        ax[i].set_ylabel("Frame (time-axis)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]
    print(f"Available backends: {torchaudio.list_audio_backends()}")
    torch.random.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(torch.__version__)
    print(torchaudio.__version__)
    print(device)

    #speech_file, bundle = get_audio()
    speech_file, bundle = get_audio_from_text("Какая-то Китайская серая птица", device)

    #play_audio(speech_file)

    features, model = features_audio(bundle, speech_file)

    #visualize_features(features)

    waveform, sample_rate = torchaudio.load(speech_file, format="wav")
    waveform = waveform.to(device)
    with torch.inference_mode():
        emission, _ = model(waveform)

    #visualize_emission(emission)

    russian_labels = [
        "а", "б", "в", "г", "д", "е", "ё", "ж", "з", "и", "й", "к", "л", "м",
        "н", "о", "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ",
        "ъ", "ы", "ь", "э", "ю", "я", " "
    ]
    blank_index = russian_labels.index(" ")
    decoder = GreedyCTCDecoder(labels=russian_labels, blank=blank_index)
    transcript = decoder(emission[0])
    print(transcript)
    play_audio(speech_file)