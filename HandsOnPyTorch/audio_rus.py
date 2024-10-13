from huggingsound import SpeechRecognitionModel

model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
audio_paths = ["_assets/speech.wav"]

transcriptions = model.transcribe(audio_paths)
print(transcriptions[0])