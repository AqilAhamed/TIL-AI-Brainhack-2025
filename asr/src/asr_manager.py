"""
Manages ASR inference.
"""

import io
import torch
import soundfile as sf
import jiwer
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os

class ASRManager:
    def __init__(self):
        #model_name = "facebook/wav2vec2-large-960h-lv60-self"
        model_dir = os.getenv("MODEL_DIR", "/app/finetuned-wav2vec2")
        self.processor = Wav2Vec2Processor.from_pretrained(model_dir)
        self.model     = Wav2Vec2ForCTC.from_pretrained(model_dir)
        #self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        #self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.target_sampling_rate = self.processor.feature_extractor.sampling_rate

    def asr(self, audio_bytes: bytes) -> str:
        audio_input, sample_rate = sf.read(io.BytesIO(audio_bytes))
        if audio_input.ndim > 1:
            audio_input = audio_input.mean(axis=1)
        if sample_rate != self.target_sampling_rate:
            raise ValueError(f"Expected sample rate {self.target_sampling_rate}, got {sample_rate}")
        inputs = self.processor(audio_input, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        transforms = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.SubstituteRegexes({"-": " "}),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(),
        ])
        words = transforms(transcription)[0]
        return " ".join(words)
