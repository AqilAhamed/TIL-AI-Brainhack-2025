import numpy as np
import onnxruntime as ort
from transformers import Wav2Vec2Processor
import torch
import jiwer

class ASRManager:
    def __init__(self, onnx_model_path: str = "./finetuned-wav2vec2/model.onnx", processor_path: str = "./finetuned-wav2vec2"):
        # Load ONNX model session with CUDA if available, else CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)

        # Load processor (tokenizer + feature extractor)
        self.processor = Wav2Vec2Processor.from_pretrained(processor_path)

        # Prepare text normalization pipeline for better output
        self.text_normalizer = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation()
        ])

    def preprocess(self, audio_array, sampling_rate=16000):
        # Process raw audio to input tensor expected by model
        inputs = self.processor(audio_array, sampling_rate=sampling_rate, return_tensors="np", padding=True)
        return inputs

    def infer(self, input_values):
        # Run ONNX inference
        ort_inputs = {self.session.get_inputs()[0].name: input_values}
        ort_outs = self.session.run(None, ort_inputs)
        logits = ort_outs[0]
        return logits

    def decode(self, logits):
        # Convert logits to predicted token IDs
        predicted_ids = np.argmax(logits, axis=-1)
        # Decode token IDs to string
        transcription = self.processor.batch_decode(predicted_ids)
        # Normalize the text output
        normalized = self.text_normalizer(transcription[0])
        return normalized

    def transcribe(self, audio_array, sampling_rate=16000):
        # Complete inference pipeline: preprocess, infer, decode
        inputs = self.preprocess(audio_array, sampling_rate)
        logits = self.infer(inputs['input_values'])
        transcription = self.decode(logits)
        return transcription
