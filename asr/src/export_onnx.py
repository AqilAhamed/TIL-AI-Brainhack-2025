from transformers import Wav2Vec2ForCTC
import torch
import os

model_dir = "/home/jupyter/turings-brainchildren/asr/finetuned-wav2vec2"
onnx_path = os.path.join(model_dir, "model.onnx")

model = Wav2Vec2ForCTC.from_pretrained(model_dir).eval()

dummy = torch.randn(1, 16000)  # 1 sample, 16000 timesteps (1 second audio @ 16kHz)
torch.onnx.export(
    model,
    dummy,
    onnx_path,
    input_names=["input_values"],
    output_names=["logits"],
    dynamic_axes={"input_values": {1: "time"}, "logits": {1: "time"}},
    opset_version=14
)

print(f"Exported to {onnx_path}")
