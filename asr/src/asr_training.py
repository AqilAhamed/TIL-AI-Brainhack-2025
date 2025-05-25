import os
import logging
from datasets import load_dataset, Audio, DatasetDict
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer
from typing import List, Dict, Union
import torch

from transformers import DataCollatorCTCWithPadding

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "/home/jupyter/novice/asr"
OUTPUT_DIR = "/home/jupyter/turings-brainchildren/asr/finetuned-wav2vec2"

jsonl_path = os.path.join(DATA_DIR, "asr.jsonl")
dataset = load_dataset("json", data_files=jsonl_path, split="train")
dataset = dataset.map(lambda x: {"audio_filepath": os.path.join(DATA_DIR, x["audio"])})
dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))

model_name = "facebook/wav2vec2-large-960h-lv60-self"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(
    model_name,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

def prepare(batch):
    audio = batch["audio_filepath"]["array"]
    batch["input_values"] = processor(audio, sampling_rate=16000).input_values[0]
    batch["labels"] = processor(text=batch["transcript"]).input_ids
    return batch

dataset = dataset.map(prepare, remove_columns=["transcript", "key", "audio", "audio_filepath"])
splits = dataset.train_test_split(test_size=0.1, seed=42)
data_dict = DatasetDict({"train": splits["train"], "eval": splits["test"]})

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=15,
    logging_dir=f"{OUTPUT_DIR}/logs",
    learning_rate=1e-4,
    warmup_steps=500,
    fp16=True,
    save_total_limit=2,
    gradient_checkpointing=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data_dict["train"],
    eval_dataset=data_dict["eval"],
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model.to(device)
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    logger.info("Training complete.")
