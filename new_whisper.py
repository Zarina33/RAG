from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import pandas as pd
import argparse
import csv
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd
from pydub import AudioSegment
from datasets import Dataset, Features, Value, Audio
import wandb, os


parser = argparse.ArgumentParser(description='Fine-tuning script for Whisper Models of various sizes.')

parser.add_argument(
    '--num_proc', 
    type=int, 
    required=False, 
    default=7, 
    help='Number of parallel jobs to run. Helps parallelize the dataset prep stage.'
)
args = parser.parse_args()
gradient_checkpointing = True


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    """
    Define evaluation metrics. We will use the Word Error Rate (WER) metric.
    For more information, check:
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}



# STEP 1. Download Dataset
def process_audio(row):
    audio_path = row['path']
    processed_audio_path = f'datasets/wavs/{audio_path}.wav'
    #print(processed_audio_path)
    return processed_audio_path

from datasets import Dataset
import pandas as pd
from pydub import AudioSegment
from datasets import Dataset, Features, Value, Audio

# Load your TSV data into a pandas DataFrame
test_tsv_file_path = 'datasets/data/test.tsv'
train_tsv_file_path = 'datasets/data/train.tsv'

test_df = pd.read_csv(test_tsv_file_path, delimiter='\t')
train_df = pd.read_csv(train_tsv_file_path, delimiter='\t')

test_df['path'] = test_df.apply(process_audio, axis=1)
train_df['path'] = train_df.apply(process_audio, axis=1)

# Define the new features based on the transformed data
new_features = Features({
    'sentence': Value(dtype='string', id=None),
    'path': Audio(sampling_rate=16000, mono=True, decode=True, id=None),
})

# Create a new dataset with the updated features
test_dataset = Dataset.from_pandas(test_df, features=new_features)
train_dataset = Dataset.from_pandas(train_df, features=new_features)

#print(train_dataset["path"][0])
# - - - - - - - - - - - - - - - - - - - - - |
# STEP 2. Prepare: Feature Extractor, Tokenizer and Data
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer

# - Load Feature extractor: WhisperFeatureExtractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

# - Load Tokenizer: WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="kazakh", task="transcribe")

# - - - - - - - - - - - - - - - - - - - - - |
# STEP 3. Combine elements with WhisperProcessor
from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="kazakh", task="transcribe")

# - - - - - - - - - - - - - - - - - - - - - |
# STEP 4. Prepare Data
def prepare_dataset(batch):
    """
    Prepare audio data to be suitable for Whisper AI model.
    """
    # (1) load and resample audio data from 48 to 16kHz
    audio = batch["path"]

    # (2) compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # (3) encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    #print(batch)
    return batch


# - - - - - - - - - - - - - - - - - - - - - |
# STEP 5. Training and evaluation
# STEP 5.1. Initialize the Data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# STEP 5.1. Define evaluation metric
import evaluate
metric = evaluate.load("wer")

# STEP 5.3. Load a pre-trained Checkpoint
from transformers import WhisperForConditionalGeneration
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")


model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

if gradient_checkpointing:
    model.config.use_cache = False

max_label_length = model.config.max_length
min_input_length = 0.0
max_input_length = 30.0
def is_in_length_range(length, labels):
    return min_input_length < length < max_input_length and 0 < len(labels) < max_label_length

train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names, num_proc=args.num_proc)
test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names, num_proc=args.num_proc)

train_dataset = train_dataset.filter(
    is_in_length_range,
    input_columns=["input_length", "labels"],
    num_proc=args.num_proc,
) 

test_dataset = test_dataset.filter(
    is_in_length_range,
    input_columns=["input_length", "labels"],
    num_proc=args.num_proc,
) 
# STEP 5.4. Define the training configuration
"""
Check for Seq2SeqTrainingArguments here:
https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments
"""
from transformers import Seq2SeqTrainingArguments

wandb.login()
wandb_project = "kg-finetune"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project
    
from datetime import datetime

project = "new-kg-finetune"
base_model_name = "whisper"
run_name = base_model_name + "-" + project



training_args = Seq2SeqTrainingArguments(
    output_dir="./new-whisper-small-ky",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=gradient_checkpointing,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=1000,
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False, # testing
)


# Initialize a trainer.
"""
Forward the training arguments to the Hugging Face trainer along with our model,
dataset, data collator and compute_metrics function.
"""
from transformers import Seq2SeqTrainer


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# Save processor object before starting training
processor.save_pretrained(training_args.output_dir)

# STEP 5.5. Training
"""
Training will take appr. 5-10 hours depending on your GPU.
"""

print('Training is started.')
trainer.train()  # <-- !!! Here the training starting !!!
print('Training is finished.')



