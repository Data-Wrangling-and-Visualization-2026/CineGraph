import glob
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from settings import settings
from torch.amp import autocast
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)


class EmotionAnalyzer:
    def __init__(
        self,
        checkpoint: str = 'bhadresh-savani/roberta-base-emotion',
        window_size: int = 512,
        stride: int = 256,
        batch_size: int = 16,
        use_amp: bool = True,
        adapter_path: str | None = None,
        num_emotions: int | None = None,
        problem_type: str = 'multi_label_classification'
    ):
        self.checkpoint = checkpoint

        if adapter_path is not None:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )

            base_model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint,
                quantization_config=bnb_config,
                num_labels=num_emotions,
                problem_type=problem_type
            )
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
            self.use_sigmoid = True
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            self.use_sigmoid = False

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_amp = use_amp and self.device == 'cuda'
        self.model.to(self.device)
        self.model.eval()

        self.output_path = Path(settings.emotion_analyzer.output_path)

        self.EMOTION_LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']


    def _create_windows(self, inputs_ids: torch.Tensor) -> list[torch.Tensor]:
        windows = []

        for i in range(0, len(inputs_ids), self.stride):
            window = inputs_ids[i : i + self.window_size]
            windows.append(window)

        return windows

    def _process_batch(self, batch_windows: list[torch.Tensor]) -> np.ndarray:
        batch = torch.nn.utils.rnn.pad_sequence(
            batch_windows,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        ).to(self.device)

        attention_mask = (batch != self.tokenizer.pad_token_id).long()

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = self.model(batch, attention_mask=attention_mask)
            else:
                outputs = self.model(batch, attention_mask=attention_mask)

            if self.use_sigmoid:
                probs = torch.sigmoid(outputs.logits)
            else:
                probs = torch.softmax(outputs.logits, dim=-1)

        return probs.cpu().numpy()


    def analyze_text(self, text: str) -> pd.DataFrame:
        tokens = self.tokenizer(text, return_tensors='pt', truncation=False)
        input_ids = tokens['input_ids'][0]

        windows = self._create_windows(input_ids)

        if not windows:
            return pd.DataFrame(columns=self.EMOTION_LABELS)

        all_emotions = []

        for i in range(0, len(windows), self.batch_size):
            batch_windows = windows[i:i + self.batch_size]
            batch_emotions = self._process_batch(batch_windows)
            all_emotions.extend(batch_emotions)

        df = pd.DataFrame(all_emotions, columns=self.EMOTION_LABELS)

        df['window_id'] = range(len(df))
        df['window_start'] = [i * self.stride for i in range(len(df))]
        df['window_end'] = [min((i * self.stride) + self.window_size, len(input_ids))
                           for i in range(len(df))]

        return df


    def analyze_file(self, input_path: Path) -> None:
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()

        print(f'Processing {input_path.name}')
        start_time = time.time()

        df = self.analyze_text(text)

        self.output_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(
            os.path.join(
                self.output_path, f'{input_path.stem}.csv'
            ), index=False
        )

        elapsed = time.time() - start_time
        print(f'Saved {len(df)} windows to {self.output_path} in {elapsed:.2f}s')


    def analyze_data(self) -> None:
        for filepath in glob.glob(
            os.path.join(settings.emotion_analyzer.input_path, '*.txt')
        ):
            self.analyze_file(Path(filepath))
