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
    """
    Wrapper for emotion analysis model
    """
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
        """
        Args:
            checkpoint (str, optional): Model's name (hf repo id). Defaults to 'bhadresh-savani/roberta-base-emotion'.
            window_size (int, optional): Amount of phrases in window. Defaults to 512.
            stride (int, optional): Overlap in windows. Defaults to 256.
            batch_size (int, optional): batch size. Defaults to 16.
            use_amp (bool, optional): Flag to use AMP. Defaults to True.
            adapter_path (str | None, optional): Path to quantized model (if was trained with PEFT). Defaults to None.
            num_emotions (int | None, optional): Number of emotions in model's output. Defaults to None.
            problem_type (str, optional): Model's task. Defaults to 'multi_label_classification'.
        """
        self.checkpoint = checkpoint

        if adapter_path is not None:
            # If the model was trained with PEFT and saved in this format
            # we need to explicitly tell HuggingFace API about it

            # Quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )

            # Basic initialization
            base_model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint,
                quantization_config=bnb_config,
                num_labels=num_emotions,
                problem_type=problem_type
            )
            self.model = PeftModel.from_pretrained(base_model, adapter_path) # Peft
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

        # Default labels
        self.EMOTION_LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']


    def _create_windows(self, inputs_ids: torch.Tensor) -> list[torch.Tensor]:
        """
        Creates windows from tokenized text

        Args:
            inputs_ids (torch.Tensor): tensor with tokens

        Returns:
            list[torch.Tensor]: windows
        """
        windows = []

        for i in range(0, len(inputs_ids), self.stride):
            window = inputs_ids[i : i + self.window_size]
            windows.append(window)

        return windows


    def _process_batch(self, batch_windows: list[torch.Tensor]) -> np.ndarray:
        """
        Processes windows in batches

        Args:
            batch_windows (list[torch.Tensor]): batch with windows

        Returns:
            np.ndarray: embeddings
        """
        batch = torch.nn.utils.rnn.pad_sequence( # padding
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

            # Depending on the task select appropriate function.
            # For example, for multilabel clf sigmoid is required,
            # but for single label extraction softmax must be used
            if self.use_sigmoid:
                probs = torch.sigmoid(outputs.logits)
            else:
                probs = torch.softmax(outputs.logits, dim=-1)

        return probs.cpu().numpy()


    def analyze_text(self, text: str) -> pd.DataFrame:
        """
        Tokenizes text, splits it into overlapping windows and
        analyzes them

        Args:
            text (str): subtitles

        Returns:
            pd.DataFrame: dataframe with embeddings
        """
        tokens = self.tokenizer(text, return_tensors='pt', truncation=False) # tokenize and cast to torch tensor
        input_ids = tokens['input_ids'][0] # select only tokens' ids

        windows = self._create_windows(input_ids)

        if not windows:
            return pd.DataFrame(columns=self.EMOTION_LABELS)

        all_emotions = []

        # process batches
        for i in range(0, len(windows), self.batch_size):
            batch_windows = windows[i:i + self.batch_size]
            batch_emotions = self._process_batch(batch_windows)
            all_emotions.extend(batch_emotions)

        # Create dataframe
        df = pd.DataFrame(all_emotions, columns=self.EMOTION_LABELS)

        df['window_id'] = range(len(df))
        df['window_start'] = [i * self.stride for i in range(len(df))]
        df['window_end'] = [min((i * self.stride) + self.window_size, len(input_ids))
                           for i in range(len(df))]

        return df


    def analyze_file(self, input_path: Path) -> None:
        """
        Analyzes a single file

        Args:
            input_path (Path): path to file with subtitles
        """
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
        """
        Selects all the files with subtitles and analyzes them
        """
        for filepath in glob.glob(
            os.path.join(settings.emotion_analyzer.input_path, '*.txt')
        ):
            self.analyze_file(Path(filepath))
