from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import glob
import numpy as np
import pandas as pd
import time
from pathlib import Path
from torch.cuda.amp import autocast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "bhadresh-savani/roberta-base-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)
model.eval()


def sliding_window_emotion(text, window_size=512, stride=256, batch_size=16):
    tokens = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = tokens['input_ids'][0]

    windows = []
    for i in range(0, len(input_ids), stride):
        window = input_ids[i:i + window_size]
        windows.append(window)

    if not windows:
        return []

    emotions = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch_windows = windows[i:i + batch_size]
            batch = torch.nn.utils.rnn.pad_sequence(
                batch_windows,
                batch_first=True,
                padding_value=tokenizer.pad_token_id
            ).to(device)

            if device.type == 'cuda':
                with autocast():
                    outputs = model(batch)
                    probs = torch.softmax(outputs.logits, dim=-1)
            else:
                outputs = model(batch)
                probs = torch.softmax(outputs.logits, dim=-1)

            emotions.extend(probs.cpu().numpy())

    return emotions


for filepath in glob.glob('subtitles/*.txt'):
    with open(filepath) as file:
        subtitle = file.read()
        current = time.time()
        result = sliding_window_emotion(subtitle)
        print(f"time: {time.time() - current}, windows size:{len(result)}")
        print(result)
        df = pd.DataFrame(result, columns=['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'])
        path = Path(filepath)
        df.to_csv(f'embedings/{path.stem}', index=False)




