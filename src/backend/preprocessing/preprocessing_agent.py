import glob
import os
import re
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

from langchain.agents import create_agent
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from settings import settings

# TODO: check dockerfile and docker compose yml


@tool
def remove_brackets_content(text: str) -> str:
    """
    Remove all content inside square brackets [],
    round brackets () and curly brackets {}.
    Useful for removing sound descriptions, speaker labels,
    stage directions like [applause], (laughs), {music}.
    """
    text = re.sub(r'\[.*?\]', '', text)   # [applause]
    text = re.sub(r'\(.*?\)', '', text)   # (laughs)
    text = re.sub(r'\{.*?\}', '', text)   # {music}
    return text.strip()


@tool
def remove_non_alphabetic(text: str) -> str:
    """
    Remove all non-alphabetic characters except spaces.
    Keeps only letters A-Z, a-z and whitespace.
    Useful for stripping punctuation, numbers, special symbols.
    """
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()


@tool
def remove_newlines(text: str) -> str:
    """
    Remove newline characters and replace them with spaces.
    Merges multi-line subtitle blocks into single lines.
    """
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = re.sub(r' +', ' ', text)  # collapse multiple spaces
    return text.strip()


@tool
def remove_dialog_punctuation(text: str) -> str:
    """
    Remove dialog-specific punctuation: dashes at line start (- text),
    ellipsis (...), double dashes (--), quotation marks,
    and excessive punctuation used in subtitles.
    """
    text = re.sub(r'^\s*-+\s*', '', text, flags=re.MULTILINE)  # leading dashes
    text = re.sub(r'\.{2,}', '', text)    # ellipsis ...
    text = re.sub(r'-{2,}', '', text)     # double dash --
    text = re.sub(r'["""\'\'\']+', '', text)  # quotes
    text = re.sub(r'[!?,;:]+', '', text)  # dialog punctuation
    return text.strip()


@tool
def remove_timestamps(text: str) -> str:
    """
    Remove SRT/VTT subtitle timestamps.
    Handles formats like:
    - 00:01:23,456 --> 00:01:25,789  (SRT)
    - 00:01:23.456 --> 00:01:25.789  (VTT)
    Also removes bare sequence numbers (1, 2, 3...) used in SRT files.
    """
    # SRT timestamps
    text = re.sub(
        r'\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}',
        '', text
    )
    # VTT cue identifiers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # WEBVTT header
    text = re.sub(r'WEBVTT.*\n?', '', text)
    return text.strip()


@tool
def remove_speaker_labels(text: str) -> str:
    """
    Remove speaker labels commonly found in subtitles.
    Handles formats like:
    - JOHN: text
    - John: text
    - [JOHN]: text
    - <v John> text  (VTT format)
    """
    text = re.sub(r'^[A-Z][A-Z\s]{1,20}:\s*', '', text, flags=re.MULTILINE)  # JOHN:
    text = re.sub(r'^\w[\w\s]{1,20}:\s*', '', text, flags=re.MULTILINE)       # John:
    text = re.sub(r'<v\s+[^>]+>', '', text)                                    # <v John>
    return text.strip()


@tool
def remove_html_tags(text: str) -> str:
    """
    Remove HTML/XML tags commonly found in subtitles.
    Handles: <i>, <b>, <u>, <font color="">, <c.colorname> etc.
    Used in SRT and VTT files for styling.
    """
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()


@tool
def normalize_whitespace(text: str) -> str:
    """
    Normalize all whitespace: collapse multiple spaces into one,
    strip leading/trailing spaces from each line,
    remove empty lines.
    Final cleanup step — use after all other tools.
    """
    lines = text.split('\n')
    lines = [re.sub(r' +', ' ', line).strip() for line in lines]
    lines = [line for line in lines if line]  # remove empty
    return ' '.join(lines)


@tool
def lowercase_text(text: str) -> str:
    """
    Convert all text to lowercase.
    Recommended for sentiment analysis preprocessing
    to ensure uniform token representation.
    """
    return text.lower()


@tool
def remove_filler_words(text: str) -> str:
    """
    Remove common spoken filler words that add noise for sentiment analysis.
    Removes: um, uh, hmm, ah, oh, er, erm, hm, gonna, wanna, gotta etc.
    """
    fillers = r'\b(um+|uh+|hmm+|hm+|ah+|oh+|er+|erm+|gonna|wanna|gotta|kinda|sorta|like|okay|ok|yeah|yep|nope)\b'
    text = re.sub(fillers, '', text, flags=re.IGNORECASE)
    text = re.sub(r' +', ' ', text)
    return text.strip()


class FormattedResponse(BaseModel):
    """Cleaned text"""
    CLEANED_TEXT: str = Field(description="The CLEANED subtitle text")


class PreprocessingAgent:

    def __init__(self):

        self.use_hugging_face = settings.preprocessor.use_hugging_face
        self.output_path = settings.preprocessor.output_path
        self.num_workers = cpu_count()

        self.prompt ="""Your goal is to clean raw subtitle text step by step using available tools.
            ## Input text format:
            The user will provide the subtitle text marked as SUBTITLE_TEXT.

            ## Recommended cleaning pipeline (follow this order):
            1. remove_timestamps         — strip SRT/VTT timing info
            2. remove_brackets_content   — remove [sound], (laughter), {{music}}
            3. remove_html_tags          — strip <i>, <b>, <font> tags
            4. remove_speaker_labels     — remove JOHN:, John:, <v John>
            5. remove_dialog_punctuation — remove ---, ..., quotes, !?;:,
            6. remove_newlines           — merge lines into single text
            7. remove_non_alphabetic     — keep only letters and spaces
            8. remove_filler_words       — remove um, uh, gonna, wanna...
            9. lowercase_text            — convert to lowercase
            10. normalize_whitespace     — final cleanup of spaces

            Apply ALL steps unless the user specifies otherwise.
            After cleaning, return the final cleaned text clearly labeled as:

            CLEANED_TEXT: <result>
            """


    def _setup_pipeline(self, tools: list):

        if self.use_hugging_face:
            endpoint = HuggingFaceEndpoint(
                repo_id=settings.preprocessor.model,
                task='text-generation',
                do_sample=False,
                temperature=0.05,
                max_new_tokens=16_000,
                streaming=False
            )

            llm = ChatHuggingFace(llm=endpoint)
            structured_llm = llm.with_structured_output(FormattedResponse, method='json_mode')
        else:
            llm = ChatLlamaCpp(
                model_path=settings.preprocessor.local_model_path,
                n_ctx=4096,
                n_gpu_layers=-1,
                n_batch=512,
                max_tokens=16_000,
                n_threads=cpu_count() - 1,
                temperature=0.05,
                verbose=True,
                f16_kv=True
            )
            structured_llm = llm.with_structured_output(FormattedResponse)

        agent = create_agent(llm, tools, system_prompt=SystemMessage(self.prompt))

        pipeline = (
            RunnableLambda(lambda text: {
                'messages': [HumanMessage(f'Clean the following text:\n\nSUBTITLE_TEXT:\n{text}')]
            })
            | agent
            | RunnableLambda(lambda result: f"Return this cleaned text in the required format:\n\n{result['messages'][-1].content}")
            | structured_llm
        )

        return pipeline


    def _init_tools(self) -> list:

        return [
            remove_timestamps,
            remove_brackets_content,
            remove_html_tags,
            remove_speaker_labels,
            remove_dialog_punctuation,
            remove_newlines,
            remove_non_alphabetic,
            remove_filler_words,
            lowercase_text,
            normalize_whitespace,
        ]


    def invoke(self, pipeline, splitter, text: str) -> str | None:

        texts = splitter.split_text(text)

        ready_parts = []
        for splitted_text in texts:
            response = pipeline.invoke(splitted_text)

            if self.use_hugging_face:
                for key in response.keys():
                    if 'text' in key.lower():
                        ready_parts.append(response[key].strip())
                        break
            else:
                ready_parts.append(response.CLEANED_TEXT.strip())

        if not len(ready_parts):
            return None

        return ' '.join(ready_parts).strip()


    def analyze_file(self, pipeline, splitter, input_path: Path) -> None:
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()

        print(f'Processing {input_path.name}')
        start_time = time.time()

        content = self.invoke(pipeline, splitter, text)

        self.output_path.mkdir(parents=True, exist_ok=True)

        with open(
            os.path.join(self.output_path, f'{input_path.stem}.csv'), "w"
        ) as file:
            file.write(content)

        elapsed = time.time() - start_time

        print(f'Saved to {self.output_path} in {elapsed:.2f}s')


    def _select_filenames(self) -> list[Path]:
        names = []

        for filepath in glob.glob(
            os.path.join(settings.preprocessor.input_path, '*.txt')
        ):
            names.append(Path(filepath))

        return names

    def _start_worker(self, files: list[Path], worker_id: int = 0) -> None:
        """
        Starts worker.

        Args:
            files (list[Path]): list of files to be preprocessed
        """
        print(f'Worker {worker_id} is preparing')

        tools = self._init_tools()
        pipeline = self._setup_pipeline(tools)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.preprocessor.chunk_size,
            chunk_overlap=0,
            length_function=len
        )

        print(f'Worker {worker_id} is starting preprocessing')

        for file in files:
            self.analyze_file(pipeline, splitter, file)

        print(f'Worker {worker_id} is finishing')


    def start_preprocessing(self) -> None:
        """
        Starts preprocessing.
        """
        offset = settings.scraper.offset
        files = self._select_filenames()

        if self.use_hugging_face:
            # Map files to workers
            chunk_size = (len(files) + self.num_workers - 1) // self.num_workers
            file_mapping = [
                files[i + offset : i + offset + chunk_size]
                for i in range(0, len(files), chunk_size)
            ]

            # Collect worker's args
            worker_args = [
                (files, i + 1)
                for i, files in enumerate(file_mapping)
            ]

            # Each worker - separate process
            with Pool(processes=self.num_workers) as p:
                p.starmap(self._start_worker, worker_args)
        else:
            files = files[offset:]
            self._start_worker(files, 1)