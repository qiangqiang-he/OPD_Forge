"""Default OPD_Forge dataset adapter for question/answer files."""

from __future__ import annotations

from pathlib import Path

import datasets
import numpy as np

from verl.utils.dataset.rl_dataset import RLHFDataset
from utils.prompts import render_prompt


class CustomDataset(RLHFDataset):
    """Read ``[{question, answer}, ...]`` files directly.

    The conversion to verl's internal fields happens in memory. Source JSON
    files remain unchanged and are still the only datasets users maintain.
    """

    def _build_messages(self, example: dict, key: str):
        """Return the already-formatted prompt instead of rebuilding chat messages."""
        prompt = example[key]
        if not isinstance(prompt, str):
            raise TypeError(f"CustomDataset expected a complete prompt string, got {type(prompt).__name__}.")
        return prompt

    def _read_files_and_tokenize(self) -> None:
        parts = []
        for data_file in self.data_files:
            if data_file.endswith(".json") or data_file.endswith(".jsonl"):
                part = datasets.load_dataset("json", data_files=data_file)["train"]
            elif data_file.endswith(".parquet"):
                part = datasets.load_dataset("parquet", data_files=data_file)["train"]
            else:
                raise ValueError(f"Unsupported question/answer dataset file: {data_file}")
            missing = {"question", "answer"}.difference(part.column_names)
            if missing:
                raise ValueError(
                    f"CustomDataset file {data_file} is missing required columns: {sorted(missing)}"
                )
            # Different math datasets frequently encode answers as strings,
            # integers, or floats. Normalize before concatenation so Arrow
            # does not reject otherwise compatible dataset shards.
            part = part.cast_column("question", datasets.Value("string"))
            part = part.cast_column("answer", datasets.Value("string"))
            part = part.add_column("_data_source", [Path(data_file).stem] * len(part))
            parts.append(part)

        dataframe = datasets.concatenate_datasets(parts)

        student_prompt = str(self.config.student_prompt)
        teacher_prompt = str(self.config.teacher_prompt)

        def adapt(example: dict, index: int) -> dict:
            question = str(example["question"])
            answer = str(example["answer"])
            return {
                "prompt": render_prompt(student_prompt, question=question, answer=answer),
                "teacher_prompt_text": render_prompt(teacher_prompt, question=question, answer=answer),
                # PS-OPD constructs its privileged prompt only after the
                # student's rollout (and candidate answer) is available.
                "ps_question": question,
                "ps_ground_truth_answer": answer,
                "data_source": str(example["_data_source"]),
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "index": int(index),
                    "question": question,
                    "answer": answer,
                },
            }

        dataframe = dataframe.map(adapt, with_indices=True, desc="Adapting question/answer JSON")
        total = len(dataframe)
        print(f"CustomDataset len: {total}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rng = np.random.default_rng(self.seed)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            dataframe = dataframe.select(indices.tolist())
            print(f"CustomDataset selected {self.max_samples} of {total} samples")

        self.dataframe = self.maybe_filter_out_long_prompts(dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        """Filter complete prompt strings without applying a chat template."""
        if not self.filter_overlong_prompts:
            return dataframe

        tokenizer = self.tokenizer
        prompt_key = self.prompt_key

        def prompt_fits(example: dict) -> bool:
            prompt = example[prompt_key]
            if not isinstance(prompt, str):
                raise TypeError(f"CustomDataset expected a complete prompt string, got {type(prompt).__name__}.")
            return len(tokenizer.encode(prompt, add_special_tokens=False)) <= self.max_prompt_length

        dataframe = dataframe.filter(
            prompt_fits,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
        )
        print(f"CustomDataset filtered len: {len(dataframe)}")
        return dataframe
