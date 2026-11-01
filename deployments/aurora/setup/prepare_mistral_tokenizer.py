#!/usr/bin/env python3
"""Prepare a Hugging Face tokenizer overlay for Mistral Small vLLM on Aurora."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def prepare_tokenizer(model_dir: Path, output_dir: Path) -> None:
    model_dir = model_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in ("tokenizer.json", "special_tokens_map.json"):
        source = model_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing tokenizer asset: {source}")
        link = output_dir / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(source)

    config_path = model_dir / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    chat_template = config.get("chat_template")
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError(f"Tokenizer config has no string chat_template: {config_path}")

    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "chat_template.jinja").write_text(chat_template, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="Local Mistral model snapshot")
    parser.add_argument("output_dir", type=Path, help="Directory for the tokenizer overlay")
    args = parser.parse_args()
    prepare_tokenizer(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
