"""Tests for rukh.publish.adapter: an adapter is not a model and its card must not pretend.

What is worth checking here is what the card is *allowed* to claim. An adapter published with the
base model's Elo and no mention of its own effect would be publishing somebody else's number, and
one published without naming the base model would be publishing a file nobody can use.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rukh.models import DecoderConfig, MoveDecoder
from rukh.models.lora import ADAPTER_CONFIG, ADAPTER_FILE, LoraConfig, apply_lora, save_adapter
from rukh.publish.adapter import AdapterEffect, adapter_params, publish_adapter
from rukh.publish.model import ModelPublishConfig

pytestmark = pytest.mark.unit

MEDIUM = {"n_layer": 16, "d_model": 768}


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    model = MoveDecoder(DecoderConfig(n_layer=2, n_head=2, d_model=16, block=8))
    cfg = LoraConfig(r=8, alpha=16, targets=("q", "v"))
    apply_lora(model, cfg)
    save_adapter(model, tmp_path / ADAPTER_FILE, cfg)
    return tmp_path


def _card(result: object) -> str:
    return Path(result.card_path).read_text(encoding="utf-8")  # type: ignore[attr-defined]


def test_the_card_leads_with_the_base_model(rukh_home: Path, run_dir: Path) -> None:
    """Without the exact base checkpoint the file is numbers with no model to correct."""
    result = publish_adapter(
        run_dir, "rukh-lora-e4", "rukh-medium-elo", base_config=MEDIUM, dry_run=True
    )
    card = _card(result)
    assert result.base_repo == "chorcat/rukh-medium-elo"
    assert "base_model: chorcat/rukh-medium-elo" in card  # the Hub reads this tag
    assert "chorcat/rukh-medium-elo" in card.split("# chorcat/rukh-lora-e4")[1][:400]


def test_the_card_counts_the_parameters_of_the_model_it_mounts_on(
    rukh_home: Path, run_dir: Path
) -> None:
    """The toy decoder of the test is not what ships; the card's arithmetic is the real model's."""
    result = publish_adapter(
        run_dir, "rukh-lora-e4", "rukh-medium-elo", base_config=MEDIUM, dry_run=True
    )
    assert result.params == 393_216  # 16 layers x 2 matrices x 2 x 8 x 768
    assert "393,216" in _card(result)


def test_adapter_params_is_the_formula_and_not_a_count_of_the_file() -> None:
    cfg = LoraConfig(r=8, alpha=16, targets=("q", "v"))
    assert adapter_params(cfg, 16, 768) == 393_216
    assert adapter_params(cfg, 12, 512) == 196_608  # `small`, for the same configuration


def test_an_unmeasured_adapter_says_so_instead_of_borrowing_a_number(
    rukh_home: Path, run_dir: Path
) -> None:
    result = publish_adapter(
        run_dir, "rukh-lora-e4", "rukh-medium-elo", base_config=MEDIUM, dry_run=True
    )
    card = _card(result)
    assert "Not measured yet." in card
    assert "Elo" not in card.split("## What it changed")[1].split("## How it was trained")[0]


def test_a_measured_adapter_publishes_what_it_changed_and_what_it_cost(
    rukh_home: Path, run_dir: Path
) -> None:
    """Both halves: the style it bought, and the strength it spent on it."""
    effect = AdapterEffect(
        first_move="e2e4",
        share_before=0.41,
        share_after=0.93,
        elo_base=1504.0,
        elo_adapted=1488.0,
        games=200,
        train_games=200_000,
        predicate="split_part(uci, ' ', 1) = 'e2e4'",
    )
    card = _card(
        publish_adapter(
            run_dir,
            "rukh-lora-e4",
            "rukh-medium-elo",
            base_config=MEDIUM,
            effect=effect,
            dry_run=True,
        )
    )
    assert "41.0 %" in card and "93.0 %" in card
    assert "1504" in card and "1488" in card
    assert "200,000 games" in card
    assert "split_part(uci, ' ', 1) = 'e2e4'" in card


def test_the_staged_folder_is_three_files_and_nothing_else(rukh_home: Path, run_dir: Path) -> None:
    """No weights, no ONNX, no vocabulary: an adapter repository is small on purpose."""
    result = publish_adapter(
        run_dir, "rukh-lora-e4", "rukh-medium-elo", base_config=MEDIUM, dry_run=True
    )
    folder = Path(result.folder)
    assert sorted(path.name for path in folder.iterdir()) == sorted(
        [ADAPTER_FILE, ADAPTER_CONFIG, "README.md"]
    )
    assert result.files == [ADAPTER_FILE, ADAPTER_CONFIG, "README.md"]


def test_the_published_config_names_the_base_model_too(rukh_home: Path, run_dir: Path) -> None:
    """The file that travels next to the weights, so loading does not depend on the card."""
    import json

    result = publish_adapter(
        run_dir, "rukh-lora-e4", "rukh-medium-elo", base_config=MEDIUM, dry_run=True
    )
    payload = json.loads((Path(result.folder) / ADAPTER_CONFIG).read_text(encoding="utf-8"))
    assert payload["base_model"] == "chorcat/rukh-medium-elo"
    assert payload["lora"]["r"] == 8
    assert payload["format"] == "rukh-lora-1"


def test_a_dry_run_touches_no_network(rukh_home: Path, run_dir: Path) -> None:
    from rukh.publish import adapter as module

    def explode() -> None:
        raise AssertionError("a dry run must not reach the Hub")

    original = module._api
    module._api = explode  # type: ignore[assignment]
    try:
        assert publish_adapter(
            run_dir, "rukh-lora-e4", "rukh-medium-elo", base_config=MEDIUM, dry_run=True
        ).dry_run
    finally:
        module._api = original  # type: ignore[assignment]


def test_a_missing_adapter_says_where_it_looked(rukh_home: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no adapter at"):
        publish_adapter(tmp_path, "rukh-lora-e4", "rukh-medium-elo", dry_run=True)


def test_the_owner_is_only_added_when_the_repo_has_none(rukh_home: Path, run_dir: Path) -> None:
    result = publish_adapter(
        run_dir,
        "someone/else",
        "another/base",
        cfg=ModelPublishConfig(owner="chorcat"),
        base_config=MEDIUM,
        dry_run=True,
    )
    assert (result.repo_id, result.base_repo) == ("someone/else", "another/base")


def _peft_run(tmp_path: Path, four_bit: bool = True) -> Path:
    """A folder shaped like what `rukh train qwen` leaves behind."""
    import json

    from safetensors.torch import save_file

    run = tmp_path / "qwen-run"
    run.mkdir(parents=True, exist_ok=True)
    save_file({"a": torch.zeros(4, 4)}, (run / "adapter_model.safetensors").as_posix())
    (run / ADAPTER_CONFIG).write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen3-0.6B",
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["v_proj", "q_proj", "k_proj", "o_proj"],
            }
        ),
        encoding="utf-8",
    )
    (run / "run.json").write_text(
        json.dumps(
            {
                "model": "Qwen/Qwen3-0.6B",
                "four_bit": four_bit,
                "trainable_params": 4_587_520,
                "total_params": 600_637_440,
                "train_samples": 100_000,
                "steps": 1500,
                "weights_memory_mb": 851.0,
                "peak_memory_mb": 1956.0,
            }
        ),
        encoding="utf-8",
    )
    return run


def test_the_qwen_adapter_is_published_in_the_format_peft_wrote(
    rukh_home: Path, tmp_path: Path
) -> None:
    """Re-staging it into our own format would break the two lines of `peft` a reader would type."""
    from rukh.publish.adapter import publish_qwen_adapter

    run = _peft_run(tmp_path)
    result = publish_qwen_adapter(run, "rukh-qwen3-pgn-qlora", dry_run=True)
    assert "adapter_model.safetensors" in result.files
    assert result.base_repo == "Qwen/Qwen3-0.6B"
    assert not (run / ADAPTER_FILE).exists()  # our format is not forced onto it


def test_the_qwen_card_quotes_the_run_that_happened_not_the_one_configured(
    rukh_home: Path, tmp_path: Path
) -> None:
    """`four_bit` in the card is what the run *used*, which is not always what it asked for."""
    from rukh.publish.adapter import publish_qwen_adapter

    card = Path(
        publish_qwen_adapter(_peft_run(tmp_path, four_bit=True), "r", dry_run=True).card_path
    ).read_text(encoding="utf-8")
    assert "4-bit NF4 (QLoRA)" in card
    assert "851 MB" in card and "1956 MB" in card
    assert "4,587,520" in card and "0.764 %" in card

    fell_back = Path(
        publish_qwen_adapter(
            _peft_run(tmp_path / "second", four_bit=False), "r", dry_run=True
        ).card_path
    ).read_text(encoding="utf-8")
    assert "bfloat16" in fell_back
    assert "QLoRA" not in fell_back.split("## How it was trained")[1].split("|")[4]


def test_an_unevaluated_qwen_adapter_says_so(rukh_home: Path, tmp_path: Path) -> None:
    from rukh.publish.adapter import publish_qwen_adapter

    card = Path(publish_qwen_adapter(_peft_run(tmp_path), "r", dry_run=True).card_path).read_text(
        encoding="utf-8"
    )
    assert "Not evaluated yet." in card


def test_a_folder_without_a_peft_adapter_says_where_it_looked(
    rukh_home: Path, tmp_path: Path
) -> None:
    from rukh.publish.adapter import publish_qwen_adapter

    with pytest.raises(FileNotFoundError, match="no peft adapter at"):
        publish_qwen_adapter(tmp_path, "r", dry_run=True)
