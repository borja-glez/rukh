"""What the export costs in Elo: fp32, fp16 and int8 of the same model on the same ladder.

Lab 2 of M6. M2 measured **parity** -- in how many positions each export picks the checkpoint's
move (fp16 99.9 %, int8 95.1 % for ``medium-v4``) -- and left the question that matters open: a
model that plays another move once in twenty games is another player, so how much weaker is it?
Parity cannot answer that; only playing can. This puts each ONNX file on the very ladder every
published Elo was measured on, through the same ``DecoderPlayer`` (same masking, sampling and
rescue), with the same rungs, games and seed, and writes one interval per precision next to the
parity that was already known:

    uv run python labs/m6/parity_cost.py --onnx artifacts/onnx/medium-v4 \
        --config configs/eval/greedy.yaml

ONNX Runtime plays on the CPU here, so the opponent's limit should be a node budget (the config's
``elo_nodes``): on a clock, a slower player would make the engine's side of the game different,
and the comparison would be between two Stockfishes, not between three exports.

Writes ``artifacts/eval/parity-cost/<name>.json`` (one ``EloResult`` per precision) and
``artifacts/web/parity-cost.json`` for the course figure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime

from rukh.config import load_yaml
from rukh.eval.elo import estimate, play_rungs
from rukh.eval.suite import EvalConfig
from rukh.infer import DecoderPlayer
from rukh.infer.onnx_player import OnnxDecoder
from rukh.paths import resolve
from rukh.tokenize.uci_vocab import UciTokenizer

PRECISIONS = {"fp32": "model.onnx", "fp16": "model-fp16.onnx", "int8": "model-int8.onnx"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--onnx", default="artifacts/onnx/medium-v4", help="Folder of the export.")
    parser.add_argument("--config", default="configs/eval/greedy.yaml")
    parser.add_argument("--games", type=int, default=None, help="Games per rung (config's).")
    parser.add_argument("--only", nargs="*", default=list(PRECISIONS), choices=list(PRECISIONS))
    parser.add_argument("--out", default="artifacts/eval/parity-cost")
    parser.add_argument("--web", default="artifacts/web/parity-cost.json")
    args = parser.parse_args()

    cfg = load_yaml(resolve(args.config), EvalConfig)
    games = args.games or cfg.elo_games
    folder = resolve(args.onnx)
    parity = json.loads((folder / "parity.json").read_text(encoding="utf-8"))
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = UciTokenizer()
    regime = f"{cfg.elo_nodes} nodes" if cfg.elo_nodes else f"{cfg.elo_move_time} s"
    print(f"{folder.name} · {games} games per rung · {len(cfg.elo_rungs)} rungs · {regime}\n")

    variants: list[dict[str, object]] = []
    for name in args.only:
        path = folder / PRECISIONS[name]
        if not path.is_file():
            print(f"{name:<6} missing ({path.name}), skipped", file=sys.stderr)
            continue
        agreement = parity["precisions"].get(name, {}).get("agreement")
        started = time.perf_counter()
        player = DecoderPlayer(OnnxDecoder(path), tok, cfg.sampling())  # type: ignore[arg-type]
        records = play_rungs(
            None,
            tok,
            cfg.elo_rungs,
            games,
            cfg.sampling(),
            move_time=cfg.elo_move_time,
            max_plies=cfg.elo_max_plies,
            header_elo=cfg.header_elo,
            player=player,
            nodes=cfg.elo_nodes,
        )
        result = estimate(records, samples=cfg.bootstrap, seed=cfg.seed)
        elapsed = time.perf_counter() - started
        (out / f"{name}.json").write_text(
            json.dumps(
                {
                    "precision": name,
                    "file": path.as_posix(),
                    "parity": agreement,
                    "games": games,
                    "regime": regime,
                    "seed": cfg.seed,
                    "seconds": round(elapsed),
                    "elo": result.model_dump(mode="json"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        variants.append(
            {
                "name": name,
                "parity": agreement,
                "elo": round(result.elo),
                "lo": None if result.ci_low is None else round(result.ci_low),
                "hi": None if result.ci_high is None else round(result.ci_high),
                "games": result.games,
            }
        )
        low = "?" if result.ci_low is None else f"{result.ci_low:.0f}"
        high = "?" if result.ci_high is None else f"{result.ci_high:.0f}"
        print(
            f"{name:<6} parity {agreement if agreement is None else f'{agreement:.3f}':>6}  "
            f"elo {result.elo:.0f} ({low}-{high})  {result.games} games  {elapsed:.0f} s"
        )

    web = resolve(args.web)
    web.parent.mkdir(parents=True, exist_ok=True)
    web.write_text(
        json.dumps(
            {
                "meta": {
                    "generated": datetime.now(UTC).isoformat(timespec="seconds"),
                    "games": games,
                    "regime": regime,
                },
                "model": folder.name,
                "variants": variants,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {web}")


if __name__ == "__main__":
    main()
