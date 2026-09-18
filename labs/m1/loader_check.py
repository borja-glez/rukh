from rukh.tokenize.loader import PackedDataset, make_loader


def main() -> None:
    ds = PackedDataset("data/tokens/uci/train", block=200)
    x, y = ds[0]
    print(len(ds), x.shape, y.shape, x[:6].tolist(), y[:6].tolist())
    loader = make_loader(ds, batch_size=64, seed=0, workers=4)
    xb, yb = next(iter(loader))
    print(xb.shape, yb.shape, xb.dtype, (yb == 0).float().mean().item())


if __name__ == "__main__":  # obligatorio en Windows: los workers arrancan con spawn
    main()
