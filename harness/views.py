"""Class example grids rendered on the host for the label-free ("view") condition."""
from pathlib import Path

import numpy as np
from PIL import Image

from wsp import data

GRID_COLS = 8
GRIDS_PER_CLASS = 3
IMAGES_PER_GRID = 64
SCALE = 3
VIEW_SEED = 7


def render_grid(images: np.ndarray, out: Path, scale: int = SCALE, cols: int = GRID_COLS) -> None:
    s = 32 * scale
    rows = -(-len(images) // cols)
    canvas = Image.new("RGB", (cols * (s + 2), rows * (s + 2)), "white")
    for i, im in enumerate(images):
        tile = Image.fromarray(im.transpose(1, 2, 0)).resize((s, s), Image.NEAREST)
        canvas.paste(tile, ((i % cols) * (s + 2), (i // cols) * (s + 2)))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def class_view_parts(y: np.ndarray, classes: list[str]):
    """Yields (class name, file number, training indices): the fixed labelled examples of the view condition."""
    rng = np.random.default_rng(VIEW_SEED)
    for c, name in enumerate(classes):
        idx = rng.choice(np.flatnonzero(y == c), GRIDS_PER_CLASS * IMAGES_PER_GRID, replace=False)
        for g in range(GRIDS_PER_CLASS):
            yield name, g, idx[g * IMAGES_PER_GRID:(g + 1) * IMAGES_PER_GRID]


def write_class_grids(out_dir: Path) -> None:
    d = np.load(data.CIFAR_DIR / "train.npz")
    for name, g, part in class_view_parts(d["y"], data.CLASSES):
        render_grid(d["x"][part], out_dir / f"{name}_{g}.png")


WORD_VIEW_PATCHES = (1, 3, 5, 8, 10, 12)  # patch rows and columns listed per image: 3 x 3 in each quadrant
QUADRANT_NAMES = ("TL", "TR", "BL", "BR")


def word_line(words: np.ndarray) -> str:
    """One image's (14, 14) word map as 'TL a b .. | TR .. | BL .. | BR ..', 9 words per quadrant, row by row."""
    sub = words[np.ix_(WORD_VIEW_PATCHES, WORD_VIEW_PATCHES)]
    quads = (sub[:3, :3], sub[:3, 3:], sub[3:, :3], sub[3:, 3:])
    return " | ".join(f"{n} " + " ".join(str(int(w)) for w in q.ravel()) for n, q in zip(QUADRANT_NAMES, quads))


def write_class_grids_and_words(out_dir: Path) -> None:
    """The class grids of write_class_grids plus, for the same images in the same order, their patch words."""
    write_class_grids(out_dir)
    y = np.load(data.CIFAR_DIR / "train.npz")["y"]
    words = np.load(data.CIFAR_DIR / "train_words.npz")["words"]
    for name, g, part in class_view_parts(y, data.CLASSES):
        lines = [f"{i:02d} {word_line(words[j])}" for i, j in enumerate(part)]
        (out_dir / f"{name}_{g}.txt").write_text("\n".join(lines) + "\n")


SHEET_WORDS = 64
SHEET_COLS = 4
SHEET_EXAMPLES = 6
SHEET_TILE = 42
SHEET_SEED = 3


def write_word_sheets(out_dir: Path = data.CIFAR_DIR) -> None:
    """patch_words_<k>.png: for every patch word its number, the mean of its patches and SHEET_EXAMPLES patches.

    Patches come from 2,000 seeded unlabelled training images and are shown as raw pixels.
    """
    import torch
    import torch.nn.functional as F
    from PIL import ImageDraw
    from wsp import vision_words as vw
    rng = np.random.default_rng(SHEET_SEED)
    x = np.load(data.CIFAR_DIR / "train.npz")["x"]
    x = torch.from_numpy(x[rng.choice(len(x), 2000, replace=False)]).float() / 255
    with torch.no_grad():
        words = vw.word_map(x).reshape(-1).numpy()
    raw = F.unfold(x, vw.VW_PATCH, stride=vw.VW_STRIDE).transpose(1, 2).reshape(-1, 3, vw.VW_PATCH, vw.VW_PATCH)
    cell_w, cell_h = 46 + (SHEET_EXAMPLES + 1) * (SHEET_TILE + 2) + 10, SHEET_TILE + 8
    for sheet in range(-(-vw.VW_WORDS // SHEET_WORDS)):
        ids = range(sheet * SHEET_WORDS, min((sheet + 1) * SHEET_WORDS, vw.VW_WORDS))
        canvas = Image.new("RGB", (SHEET_COLS * cell_w, -(-len(ids) // SHEET_COLS) * cell_h), "white")
        draw = ImageDraw.Draw(canvas)
        for n, w in enumerate(ids):
            x0, y0 = (n % SHEET_COLS) * cell_w, (n // SHEET_COLS) * cell_h
            draw.text((x0 + 4, y0 + SHEET_TILE // 2 - 4), f"{w:3d}", fill="black")
            members = np.flatnonzero(words == w)
            tiles = [raw[members].mean(0)] if len(members) else []
            tiles += [raw[j] for j in rng.choice(members, min(SHEET_EXAMPLES, len(members)), replace=False)]
            for t, tile in enumerate(tiles):
                im = Image.fromarray((tile.numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8))
                canvas.paste(im.resize((SHEET_TILE, SHEET_TILE), Image.NEAREST),
                             (x0 + 46 + t * (SHEET_TILE + 2) + (6 if t else 0), y0 + 4))
        canvas.save(out_dir / f"patch_words_{sheet}.png")


TOKEN_SHEET_ROWS = 16
TOKEN_SHEET_EXAMPLES = 5
TOKEN_TILE = 96
TOKEN_DIM = 0.3
TOKEN_VIEW_TOP = 10


def write_token_sheets() -> None:
    """token_sheet_<k>.png: per token of the vtok tokenizer, TOKEN_SHEET_EXAMPLES patches drawn from 2,000 seeded
    unlabelled training images, each as the image it lies in and that image with the other tokens' patches dimmed.
    """
    from PIL import ImageDraw
    from wsp import vision_tokens as vt
    rng = np.random.default_rng(SHEET_SEED)
    out_dir = vt.token_dir() / vt.TOKENIZER
    pick = rng.choice(50000, 2000, replace=False)
    images = np.load(data.CIFAR_DIR / "train.npz")["x"][pick].transpose(0, 2, 3, 1)
    tokens = np.load(out_dir / "train.npz")["x"][pick]
    side = int(vt.VT_TOKENS ** 0.5)
    cell_w, cell_h = 46 + TOKEN_SHEET_EXAMPLES * (2 * TOKEN_TILE + 10), TOKEN_TILE + 4

    def tile(im: np.ndarray) -> Image.Image:
        return Image.fromarray(im).resize((TOKEN_TILE, TOKEN_TILE), Image.NEAREST)

    for sheet in range(vt.VT_VOCAB // TOKEN_SHEET_ROWS):
        canvas = Image.new("RGB", (cell_w, TOKEN_SHEET_ROWS * cell_h), "white")
        draw = ImageDraw.Draw(canvas)
        for row in range(TOKEN_SHEET_ROWS):
            t = sheet * TOKEN_SHEET_ROWS + row
            draw.text((4, row * cell_h + TOKEN_TILE // 2 - 4), f"{t:3d}", fill="black")
            drawn = rng.permutation(np.argwhere(tokens == t)[:, 0])
            owners = drawn[np.sort(np.unique(drawn, return_index=True)[1])]
            for n, i in enumerate(owners[:TOKEN_SHEET_EXAMPLES]):
                keep = (tokens[i] == t).reshape(side, side)
                mask = np.kron(keep, np.ones((32 // side, 32 // side), bool))[:, :, None]
                dimmed = np.where(mask, images[i], (images[i] * TOKEN_DIM).astype(np.uint8))
                x0 = 46 + n * (2 * TOKEN_TILE + 10)
                canvas.paste(tile(images[i]), (x0, row * cell_h + 2))
                canvas.paste(tile(dimmed), (x0 + TOKEN_TILE + 2, row * cell_h + 2))
        canvas.save(out_dir / f"token_sheet_{sheet:02d}.png")


DIM_SHEET_ROWS = 16
DIM_SHEET_EXAMPLES = 5


def write_dimension_sheets() -> None:
    """dim_sheet_<k>.png: per dimension of the vemb embedding, the DIM_SHEET_EXAMPLES images with the highest value
    (left, "+") and the lowest value (right, "-") among the same 2,000 seeded unlabelled training images as the token
    sheets."""
    from PIL import ImageDraw
    from wsp import vision_tokens as vt
    rng = np.random.default_rng(SHEET_SEED)
    pick = rng.choice(50000, 2000, replace=False)
    images = np.load(data.CIFAR_DIR / "train.npz")["x"][pick].transpose(0, 2, 3, 1)
    emb = np.load(vt.embed_dir() / "train.npz")["x"][pick].astype(np.float32)
    gap = 30
    cell_w, cell_h = 60 + 2 * DIM_SHEET_EXAMPLES * (TOKEN_TILE + 4) + gap, TOKEN_TILE + 4

    def tile(im: np.ndarray) -> Image.Image:
        return Image.fromarray(im).resize((TOKEN_TILE, TOKEN_TILE), Image.NEAREST)

    for sheet in range(vt.VE_DIM // DIM_SHEET_ROWS):
        canvas = Image.new("RGB", (cell_w, DIM_SHEET_ROWS * cell_h), "white")
        draw = ImageDraw.Draw(canvas)
        for row in range(DIM_SHEET_ROWS):
            d = sheet * DIM_SHEET_ROWS + row
            order = np.argsort(emb[:, d], kind="stable")
            y0 = row * cell_h + 2
            draw.text((4, y0 + TOKEN_TILE // 2 - 4), f"{d:3d}", fill="black")
            for n, i in enumerate(order[::-1][:DIM_SHEET_EXAMPLES]):
                canvas.paste(tile(images[i]), (60 + n * (TOKEN_TILE + 4), y0))
            x_low = 60 + DIM_SHEET_EXAMPLES * (TOKEN_TILE + 4) + gap
            draw.text((x_low - gap + 8, y0 + TOKEN_TILE // 2 - 4), "|", fill="black")
            for n, i in enumerate(order[:DIM_SHEET_EXAMPLES]):
                canvas.paste(tile(images[i]), (x_low + n * (TOKEN_TILE + 4), y0))
        canvas.save(vt.embed_dir() / f"dim_sheet_{sheet:02d}.png")


def write_class_grids_and_tokens(out_dir: Path) -> None:
    """The class grids plus, for the same images in the same order, each image's most frequent tokens."""
    from wsp import vision_tokens as vt
    write_class_grids(out_dir)
    d = np.load(vt.token_dir() / vt.TOKENIZER / "train.npz")
    for name, g, part in class_view_parts(d["y"], data.CLASSES):
        lines = []
        for i, j in enumerate(part):
            n = np.bincount(d["x"][j], minlength=vt.VT_VOCAB)
            top = np.argsort(-n, kind="stable")[:TOKEN_VIEW_TOP]
            lines.append(f"{i:02d} " + " ".join(f"{t}:{n[t]}" for t in top if n[t]))
        (out_dir / f"{name}_{g}.txt").write_text("\n".join(lines) + "\n")


def write_unlabeled_images(out: Path) -> None:
    np.savez(out, x=np.load(data.CIFAR_DIR / "train.npz")["x"])


def write_class_texts(out_dir: Path) -> None:
    """AG News analogue of the class grids: GRIDS_PER_CLASS files of IMAGES_PER_GRID articles per class."""
    import json
    from wsp import text_data
    rows = [json.loads(line) for line in (text_data.AGNEWS_DIR / "train.jsonl").read_text().splitlines()]
    y = np.array([text_data.CLASSES.index(r["label"]) for r in rows])
    rng = np.random.default_rng(VIEW_SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    for c, name in enumerate(text_data.CLASSES):
        idx = rng.choice(np.flatnonzero(y == c), GRIDS_PER_CLASS * IMAGES_PER_GRID, replace=False)
        for g in range(GRIDS_PER_CLASS):
            part = idx[g * IMAGES_PER_GRID:(g + 1) * IMAGES_PER_GRID]
            (out_dir / f"{name}_{g}.txt").write_text("\n\n".join(rows[i]["text"] for i in part) + "\n")


def write_class_ids(out_dir: Path) -> None:
    """Cipher analogue of write_class_texts: the same articles per class file, as space-separated ids."""
    from wsp import text_data
    d = np.load(text_data.CIPHER_DIR / "train.npz")
    x, y = d["x"], d["y"]
    rng = np.random.default_rng(VIEW_SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    for c, name in enumerate(text_data.CLASSES):
        idx = rng.choice(np.flatnonzero(y == c), GRIDS_PER_CLASS * IMAGES_PER_GRID, replace=False)
        for g in range(GRIDS_PER_CLASS):
            part = idx[g * IMAGES_PER_GRID:(g + 1) * IMAGES_PER_GRID]
            (out_dir / f"{name}_{g}.txt").write_text(
                "\n".join(" ".join(str(t) for t in row[row != text_data.PAD]) for row in x[part]) + "\n")


CLASS_VIEWS = {"cifar": write_class_grids, "cifar_words": write_class_grids_and_words,
               "cifar_tokens": write_class_grids_and_tokens, "agnews": write_class_texts,
               "agnews_cipher": write_class_ids}


if __name__ == "__main__":
    import sys
    if "--dimensions" in sys.argv:
        write_dimension_sheets()
    else:
        write_token_sheets() if "--tokens" in sys.argv else write_word_sheets()
