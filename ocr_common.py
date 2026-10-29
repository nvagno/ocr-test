"""Outils partagés par les notebooks OCR HuggingFace.

- chargement uniforme des entrées : images (png, jpg, ...) et PDF (pages rastérisées avec pypdfium2)
- jeu de test synthétique avec vérité terrain (images + PDF multi-pages scanné)
- métriques CER / WER
- jeux de données Hugging Face : textes malgaches rendus en images, ou vraies images annotées
- validation visuelle : image + boîtes détectées + texte lu, à côté de la vérité terrain
- boucle de benchmark commune : chaque notebook ne fournit qu'une fonction `extract(image) -> str`
"""
import html
import json
import platform
import random
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SYNTH_DIR = DATA_DIR / "synthetic"
CUSTOM_DIR = DATA_DIR / "custom"      # déposez ici vos images / PDF (+ nom.txt = vérité terrain)
RESULTS_DIR = ROOT / "results"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PDF_EXTS = {".pdf"}


# --------------------------------------------------------------------------- entrées


def load_document(path, dpi=200, max_pages=None):
    """Renvoie la liste des pages (PIL.Image RGB) d'une image ou d'un PDF."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in PDF_EXTS:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        n = len(pdf) if max_pages is None else min(len(pdf), max_pages)
        pages = [pdf[i].render(scale=dpi / 72).to_pil().convert("RGB") for i in range(n)]
        pdf.close()
        return pages
    if suffix in IMAGE_EXTS:
        img = Image.open(path)
        # TIFF multi-pages
        frames = []
        for i in range(getattr(img, "n_frames", 1)):
            img.seek(i)
            frames.append(img.convert("RGB"))
        return frames[:max_pages] if max_pages else frames
    raise ValueError(f"Format non supporté : {path.suffix}")


def list_inputs(dirs=None):
    """Fichiers de test ; par défaut tous les sous-dossiers de data/ (synthetic, malagasy, hf_*, custom)."""
    if dirs is None:
        dirs = sorted(d for d in DATA_DIR.iterdir() if d.is_dir()) if DATA_DIR.exists() else []
    exts = IMAGE_EXTS | PDF_EXTS
    return sorted(p for d in dirs if Path(d).exists() for p in Path(d).iterdir() if p.suffix.lower() in exts)


def ground_truth(path):
    gt = Path(path).with_suffix(".txt")
    return gt.read_text(encoding="utf-8") if gt.exists() else None


# --------------------------------------------------------------------------- jeu de test


def _font(size):
    for c in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/Library/Fonts/Arial.ttf",
              "C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _render(lines, font_size=32, width=1400, margin=60, line_gap=1.6, height=None):
    font = _font(font_size)
    height = height or int(2 * margin + len(lines) * font_size * line_gap)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    y = margin
    for line in lines:
        draw.text((margin, y), line, fill="black", font=font)
        y += int(font_size * line_gap)
    return img


def _noise(img, amount=0.04, seed=0):
    rnd = random.Random(seed)
    px = img.load()
    w, h = img.size
    for _ in range(int(w * h * amount)):
        v = rnd.randint(0, 255)
        px[rnd.randrange(w), rnd.randrange(h)] = (v, v, v)
    return img.filter(ImageFilter.GaussianBlur(0.8))


IMAGE_SAMPLES = {
    "simple_en": (["The quick brown fox jumps over the lazy dog.",
                   "Optical character recognition benchmark, sample 1."], {}),
    "accents_fr": (["Élève, cœur, garçon, où êtes-vous ? À bientôt !",
                    "Ça coûte 12,50 € TTC, déjà payé le 3 mars."], {}),
    "chiffres_tableau": (["Réf.      Qté    Prix unit.   Total",
                          "A-1043     12      4,75        57,00",
                          "B-2208      3     19,90        59,70",
                          "C-0051    150      0,12        18,00",
                          "Total HT                     134,70"], {}),
    "petite_police": (["Ce texte utilise une police de petite taille pour tester",
                       "la robustesse de la détection sur des caractères fins."], {"font_size": 14, "width": 700}),
    "rotation": (["Document légèrement incliné lors de la numérisation.",
                  "La ligne de base n'est plus horizontale."], {"rotate": 4}),
    "bruit_flou": (["Numérisation de mauvaise qualité avec du bruit.",
                    "Les contours des lettres sont flous."], {"noise": True}),
}

PDF_PAGES = [
    ["Rapport trimestriel - Synthèse",
     "Le chiffre d'affaires progresse de 8,4 % sur le trimestre,",
     "porté par les ventes en ligne et l'ouverture de deux agences.",
     "Les charges d'exploitation restent stables à 1,2 million d'euros."],
    ["Perspectives",
     "Poursuite des investissements dans l'automatisation",
     "et le traitement intelligent des documents (OCR, extraction).",
     "Contact : service.finance@example.com - +33 1 23 45 67 89"],
]


def make_test_set(out_dir=SYNTH_DIR, overwrite=False):
    """Crée 6 images + 1 PDF scanné de 2 pages (format A4 à 150 dpi), chacun avec sa vérité terrain .txt."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    for name, (lines, opts) in IMAGE_SAMPLES.items():
        target = out_dir / f"{name}.png"
        if target.exists() and not overwrite:
            continue
        img = _render(lines, font_size=opts.get("font_size", 32), width=opts.get("width", 1400))
        if opts.get("rotate"):
            img = img.rotate(opts["rotate"], expand=True, fillcolor="white")
        if opts.get("noise"):
            img = _noise(img)
        img.save(target)
        (out_dir / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")

    pdf_path = out_dir / "rapport_2_pages.pdf"
    if overwrite or not pdf_path.exists():
        pages = [_render(lines, font_size=28, width=1240, margin=120, height=1754) for lines in PDF_PAGES]
        pages[0].save(pdf_path, save_all=True, append_images=pages[1:], resolution=150)
        pdf_path.with_suffix(".txt").write_text("\n".join("\n".join(p) for p in PDF_PAGES), encoding="utf-8")
    return list_inputs([out_dir])


MALAGASY_DIR = DATA_DIR / "malagasy"

MALAGASY_SAMPLES = {
    "mg_ohabolana": (["Ny hevitra tsy iray tsy mahatanty hazo lena.",
                      "Aza manao herim-boantay, fa ny hery tsy maharitra."], {}),
    "mg_vaovao": (["Nivory teto Antananarivo omaly ireo solontenan'ny faritra enina amby roapolo.",
                   "Nifanaraka izy ireo fa hotohizana ny fanamboarana ny lalana mankany Toamasina."], {"font_size": 28}),
    "mg_faktiora": (["Faktiora laharana 2026-0458",
                     "Vary 25 kg ............ 87 500 Ar",
                     "Menaka 5 litatra ...... 42 000 Ar",
                     "Totaliny ............. 129 500 Ar"], {}),
    "mg_marika": (["Tsy tokony hatao ambanin-javatra ny fahasalamana.",
                   "Toerana : Fianarantsoa, Mahajanga, Antsiranana, Toliara."], {"noise": True}),
    "mg_mitongilana": (["Fanambarana ofisialy navoakan'ny minisitera.",
                        "Ho an'ny mpianatra rehetra ao amin'ny oniversite."], {"rotate": 3}),
}


def make_malagasy_set(out_dir=MALAGASY_DIR, overwrite=False):
    """5 images de phrases malgaches (proverbes, actualité, facture, bruit, rotation), sans réseau."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (lines, opts) in MALAGASY_SAMPLES.items():
        target = out_dir / f"{name}.png"
        if target.exists() and not overwrite:
            continue
        img = _render(lines, font_size=opts.get("font_size", 32), width=opts.get("width", 1400))
        if opts.get("rotate"):
            img = img.rotate(opts["rotate"], expand=True, fillcolor="white")
        if opts.get("noise"):
            img = _noise(img)
        img.save(target)
        (out_dir / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
    return list_inputs([out_dir])


def _slug(name):
    return re.sub(r"[^\w.-]+", "_", name)


def load_hf_text_dataset(repo="Lo-Renz-O/malagasy-sentence", column="text", split="train", n=8,
                         min_chars=40, max_chars=280, malagasy_only=True, seed=0, overwrite=False):
    """Prend n textes d'un dataset HF (texte seul) et les rend en images avec vérité terrain.

    Par défaut : phrases malgaches réelles (aucun dataset OCR malgache image + texte n'existe sur le Hub).
    Les images sont écrites dans data/hf_<repo>/ ; une variante sur trois est bruitée, une sur trois inclinée.
    `malagasy_only` écarte les textes contenant c, q, u, w ou x (absents de l'alphabet malgache) : le corpus
    contient aussi des phrases en français.
    """
    import textwrap

    from datasets import load_dataset

    out_dir = DATA_DIR / f"hf_{_slug(repo)}"
    if out_dir.exists() and any(out_dir.glob("*.png")) and not overwrite:
        return list_inputs([out_dir])
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(repo, split=split, streaming=True).shuffle(seed=seed, buffer_size=2000)
    i = 0
    for row in ds:
        text = re.sub(r"\s+", " ", str(row[column])).strip()
        if not (min_chars <= len(text) <= max_chars):
            continue
        if malagasy_only and re.search(r"[cquwx]", text, re.I):
            continue
        lines = textwrap.wrap(text, 60)
        img = _render(lines, font_size=30, width=1300)
        if i % 3 == 1:
            img = _noise(img, seed=i)
        elif i % 3 == 2:
            img = img.rotate(2, expand=True, fillcolor="white")
        img.save(out_dir / f"{i:03d}.png")
        (out_dir / f"{i:03d}.txt").write_text("\n".join(lines), encoding="utf-8")
        i += 1
        if i >= n:
            break
    return list_inputs([out_dir])


def load_hf_image_dataset(repo="Teklia/IAM-line", split="test", image_column="image", text_column="text", n=5,
                          overwrite=False):
    """Télécharge n vraies images annotées d'un dataset HF (image + texte) dans data/hf_<repo>/.

    Par défaut : lignes manuscrites anglaises IAM (test difficile pour des modèles entraînés sur l'imprimé).
    Mettre text_column=None si le dataset n'a pas de vérité terrain.
    """
    from datasets import load_dataset

    out_dir = DATA_DIR / f"hf_{_slug(repo)}"
    if out_dir.exists() and any(out_dir.glob("*.png")) and not overwrite:
        return list_inputs([out_dir])
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(load_dataset(repo, split=split, streaming=True)):
        if i >= n:
            break
        row[image_column].convert("RGB").save(out_dir / f"{i:03d}.png")
        if text_column:
            (out_dir / f"{i:03d}.txt").write_text(str(row[text_column]), encoding="utf-8")
    return list_inputs([out_dir])


# --------------------------------------------------------------------------- visualisation


def html_to_text(text):
    """Texte lisible à partir d'une sortie HTML (Surya) : retours à la ligne conservés, balises supprimées."""
    text = re.sub(r"<br\s*/?>|</(p|div|li|tr|h\d)>", "\n", text or "", flags=re.I)
    return html.unescape(re.sub(r"<[^>]+>", "", text)).strip()


def _as_polygon(box):
    """Accepte [x0, y0, x1, y1] ou [[x, y], ...] (liste, tuple ou numpy) et renvoie [[x, y], ...]."""
    pts = [list(map(float, p)) if hasattr(p, "__len__") else float(p) for p in box]
    if pts and not isinstance(pts[0], list):
        x0, y0, x1, y1 = pts[:4]
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return pts


def show_dataset(files, cols=4, width=4):
    """Grille de vignettes (1re page de chaque fichier) pour vérifier ce qui sera testé."""
    import matplotlib.pyplot as plt

    files = list(files)
    rows = max(1, -(-len(files) // cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * width, rows * width * 0.45), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, path in zip(axes.flat, files):
        ax.imshow(load_document(path, dpi=72, max_pages=1)[0])
        ax.set_title(f"{path.parent.name}/{path.name}", fontsize=8)
    plt.tight_layout()
    plt.show()


def show_result(image, text, boxes=None, ref=None, title=""):
    """Image avec les boîtes détectées (gauche) et texte lu vs vérité terrain (droite)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, (ax_img, ax_txt) = plt.subplots(1, 2, figsize=(16, max(3, 7 * image.height / image.width)),
                                         gridspec_kw={"width_ratios": [1.2, 1]})
    ax_img.imshow(image)
    for box in boxes or []:
        ax_img.add_patch(Polygon(_as_polygon(box), closed=True, fill=False, edgecolor="red", linewidth=1.2))
    ax_img.axis("off")
    head = f"{title} — {len(boxes or [])} zones"
    if ref is not None:
        head += f" — CER {cer(ref, text):.1%}"
    ax_img.set_title(head, fontsize=10)
    ax_txt.axis("off")
    body = "SORTIE\n" + (text or "(vide)")[:1500]
    if ref is not None:
        body = "RÉFÉRENCE\n" + ref[:700] + "\n\n" + body
    ax_txt.text(0, 1, body, va="top", ha="left", fontsize=9, family="DejaVu Sans", wrap=True,
                transform=ax_txt.transAxes)
    plt.tight_layout()
    plt.show()


def visual_check(predict, files, max_files=8):
    """Applique `predict(image) -> (texte, boîtes)` à la 1re page de chaque fichier et affiche le résultat."""
    for path in list(files)[:max_files]:
        page = load_document(path, max_pages=1)[0]
        try:
            t = time.perf_counter()
            text, boxes = predict(page)
            dt = time.perf_counter() - t
        except Exception as e:
            print(f"{path.name} : ÉCHEC {type(e).__name__}: {e}")
            continue
        show_result(page, text, boxes, ref=ground_truth(path), title=f"{path.parent.name}/{path.name} ({dt:.1f}s)")


# --------------------------------------------------------------------------- métriques


def normalize(text):
    """Supprime la mise en forme (Markdown, HTML, LaTeX léger, DocTags) et normalise les espaces."""
    text = unicodedata.normalize("NFC", text or "").replace("’", "'").replace("‘", "'")
    text = re.sub(r"<loc_\d+>", " ", text)                # coordonnées DocTags
    text = re.sub(r"<[^>]+>", " ", text)                  # balises HTML / DocTags
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)      # images markdown
    text = re.sub(r"\\(text|mathrm|textbf)\{([^}]*)\}", r"\2", text)
    text = re.sub(r"[#*_`|>]+|\\[()\[\]]", " ", text)      # syntaxe markdown / délimiteurs LaTeX
    text = re.sub(r"^\s*[-:]{3,}\s*$", " ", text, flags=re.M)
    return re.sub(r"\s+", " ", text).strip()


def cer(ref, hyp):
    ref, hyp = normalize(ref), normalize(hyp)
    return Levenshtein.distance(ref, hyp) / max(len(ref), 1)


def wer(ref, hyp):
    ref, hyp = normalize(ref).split(), normalize(hyp).split()
    return Levenshtein.distance(ref, hyp) / max(len(ref), 1)


# --------------------------------------------------------------------------- benchmark


def hardware():
    try:
        import torch
    except ImportError:  # notebooks Paddle / API : pas de torch
        return f"{platform.processor() or platform.machine()} (torch absent)"
    if torch.cuda.is_available():
        return f"CUDA {torch.cuda.get_device_name(0)}"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "Apple MPS"
    return f"CPU {platform.processor() or platform.machine()} ({torch.get_num_threads()} threads)"


def run_benchmark(extract, model_name, files=None, dpi=200, max_pages=None, save_texts=True):
    """Applique `extract(image) -> str` à chaque page de chaque fichier.

    Renvoie un DataFrame (une ligne par fichier) et l'enregistre dans results/<model_name>.csv.
    """
    files = list_inputs() if files is None else [Path(f) for f in files]
    slug = re.sub(r"[^\w.-]+", "_", model_name)
    text_dir = RESULTS_DIR / slug
    text_dir.mkdir(parents=True, exist_ok=True)
    hw = hardware()
    rows = []
    for path in files:
        row = {"model": model_name, "file": path.name, "type": "pdf" if path.suffix.lower() in PDF_EXTS else "image",
               "hardware": hw, "ok": False, "error": None}
        try:
            t0 = time.perf_counter()
            pages = load_document(path, dpi=dpi, max_pages=max_pages)
            t_load = time.perf_counter() - t0
            texts, page_times = [], []
            for page in pages:
                t = time.perf_counter()
                texts.append(extract(page))
                page_times.append(time.perf_counter() - t)
            text = "\n\n".join(texts)
            row.update(ok=True, pages=len(pages), load_s=t_load, total_s=sum(page_times),
                       s_per_page=sum(page_times) / len(pages), chars=len(text), text=text)
            gt = ground_truth(path)
            if gt is not None:
                row.update(ref=gt, cer=cer(gt, text), wer=wer(gt, text))
            if save_texts:
                (text_dir / f"{path.stem}.txt").write_text(text, encoding="utf-8")
        except Exception as e:  # on continue sur les autres fichiers
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        status = (f"{row['pages']} p. en {row['total_s']:6.1f}s" + (f"  CER {row['cer']:.1%}" if "cer" in row else "")
                  if row["ok"] else f"ÉCHEC {row['error'][:80]}")
        print(f"{path.name:28s} {status}")
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["text", "ref"], errors="ignore").to_csv(RESULTS_DIR / f"{slug}.csv", index=False)
    return df


def summarize(df):
    ok = df[df.ok]
    return pd.Series({
        "fichiers OK": f"{len(ok)}/{len(df)}",
        "pages": int(ok.pages.sum()),
        "s / page (moy.)": round(ok.total_s.sum() / max(ok.pages.sum(), 1), 2),
        "CER moyen": f"{ok.cer.mean():.1%}" if "cer" in ok else "n/a",
        "WER moyen": f"{ok.wer.mean():.1%}" if "wer" in ok else "n/a",
        "matériel": df.hardware.iloc[0] if len(df) else "",
    })


def show_comparison(df, n=3):
    """Affiche référence vs sortie pour les n fichiers au CER le plus élevé."""
    if "cer" not in df:
        return
    for _, r in df[df.ok].dropna(subset=["cer"]).sort_values("cer", ascending=False).head(n).iterrows():
        print(f"=== {r.file} — CER {r.cer:.1%}")
        print("Référence :", normalize(r.ref))
        print("Sortie    :", normalize(r.text)[:600], "\n")


def save_meta(model_name, **meta):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^\w.-]+", "_", model_name)
    (RESULTS_DIR / f"{slug}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
