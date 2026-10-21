"""Outils partagés par les notebooks OCR HuggingFace.

- chargement uniforme des entrées : images (png, jpg, ...) et PDF (pages rastérisées avec pypdfium2)
- jeu de test synthétique avec vérité terrain (images + PDF multi-pages scanné)
- métriques CER / WER
- boucle de benchmark commune : chaque notebook ne fournit qu'une fonction `extract(image) -> str`
"""
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


def list_inputs(dirs=(SYNTH_DIR, CUSTOM_DIR)):
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


# --------------------------------------------------------------------------- métriques


def normalize(text):
    """Supprime la mise en forme (Markdown, HTML, LaTeX léger, DocTags) et normalise les espaces."""
    text = unicodedata.normalize("NFC", text or "")
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
    import torch

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
