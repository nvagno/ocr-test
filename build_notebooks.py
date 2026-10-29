"""Génère un notebook par modèle à partir de specs, avec une structure commune.

Usage : python build_notebooks.py            # tous les notebooks (ceux déjà exécutés sont conservés)
        python build_notebooks.py got_ocr2   # un seul
        python build_notebooks.py --force    # régénère aussi les notebooks déjà exécutés (perd leurs sorties)

Chaque spec définit soit `extract(image) -> str`, soit `predict(image) -> (texte, boîtes)` ;
la fonction manquante est dérivée de l'autre par le template.
"""
import json
import sys
from pathlib import Path

from models_specs import SPECS

HERE = Path(__file__).resolve().parent

DEFAULT_SETUP = """
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
"""


def _cell(kind, src):
    src = src.strip("\n")
    cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def build(spec, force=False):
    cells = [
        ("markdown", f"""
# {spec['title']}

{spec['intro'].strip()}

| | |
|---|---|
| Modèle | [{spec['repo']}]({spec.get('link', 'https://huggingface.co/' + spec['repo'])}) |
| Licence | {spec['license']} |
| Taille | {spec['size']} |
| Matériel | {spec['hardware']} |
| Sortie | {spec['output']} |
| PDF | pages rastérisées avec `pypdfium2` (fonction commune `load_document`) |

{spec.get('access', """Aucune clé d'API ni token HuggingFace n'est nécessaire : les poids publics sont téléchargés au premier lancement
(dans `~/.cache/huggingface`).""").strip()} Le notebook se lance depuis le dossier `hf_ocr_notebooks/` (il importe `ocr_common.py`).
"""),
        ("code", "\n".join(l if l.startswith(("!", "%")) else f"%pip install -q {l}"
                           for l in spec["pip"].strip().splitlines())),
        ("markdown", """
## 1. Jeux de données

Toutes les sources écrivent dans `data/<source>/` une image par échantillon et, à côté, sa vérité terrain `.txt` :

| Source | Contenu |
|---|---|
| `synthetic/` | 6 images + 1 PDF (français, anglais, tableau, petite police, rotation, bruit) |
| `malagasy/` | 5 images de phrases malgaches intégrées (sans réseau) |
| `hf_Lo-Renz-O_malagasy-sentence/` | phrases malgaches réelles du Hub, rendues en images (bruit / inclinaison) |
| `hf_Teklia_IAM-line/` | vraies images de lignes manuscrites anglaises (IAM) |
| `custom/` | vos documents (image ou PDF, + `nom.txt` optionnel pour mesurer le CER) |

Les datasets du Hub sont téléchargés une seule fois (paquet `datasets`) ; sans réseau, ils sont ignorés.
"""),
        ("code", """
import time
from pathlib import Path

from IPython.display import display

import ocr_common as oc

oc.make_test_set()
oc.make_malagasy_set()
for loader in (oc.load_hf_text_dataset, oc.load_hf_image_dataset):
    try:
        loader()
    except Exception as e:  # pas de réseau / pas de paquet datasets
        print(f"{loader.__name__} ignoré : {type(e).__name__}: {e}")

FILES = oc.list_inputs()
print(f"{len(FILES)} fichiers :", ", ".join(f"{f.parent.name}/{f.name}" for f in FILES))
oc.show_dataset([f for f in FILES if f.suffix != ".pdf"])
"""),
        ("code", spec.get("setup", DEFAULT_SETUP).strip("\n") + '\nprint("Matériel :", oc.hardware(), "| DEVICE =", DEVICE)'),
        ("markdown", "## 2. Chargement du modèle"),
        ("code", "t0 = time.perf_counter()\n" + spec["load"].strip("\n")
         + "\nLOAD_S = time.perf_counter() - t0\nprint(f\"Modèle chargé en {LOAD_S:.1f}s sur {DEVICE}\")"),
        ("markdown", "## 3. Fonction d'extraction\n\n" + spec.get("extract_md", "`extract(image)` prend une page (PIL.Image) et renvoie le texte extrait.")),
        ("code", spec["extract"].strip("\n") + ("""


def extract(image):
    return predict(image)[0]
""" if "def predict" in spec["extract"] else """


def predict(image):
    return extract(image), []  # ce modèle ne renvoie pas de boîtes
""")),
        ("markdown", """
## 4. Validation visuelle

Pour chaque image : à gauche la page avec les zones détectées (en rouge), à droite la vérité terrain et le texte lu,
avec le CER dans le titre. Commencez par une image malgache, puis parcourez tout le jeu de test.
"""),
        ("code", """
img = oc.load_document(oc.MALAGASY_DIR / "mg_vaovao.png")[0]
t = time.perf_counter()
text, boxes = predict(img)
print(f"({time.perf_counter() - t:.1f}s)\\n{text}")
oc.show_result(img, text, boxes, ref=oc.ground_truth(oc.MALAGASY_DIR / "mg_vaovao.png"), title="mg_vaovao.png")
"""),
        ("code", """
# toutes les sources sauf les PDF ; augmentez max_files pour tout voir
oc.visual_check(predict, [f for f in FILES if f.suffix != ".pdf"], max_files=12)
"""),
        ("markdown", "### PDF multi-pages"),
        ("code", """
pdf_path = oc.SYNTH_DIR / "rapport_2_pages.pdf"
for i, page in enumerate(oc.load_document(pdf_path), 1):
    t = time.perf_counter()
    text = extract(page)
    print(f"--- page {i} ({time.perf_counter() - t:.1f}s)\\n{text}\\n")
"""),
        ("markdown", """
## 5. Benchmark sur tout le jeu de test

Temps par page et précision (CER / WER) sur tous les fichiers. Le premier appel inclut un éventuel temps de compilation/« chauffe ».
Les textes extraits sont enregistrés dans `results/<modèle>/` et les mesures dans `results/<modèle>.csv`
(utilisées par le notebook `00_comparaison.ipynb`).
"""),
        ("code", f"""
MODEL_NAME = "{spec['name']}"
df = oc.run_benchmark(extract, MODEL_NAME, FILES)
oc.save_meta(MODEL_NAME, repo="{spec['repo']}", load_s=LOAD_S, device=str(DEVICE))
display(oc.summarize(df).to_frame(MODEL_NAME))
"""),
        ("code", """
display(df[[c for c in ("file", "type", "pages", "s_per_page", "cer", "wer", "error") if c in df]].round(3))
mg = df[df.file.str.startswith("mg_") | df.file.str.fullmatch(r"\\d{3}\\.png")]
if "cer" in mg and len(mg):
    print(f"CER moyen sur les échantillons malgaches et HF : {mg.cer.mean():.1%}")
oc.show_comparison(df, n=3)
"""),
        ("markdown", """
## 6. Utiliser le modèle sur vos propres fichiers

Remplacez le chemin ci-dessous par une image ou un PDF ; le texte est enregistré à côté du fichier (`.ocr.txt`).
"""),
        ("code", """
def extract_file(path, dpi=200, max_pages=None):
    path = Path(path)
    pages = oc.load_document(path, dpi=dpi, max_pages=max_pages)
    text = "\\n\\n".join(extract(p) for p in pages)
    path.with_suffix(".ocr.txt").write_text(text, encoding="utf-8")
    return text


# print(extract_file("data/custom/mon_document.pdf"))
"""),
    ]
    nb = {
        "cells": [_cell(k, s) for k, s in cells],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{i:02d}"
    out = HERE / f"{spec['file']}.ipynb"
    if out.exists() and not force and any(c.get("outputs") for c in json.loads(out.read_text())["cells"]):
        print(f"{out.name} déjà exécuté : conservé (--force pour le régénérer)")
        return None
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    force = "--force" in sys.argv
    wanted = {a for a in sys.argv[1:] if a != "--force"}
    for key, spec in SPECS.items():
        if not wanted or key in wanted:
            out = build(spec, force)
            if out:
                print("écrit", out.name)
