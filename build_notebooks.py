"""Génère un notebook par modèle à partir de specs, avec une structure commune.

Usage : python build_notebooks.py            # tous les notebooks
        python build_notebooks.py got_ocr2   # un seul
"""
import json
import sys
from pathlib import Path

from models_specs import SPECS

HERE = Path(__file__).resolve().parent


def _cell(kind, src):
    src = src.strip("\n")
    cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def build(spec):
    cells = [
        ("markdown", f"""
# {spec['title']}

{spec['intro'].strip()}

| | |
|---|---|
| Modèle | [{spec['repo']}](https://huggingface.co/{spec['repo']}) |
| Licence | {spec['license']} |
| Taille | {spec['size']} |
| Matériel | {spec['hardware']} |
| Sortie | {spec['output']} |
| PDF | pages rastérisées avec `pypdfium2` (fonction commune `load_document`) |

Aucune clé d'API ni token HuggingFace n'est nécessaire : les poids publics sont téléchargés au premier lancement
(dans `~/.cache/huggingface`). Le notebook se lance depuis le dossier `hf_ocr_notebooks/` (il importe `ocr_common.py`).
"""),
        ("code", f"%pip install -q {spec['pip']}"),
        ("markdown", "## 1. Préparation"),
        ("code", """
import time
from pathlib import Path

import torch
from IPython.display import display

import ocr_common as oc

FILES = oc.make_test_set()          # 6 images + 1 PDF de 2 pages, avec vérité terrain
FILES = oc.list_inputs()            # + vos fichiers déposés dans data/custom/
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Matériel :", oc.hardware())
print(f"{len(FILES)} fichiers :", ", ".join(f.name for f in FILES))
"""),
        ("markdown", "## 2. Chargement du modèle"),
        ("code", "t0 = time.perf_counter()\n" + spec["load"].strip("\n")
         + "\nLOAD_S = time.perf_counter() - t0\nprint(f\"Modèle chargé en {LOAD_S:.1f}s sur {DEVICE}\")"),
        ("markdown", "## 3. Fonction d'extraction\n\n" + spec.get("extract_md", "`extract(image)` prend une page (PIL.Image) et renvoie le texte extrait.")),
        ("code", spec["extract"]),
        ("markdown", "## 4. Démonstration sur une image et sur un PDF"),
        ("code", """
img_path = oc.SYNTH_DIR / "accents_fr.png"
img = oc.load_document(img_path)[0]
display(img.resize((img.width // 2, img.height // 2)))
t = time.perf_counter()
print(extract(img))
print(f"\\n({time.perf_counter() - t:.1f}s)")
"""),
        ("code", """
pdf_path = oc.SYNTH_DIR / "rapport_2_pages.pdf"
for i, page in enumerate(oc.load_document(pdf_path), 1):
    t = time.perf_counter()
    text = extract(page)
    print(f"--- page {i} ({time.perf_counter() - t:.1f}s)\\n{text}\\n")
"""),
        ("markdown", """
## 5. Benchmark sur le jeu de test

Temps par page et précision (CER / WER) sur tous les fichiers. Le premier appel inclut un éventuel temps de compilation/« chauffe ».
Les textes extraits sont enregistrés dans `results/<modèle>/` et les mesures dans `results/<modèle>.csv`
(utilisées par le notebook `00_comparaison.ipynb`).
"""),
        ("code", f"""
MODEL_NAME = "{spec['name']}"
df = oc.run_benchmark(extract, MODEL_NAME, FILES)
oc.save_meta(MODEL_NAME, repo="{spec['repo']}", load_s=LOAD_S, device=DEVICE)
display(oc.summarize(df).to_frame(MODEL_NAME))
"""),
        ("code", """
display(df[[c for c in ("file", "type", "pages", "s_per_page", "cer", "wer", "error") if c in df]].round(3))
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
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    wanted = set(sys.argv[1:])
    for key, spec in SPECS.items():
        if not wanted or key in wanted:
            print("écrit", build(spec).name)
