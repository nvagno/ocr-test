"""Convertit les notebooks en scripts Python autonomes, à importer dans Kaggle (File > Import Notebook).

Usage : python build_kaggle_scripts.py                  # tous les notebooks
        python build_kaggle_scripts.py 01_got_ocr2.ipynb

Chaque script est écrit dans kaggle/<notebook>.py au format « percent » (`# %%`), donc découpé en cellules :
- `ocr_common.py` est intégré dans le script (Kaggle n'importe qu'un seul fichier) ;
- `%pip` / `!commande` deviennent des appels `subprocess` ;
- les données et résultats sont écrits dans /kaggle/working (ou le dossier courant hors Kaggle) ;
- les figures matplotlib sont aussi enregistrées en PNG dans figures/ ;
- la clé Google Cloud Vision est lue dans les secrets Kaggle (GOOGLE_API_KEY) si elle n'est pas dans l'environnement.
"""
import json
import re
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "kaggle"

PRELUDE = '''
import os
import subprocess
import sys


def sh(cmd):
    """Équivalent de `!cmd` dans un notebook."""
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def pip_install(*args):
    """Équivalent de `%pip install ...`."""
    sh([sys.executable, "-m", "pip", "install", *args])


try:
    from IPython.display import display
except ImportError:
    display = print
'''

SECRETS = '''api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    try:  # Kaggle : Add-ons > Secrets > GOOGLE_API_KEY
        from kaggle_secrets import UserSecretsClient

        api_key = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    except Exception as e:
        print(f"Secret Kaggle GOOGLE_API_KEY indisponible ({type(e).__name__}) : identifiants par défaut")'''


def common_module():
    """Source de ocr_common.py adaptée à un script unique exécuté sur Kaggle."""
    src = (HERE / "ocr_common.py").read_text(encoding="utf-8")
    src = re.sub(r'\A""".*?"""\n', "", src, flags=re.S)
    src = src.replace(
        'ROOT = Path(__file__).resolve().parent\n',
        'ROOT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()\n'
        'FIG_DIR = ROOT / "figures"\n',
    )
    # les figures sont aussi enregistrées sur disque (visibles dans l'onglet Output de Kaggle)
    src = src.replace("    plt.show()\n", "    _show(plt)\n")
    src = src.replace("# --------------------------------------------------------------------------- visualisation\n", '''# --------------------------------------------------------------------------- visualisation


def _show(plt):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / f"fig_{len(list(FIG_DIR.glob('fig_*.png'))):03d}.png", dpi=100, bbox_inches="tight")
    plt.show()
    plt.close()
''')
    assert "_show(plt)" in src and "FIG_DIR =" in src, "ocr_common.py a changé : adapter common_module()"
    return src.strip("\n")


def convert_magic(line):
    if line.startswith("%pip install"):
        return f"pip_install({', '.join(repr(a) for a in shlex.split(line)[2:])})"
    if line.startswith("!"):
        return f"sh({shlex.split(line[1:])!r})"
    if line.startswith("%"):
        return "# " + line  # autres magies : sans équivalent dans un script
    return line


def convert_code(src):
    src = "\n".join(convert_magic(l) for l in src.splitlines())
    src = re.sub(r"^import ocr_common as oc\n", "", src, flags=re.M)
    src = re.sub(r"^from IPython\.display import display\n", "", src, flags=re.M)
    src = re.sub(r"\boc\.", "", src)
    if 'api_key = os.environ.get("GOOGLE_API_KEY")' in src:
        src = src.replace('api_key = os.environ.get("GOOGLE_API_KEY")', SECRETS)
    return src.strip("\n")


def markdown(src):
    src = src.replace(" Le notebook se lance depuis le dossier `hf_ocr_notebooks/` (il importe `ocr_common.py`).", "")
    return "# %% [markdown]\n" + "\n".join(("# " + l).rstrip() for l in src.strip("\n").splitlines())


def build(nb_path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = [("".join(c["source"]), c["cell_type"]) for c in nb["cells"]]
    out = [f'"""{nb_path.name} converti pour Kaggle (généré par build_kaggle_scripts.py, ne pas éditer)."""']
    # 1re cellule markdown (présentation), puis installation, puis outils communs, puis le reste
    intro, cells = cells[0], cells[1:]
    out.append(markdown(intro[0]))
    out.append("# %%" + PRELUDE.rstrip("\n"))
    if cells and cells[0][1] == "code" and cells[0][0].lstrip().startswith(("%pip", "!")):
        out.append("# %%\n" + convert_code(cells[0][0]))
        cells = cells[1:]
    out.append("# %% [markdown]\n# ## Outils communs (contenu de `ocr_common.py`)")
    out.append("# %%\n" + common_module())
    for src, kind in cells:
        out.append(markdown(src) if kind == "markdown" else "# %%\n" + convert_code(src))
    target = OUT_DIR / nb_path.with_suffix(".py").name
    OUT_DIR.mkdir(exist_ok=True)
    target.write_text("\n\n\n".join(out) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    paths = [HERE / a for a in sys.argv[1:]] or sorted(HERE.glob("[0-9][0-9]_*.ipynb"))
    for p in paths:
        print("écrit", build(p).relative_to(HERE))
