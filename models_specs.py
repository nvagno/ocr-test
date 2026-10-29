"""Spécifications des notebooks : un dict par modèle (chargement + fonction extract)."""

SPECS = {}

SPECS["got_ocr2"] = dict(
    file="01_got_ocr2",
    name="GOT-OCR-2.0",
    title="GOT-OCR 2.0 — OCR généraliste (transformers natif)",
    repo="stepfun-ai/GOT-OCR-2.0-hf",
    license="Apache-2.0",
    size="≈ 580 M paramètres",
    hardware="CPU possible (lent), GPU recommandé",
    output="texte brut (ou Markdown/LaTeX avec `format=True`)",
    intro="""
GOT-OCR 2.0 (StepFun) est un modèle OCR de bout en bout : un encodeur visuel et un petit décodeur de langage
qui lisent directement la page. Il est intégré nativement dans `transformers` (pas de `trust_remote_code`).
""",
    pip="transformers accelerate pypdfium2 pillow pandas rapidfuzz datasets matplotlib",
    load="""
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO = "stepfun-ai/GOT-OCR-2.0-hf"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
processor = AutoProcessor.from_pretrained(REPO)
model = AutoModelForImageTextToText.from_pretrained(REPO, dtype=DTYPE, device_map=DEVICE).eval()
""",
    extract_md="""
`format=True` produit du Markdown/LaTeX (utile pour tableaux et formules) ; `False` donne du texte brut.
Pour les pages très denses, `crop_to_patches=True` découpe l'image en tuiles (plus précis, plus lent).
""",
    extract="""
FORMAT = False
CROP_TO_PATCHES = False


@torch.inference_mode()
def extract(image, max_new_tokens=4096):
    inputs = processor(image, return_tensors="pt", format=FORMAT, crop_to_patches=CROP_TO_PATCHES,
                       max_patches=6).to(DEVICE, DTYPE)
    out = model.generate(**inputs, do_sample=False, tokenizer=processor.tokenizer,
                         stop_strings="<|im_end|>", max_new_tokens=max_new_tokens)
    return processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
""",
)

COMMON_PIP = "pypdfium2 pillow pandas rapidfuzz datasets matplotlib"

PADDLE_SETUP = """
import sys

import numpy as np

try:
    import paddle
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"paddlepaddle absent : aucune wheel pour Python {sys.version_info.major}.{sys.version_info.minor}. "
        "Relancez ce notebook dans un environnement Python 3.8 – 3.12."
    ) from e

DEVICE = "gpu" if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0 else "cpu"
"""

SPECS["paddleocr"] = dict(
    file="02_paddleocr_ppocrv6",
    name="PP-OCRv6",
    title="PaddleOCR PP-OCRv6 — détection + reconnaissance de lignes (léger, CPU)",
    repo="PaddlePaddle/PaddleOCR",
    link="https://github.com/PaddlePaddle/PaddleOCR",
    license="Apache-2.0",
    size="1,5 M (tiny) / 7,7 M (small) / 34,5 M (medium) paramètres",
    hardware="CPU suffisant ; GPU optionnel. Python 3.8 – 3.12 (pas de wheel Paddle pour 3.13+)",
    output="une ligne de texte par boîte détectée (polygone + score)",
    intro="""
Pipeline OCR classique en deux étapes : détection des lignes puis reconnaissance. PP-OCRv6 lit 50 langues avec un
modèle unique (chinois, anglais, japonais + 46 langues latines). **Le malgache n'est pas listé**, mais il s'écrit en
alphabet latin : ce notebook sert justement à vérifier visuellement ce que donne le modèle sur du malgache.
""",
    pip="""
paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
paddleocr """ + COMMON_PIP,
    setup=PADDLE_SETUP,
    load="""
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    ocr_version="PP-OCRv6",                # ou "PP-OCRv5" pour comparer
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device=DEVICE,
)
""",
    extract_md="""
`predict(image)` renvoie le texte (lignes dans l'ordre de détection) et les polygones `rec_polys`, affichés en rouge
pendant la validation visuelle. L'image est passée en tableau numpy BGR (convention OpenCV utilisée par Paddle).
""",
    extract="""
def predict(image):
    res = ocr.predict(np.array(image)[:, :, ::-1].copy())[0]
    return "\\n".join(res["rec_texts"]), list(res["rec_polys"])
""",
)

SPECS["paddleocr_vl"] = dict(
    file="03_paddleocr_vl",
    name="PaddleOCR-VL-1.6",
    title="PaddleOCR-VL 1.6 — analyse de document par modèle vision-langage (0,9 B)",
    repo="PaddlePaddle/PaddleOCR-VL-1.6",
    license="Apache-2.0",
    size="≈ 0,9 B paramètres (+ modèle de layout)",
    hardware="GPU recommandé ; CPU possible mais lent (plusieurs dizaines de s / page). Python 3.8 – 3.12",
    output="Markdown (texte, tableaux, formules) + blocs avec boîte et type",
    intro="""
PaddleOCR-VL détecte d'abord la mise en page (titres, paragraphes, tableaux…), puis lit chaque bloc avec un petit
modèle vision-langage. Le malgache ne figure pas dans la liste des 109 langues de la version d'origine : la
validation visuelle ci-dessous permet de juger sur vos propres documents.
Démo en ligne : https://huggingface.co/spaces/PaddlePaddle/PaddleOCR-VL-1.6_Online_Demo
""",
    pip="""
paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
"paddleocr[doc-parser]" """ + COMMON_PIP,
    setup=PADDLE_SETUP + """
# Sur GPU NVIDIA, remplacez la 1re ligne d'installation par :
# %pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
""",
    load="""
from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL(pipeline_version="v1.6", device=DEVICE)
""",
    extract_md="""
Le texte renvoyé est le Markdown de la page ; les boîtes sont celles des blocs de mise en page (`parsing_res_list`).
""",
    extract="""
def predict(image):
    res = pipeline.predict(np.array(image)[:, :, ::-1].copy())[0]
    blocks = res.json["res"]["parsing_res_list"]
    return res.markdown["markdown_texts"], [b["block_bbox"] for b in blocks]
""",
)

SPECS["surya"] = dict(
    file="04_surya",
    name="Surya-2",
    title="Surya 2 — OCR multilingue, malgache officiellement supporté",
    repo="datalab-to/surya",
    link="https://github.com/datalab-to/surya",
    license="code Apache-2.0 ; poids OpenRAIL-M modifiée (gratuit < 5 M$ de revenus/financement)",
    size="≈ 0,65 B paramètres",
    hardware="GPU NVIDIA (vLLM via Docker) ou CPU / Apple Silicon (`llama-server` de llama.cpp)",
    output="blocs en HTML avec polygone, boîte et confiance",
    intro="""
Surya est le seul modèle de cette série dont la documentation liste le malgache (`mg`, 95,0 % sur le benchmark
interne de 91 langues). Depuis la version 2, la reconnaissance passe par un serveur d'inférence que Surya lance
lui-même au premier appel : **vLLM** (Docker + NVIDIA Container Toolkit) sur GPU NVIDIA, sinon **llama.cpp** —
installez au préalable le binaire `llama-server` (https://github.com/ggml-org/llama.cpp/releases) et mettez-le dans le `PATH`.

Licence des poids : https://github.com/datalab-to/surya/blob/master/MODEL_LICENSE
""",
    pip="surya-ocr " + COMMON_PIP,
    setup="""
import os
import shutil

# Pour réutiliser un serveur déjà lancé :
# os.environ["SURYA_INFERENCE_BACKEND"] = "vllm"            # ou "llamacpp"
# os.environ["SURYA_INFERENCE_URL"] = "http://localhost:8000/v1"
DEVICE = os.environ.get("SURYA_INFERENCE_BACKEND", "auto")
if DEVICE != "vllm" and not shutil.which("llama-server") and not shutil.which("nvidia-smi"):
    print("⚠ ni GPU NVIDIA ni llama-server trouvés : installez llama.cpp avant de charger le modèle")
""",
    load="""
from surya.inference import SuryaInferenceManager
from surya.recognition import RecognitionPredictor

manager = SuryaInferenceManager()          # lance (ou rejoint) le serveur d'inférence
recognition = RecognitionPredictor(manager)
""",
    extract_md="""
Chaque page renvoie des `blocks` (HTML + polygone). Les blocs sont triés par ordre de lecture, le HTML est converti
en texte brut. Pour un découpage par mise en page, passez `layouts` (voir `surya.layout.LayoutPredictor`).
""",
    extract="""
def _get(obj, key):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


def predict(image):
    page = recognition([image])[0]
    blocks = sorted((b for b in _get(page, "blocks") if not _get(b, "skipped")),
                    key=lambda b: _get(b, "reading_order"))
    text = "\\n".join(oc.html_to_text(_get(b, "html")) for b in blocks)
    return text, [_get(b, "polygon") for b in blocks]
""",
)

SPECS["interfaze"] = dict(
    file="05_interfaze_lite",
    name="Interfaze-1.1-Lite",
    title="Interfaze 1.1 Lite — modèle composite (OCR, détection, traduction)",
    repo="interfaze-ai/interfaze-1-lite",
    license="interfaze-open (licence propre, à lire avant tout usage)",
    size="modèle composite (PP-OCRv5 pour les lignes + dots.mocr pour la lecture)",
    hardware="1 GPU de 80 Go, compute capability ≥ 8.9 (testé sur H100) ; `ffmpeg` installé",
    output="texte + pages → lignes → mots, avec boîtes et confiance",
    intro="""
Interfaze regroupe plusieurs modèles derrière une seule API (`model.ocr(...)`). Score annoncé : 83,8 sur olmOCR-Bench,
rien de publié sur le malgache. **Il ne tourne pas sur un poste ordinaire** : lancez ce notebook sur une machine
avec un GPU de 80 Go (H100, A100 80 Go, location cloud). `trust_remote_code=True` exécute le code du dépôt HF :
relisez-le avant.
""",
    pip="""
!hf download interfaze-ai/interfaze-1-lite requirements.txt --local-dir interfaze_req
%pip install -q -r interfaze_req/requirements.txt
""" + COMMON_PIP,
    setup="""
import tempfile

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    vram = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"GPU : {torch.cuda.get_device_name(0)}, {vram:.0f} Go")
    if vram < 75:
        print("⚠ moins de 80 Go de VRAM : le chargement risque d'échouer")
else:
    print("⚠ pas de GPU : Interfaze ne peut pas tourner ici")
""",
    load="""
from transformers import AutoModel

model = AutoModel.from_pretrained("interfaze-ai/interfaze-1-lite", trust_remote_code=True)
""",
    extract_md="""
`model.ocr` prend un chemin de fichier : chaque page est écrite dans un PNG temporaire, donc les coordonnées des
lignes sont en pixels de la page.
""",
    extract="""
TMP = Path(tempfile.mkdtemp())


def predict(image):
    path = TMP / "page.png"
    image.save(path)
    doc = model.ocr(str(path))
    boxes = [[*line["bounds"]["top_left"], *line["bounds"]["bottom_right"]]
             for section in doc["sections"] for line in section["lines"]]
    return doc["text"], boxes
""",
)

SPECS["google_vision"] = dict(
    file="06_google_cloud_vision",
    name="Google-Cloud-Vision",
    title="Google Cloud Vision — OCR par API (référence commerciale)",
    repo="Cloud Vision API · DOCUMENT_TEXT_DETECTION",
    link="https://cloud.google.com/vision",
    license="service payant (1 000 unités gratuites / mois, puis facturé à l'image)",
    size="modèle Google hébergé, rien à télécharger",
    hardware="aucun : appel réseau",
    output="texte complet + pages → blocs → paragraphes → mots avec boîtes",
    intro="""
Sert de référence pour comparer les modèles locaux. Test rapide sans code : déposez une image dans la démo
« Try the API » de https://cloud.google.com/vision#how-it-works.
**Chaque image est envoyée à Google** : n'utilisez pas de documents confidentiels sans accord.
""",
    access="""
Il faut un projet Google Cloud avec l'API Vision activée, puis **une** des deux authentifications :
`gcloud auth application-default login` (ou `GOOGLE_APPLICATION_CREDENTIALS=/chemin/cle.json`), ou une clé d'API
dans la variable `GOOGLE_API_KEY`.""",
    pip="google-cloud-vision " + COMMON_PIP,
    setup="""
import io
import os

DEVICE = "api"
# Indices de langue : [] = détection automatique. Essayez ["mg"] ou ["fr", "mg"] pour le malgache ;
# l'API renvoie une erreur si une langue n'est pas prise en charge.
LANGUAGE_HINTS = []
""",
    load="""
from google.cloud import vision

api_key = os.environ.get("GOOGLE_API_KEY")
client = vision.ImageAnnotatorClient(client_options={"api_key": api_key} if api_key else None)
""",
    extract_md="""
`document_text_detection` est le mode adapté aux pages denses. Les boîtes affichées sont celles des paragraphes.
""",
    extract="""
def predict(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    resp = client.document_text_detection(image=vision.Image(content=buf.getvalue()),
                                          image_context=vision.ImageContext(language_hints=LANGUAGE_HINTS))
    if resp.error.message:
        raise RuntimeError(resp.error.message)
    ann = resp.full_text_annotation
    boxes = [[(v.x, v.y) for v in para.bounding_box.vertices]
             for page in ann.pages for block in page.blocks for para in block.paragraphs]
    return ann.text, boxes
""",
)
