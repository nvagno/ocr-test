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
    pip="transformers accelerate pypdfium2 pillow pandas rapidfuzz",
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
