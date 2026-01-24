import base64, io, re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image

def pil_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img = img.convert("RGB")
    img.save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

_CANON = {
    "yes":"Yes","no":"No","unknown":"Unknown",
    "xray":"X-ray","x-ray":"X-ray","ct":"CT","mri":"MRI","ultrasound":"Ultrasound","pet":"PET",
    "lungs":"Lung","lung":"Lung","heart":"Heart","hearts":"Heart","chest":"Chest",
    "right lung":"Right lung","left lung":"Left lung",
    "present":"Present","absent":"Absent","unclear":"Unclear",
    "abnormal":"Abnormal","noabnormal":"NoAbnormal",
}

def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip().strip(".").strip(",").strip()
    if not s:
        return "Unknown"
    key = s.lower()
    if key in _CANON:
        return _CANON[key]
    return s

def _strip_leading_yesno(s: str) -> str:
    return re.sub(r"^(yes|no|unknown)[,:\s]+", "", (s or "").strip(), flags=re.I)

def normalize_modality(s: str) -> str:
    sl = (s or "").lower()
    sl = re.sub(r"[^a-z0-9\s/-]", " ", sl)
    sl = re.sub(r"\s+", " ", sl).strip()
    if not sl:
        return "Unknown"
    has_pet = bool(re.search(r"\bpet\b", sl) or "positron emission" in sl)
    has_ct = bool(
        re.search(r"\bct\b", sl)
        or "ct scan" in sl
        or "computed tomography" in sl
        or "cat scan" in sl
    )
    if has_pet and has_ct:
        return "PET"
    if has_ct:
        return "CT"
    if re.search(r"\bmri\b", sl) or "magnetic resonance" in sl:
        return "MRI"
    if "ultrasound" in sl or "sonogram" in sl or "sonography" in sl:
        return "Ultrasound"
    if (
        re.search(r"\bx[- ]?ray\b", sl)
        or re.search(r"\bcxr\b", sl)
        or "radiograph" in sl
        or "fluoroscopy" in sl
    ):
        return "X-ray"
    return normalize_text(s)

def normalize_yesno(s: str) -> str:
    sl = (s or "").strip().lower()
    sl = re.sub(r"[^a-z\s]", " ", sl)
    sl = re.sub(r"\s+", " ", sl).strip()
    if not sl:
        return "Unknown"
    if sl.startswith("yes"):
        return "Yes"
    if sl.startswith("no"):
        return "No"
    if sl in ("unknown", "unclear", "unsure", "not sure"):
        return "Unknown"
    if sl in ("absent", "noabnormal", "no abnormal", "no abnormality", "no abnormalities", "none"):
        return "No"
    if sl in ("present", "abnormal", "positive"):
        return "Yes"
    if sl in ("negative", "not present"):
        return "No"
    return normalize_text(s)

def normalize_answer(s: str, qtype: str) -> str:
    if qtype == "modality":
        return normalize_modality(s)
    if qtype in ("contain_yesno", "healthy_yesno", "abnormal_yesno"):
        return normalize_yesno(s)
    cleaned = _strip_leading_yesno(s)
    cleaned = normalize_text(cleaned)
    if cleaned.lower() in ("none", "no"):
        return "Unknown"
    return cleaned

def classify_question(q: str) -> str:
    ql = (q or "").lower().strip()
    if "modality" in ql or "what modality" in ql:
        return "modality"
    if ql.startswith("where ") or "where is/are the abnormal" in ql:
        return "location"
    if ql.startswith("does the picture contain") or ql.startswith("does the image contain"):
        return "contain_yesno"
    if ql.startswith("is ") and (" healthy" in ql or "normal" in ql):
        return "healthy_yesno"
    if "which organ" in ql and "abnormal" in ql:
        return "short"
    if (
        (ql.startswith("is there") or ql.startswith("are there"))
        and ("abnormal" in ql or "abnormality" in ql)
    ):
        return "abnormal_yesno"
    if (
        (ql.startswith("does the image show") or ql.startswith("does the picture show"))
        and ("abnormal" in ql or "abnormality" in ql)
    ):
        return "abnormal_yesno"
    if "disease" in ql or "diagnosis" in ql:
        return "disease"
    if "abnormal" in ql or "abnormality" in ql or "where is/are the abnormal" in ql:
        return "location"
    return "short"

def rules_for(qtype: str) -> Tuple[str,int]:
    if qtype == "modality":
        return (
            "Output ONLY the imaging modality (1-3 words). No sentences or explanations.\n"
            "If unsure, output Unknown.",
            8,
        )
    if qtype in ("contain_yesno","healthy_yesno","abnormal_yesno"):
        return (
            "Output ONLY: Yes, No, or Unknown.\n"
            "If image evidence is unclear, output Unknown. No explanations.",
            6,
        )
    if qtype == "disease":
        return (
            "Output ONLY disease name(s), 1-6 words total. No sentences or explanations.\n"
            "You may list multiple items separated by commas. If unsure, output Unknown.",
            12,
        )
    if qtype == "location":
        return (
            "Output ONLY location(s), 1-6 words total. No sentences or explanations.\n"
            "You may list multiple items separated by commas. If unsure, output Unknown.",
            12,
        )
    return (
        "Answer briefly (1-6 words). No sentences or explanations.\n"
        "You may list multiple items separated by commas. If unsure, output Unknown.",
        12,
    )

@dataclass
class GenOut:
    texts: List[str]
    voted: str
    raw: Dict[str, Any]
    qtype: str

class VLLMOpenAIMMPolicyV5:
    def __init__(self, base_url="http://localhost:8000/v1", model="medgemma-27b-it", api_key="dummy", timeout_s=180):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.headers = {"Content-Type":"application/json","Authorization":f"Bearer {api_key}"}
        self.timeout_s = timeout_s

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(f"{self.base_url}/chat/completions", headers=self.headers, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def _ask_once_mm(self, img: Image.Image, text: str, temperature: float, max_tokens: int) -> str:
        img_url = pil_to_data_url(img)
        payload = {
            "model": self.model,
            "n": 1,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [{
                "role": "user",
                "content": [
                    {"type":"text","text": text},
                    {"type":"image_url","image_url":{"url": img_url}},
                ],
            }],
        }
        raw = self._post(payload)
        return raw["choices"][0]["message"]["content"].strip()

    def generate_n_text(self, prompt: str, n: int = 5, temperature: float = 0.7, top_p: float = 0.95, max_tokens: int = 64) -> List[str]:
        payload = {
            "model": self.model,
            "n": int(max(1, n)),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "messages": [{"role": "user", "content": prompt}],
        }
        raw = self._post(payload)
        texts = [c["message"]["content"].strip() for c in raw.get("choices", [])]
        if not texts:
            texts = ["Unknown"] * int(max(1, n))
        return texts

    def generate_n_mm(
        self,
        image: Image.Image,
        question: str,
        n: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: Optional[int] = None,
        use_gate: bool = True,
    ) -> GenOut:
        qtype = classify_question(question)
        rules, default_max = rules_for(qtype)
        max_tokens = int(max_tokens or default_max)

        # -------- contain gate (heart gets stricter wording) --------
        if use_gate and qtype == "contain_yesno":
            m = re.search(r"contain\s+(.+?)\?", (question or "").lower())
            organ = (m.group(1) if m else "the organ").strip()

            # heart bias killer: do NOT assume typical anatomy; only count if clearly visible in THIS slice
            if "heart" in organ or "cardiac" in organ:
                gate_q = (
                    "Answer ONLY one token: Present, Absent, or Unclear.\n"
                    "Important: Do NOT answer based on typical anatomy. Judge ONLY from visible pixels in THIS image slice.\n"
                    "Present = you can clearly identify the heart structure/cardiac silhouette.\n"
                    "Absent = heart is not visible in this image.\n"
                    "Unclear = too ambiguous.\n"
                    "Question: Is the heart visible in this image?"
                )
            else:
                gate_q = (
                    "Answer ONLY one token: Present, Absent, or Unclear.\n"
                    f"Is there clear visual evidence that {organ} is present in this image?"
                )

            gate = normalize_text(self._ask_once_mm(image, gate_q, temperature=0.0, max_tokens=2)).lower()
            if gate == "present":
                return GenOut(texts=["Yes"]*n, voted="Yes", raw={"gate":"Present"}, qtype=qtype)
            if gate == "absent":
                return GenOut(texts=["No"]*n, voted="No", raw={"gate":"Absent"}, qtype=qtype)
            return GenOut(texts=["Unknown"]*n, voted="Unknown", raw={"gate":"Unclear"}, qtype=qtype)

        # -------- keep healthy gate conservative --------
        if use_gate and qtype == "healthy_yesno":
            gate_q = (
                "Answer ONLY one token: Abnormal, NoAbnormal, or Unclear.\n"
                "Is there any visible abnormality in the lung on this image?"
            )
            gate = normalize_text(self._ask_once_mm(image, gate_q, temperature=0.0, max_tokens=2)).lower()
            if gate == "abnormal":
                return GenOut(texts=["No"]*n, voted="No", raw={"gate":"Abnormal"}, qtype=qtype)
            if gate == "noabnormal":
                return GenOut(texts=["Yes"]*n, voted="Yes", raw={"gate":"NoAbnormal"}, qtype=qtype)
            return GenOut(texts=["Unknown"]*n, voted="Unknown", raw={"gate":"Unclear"}, qtype=qtype)

        # -------- abnormal yes/no gate --------
        if use_gate and qtype == "abnormal_yesno":
            gate_q = (
                "Answer ONLY one token: Abnormal, NoAbnormal, or Unclear.\n"
                "Is there any visible abnormality in this image?"
            )
            gate = normalize_text(self._ask_once_mm(image, gate_q, temperature=0.0, max_tokens=2)).lower()
            if gate == "abnormal":
                return GenOut(texts=["Yes"]*n, voted="Yes", raw={"gate":"Abnormal"}, qtype=qtype)
            if gate == "noabnormal":
                return GenOut(texts=["No"]*n, voted="No", raw={"gate":"NoAbnormal"}, qtype=qtype)
            return GenOut(texts=["Unknown"]*n, voted="Unknown", raw={"gate":"Unclear"}, qtype=qtype)

        # -------- fallback: just ask once with rules (and replicate n) --------
        # (你现在主要用 gate 的 qtype，其他题不需要多采样了；先简单稳定)
        img_url = pil_to_data_url(image)
        if qtype in ("contain_yesno", "healthy_yesno", "abnormal_yesno"):
            sample_temp = 0.2
            sample_top_p = min(float(top_p), 0.9)
        else:
            sample_temp = float(temperature)
            sample_top_p = float(top_p)
        payload = {
            "model": self.model,
            "n": int(max(1, n)),
            "temperature": sample_temp,
            "top_p": sample_top_p,
            "max_tokens": max_tokens,
            "messages": [{
                "role": "user",
                "content": [
                    {"type":"text","text": f"{rules}\nQuestion: {question}"},
                    {"type":"image_url","image_url":{"url": img_url}},
                ],
            }],
        }
        raw = self._post(payload)
        texts = [c["message"]["content"].strip() for c in raw.get("choices", [])]
        if not texts:
            texts = ["Unknown"] * int(max(1, n))
        norm_texts = [normalize_answer(t, qtype) for t in texts]
        voted = max(norm_texts, key=norm_texts.count)
        return GenOut(texts=texts, voted=voted, raw=raw, qtype=qtype)
