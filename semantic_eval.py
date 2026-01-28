from typing import Iterable, Optional, Tuple

DEFAULT_SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _iter_answers(ans) -> Iterable[str]:
    if ans is None:
        return []
    if isinstance(ans, (list, tuple)):
        out = []
        for a in ans:
            if a is None:
                continue
            text = str(a).strip()
            if text:
                out.append(text)
        return out
    text = str(ans).strip()
    return [text] if text else []


class SemanticScorer:
    def __init__(
        self,
        enabled: bool = True,
        model_name: str = DEFAULT_SEMANTIC_MODEL,
        device: str = "auto",
        max_length: int = 128,
    ) -> None:
        self.enabled = enabled
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._loaded = False
        self._available = True
        self._warned = False
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device = None

    def _warn_once(self, message: str) -> None:
        if not self._warned:
            print(message)
            self._warned = True

    def _ensure_loaded(self) -> bool:
        if not self.enabled:
            return False
        if self._loaded:
            return True
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:
            self._available = False
            self._warn_once(f"Semantic scorer disabled (missing deps): {exc}")
            return False
        self._torch = torch
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
            self._model.to(device)
        except Exception as exc:
            self._available = False
            self._warn_once(f"Semantic scorer disabled (model load failed): {exc}")
            return False
        self._device = device
        self._loaded = True
        return True

    def max_similarity(self, pred: str, gt) -> Optional[float]:
        if not pred or gt is None:
            return None
        if not self._ensure_loaded():
            return None
        gt_texts = list(_iter_answers(gt))
        if not gt_texts:
            return None
        texts = [str(pred)] + gt_texts
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
        token_embeddings = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = (token_embeddings * mask).sum(1)
        sum_mask = mask.sum(1).clamp(min=1e-9)
        embeddings = sum_embeddings / sum_mask
        embeddings = self._torch.nn.functional.normalize(embeddings, p=2, dim=1)
        pred_emb = embeddings[0:1]
        gt_embs = embeddings[1:]
        sims = (pred_emb @ gt_embs.T).squeeze(0)
        return float(sims.max().item())


def semantic_eval(
    pred: str,
    gt,
    scorer: SemanticScorer,
    threshold: float = 0.75,
) -> Tuple[Optional[bool], Optional[float]]:
    score = scorer.max_similarity(pred, gt)
    if score is None:
        return None, None
    return score >= threshold, score
