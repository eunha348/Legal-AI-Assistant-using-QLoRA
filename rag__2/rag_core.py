from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import faiss
import numpy as np
import requests
import json
from sentence_transformers import SentenceTransformer

# --- 설정값 ---
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
# !!! 중요: 이 URL을 반드시 본인의 원격 LLM 서버 주소로 변경하세요. !!!
MODEL_URL = "https://interoffice-lustrative-gustavo.ngrok-free.dev/generate"
MAX_CONTEXT_CHARS = 3000
MAX_DOC_CHARS = 700
MAX_WEB_CHARS = 1000

# --- 프롬프트 ---
SYSTEM_INST = "당신은 한국 법률 전문가 AI 어시스턴트입니다. 주어진 자료와 법률 지식을 바탕으로 질문에 대해 명확하고 논리적으로 답변하세요."

SELF_CORRECTION_PROMPT = """
너는 법률 QA 시스템의 답변을 검증하는 '내부 감사관' 역할이다.
주어진 (질문)과 (모델의 1차 답변)을 보고, 이 답변이 다음 기준 중 하나에 해당하는지 판단해라.

[검증 기준]
1. 환각(Hallucination): 질문과 관련 없거나, 근거 없이 사실인 것처럼 꾸며낸 내용이 있는가?
2. 회피(Evasion): "모르겠다", "답변할 수 없다" 등 직접적인 답변을 피하고 있는가?
3. 불충분(Insufficient): 답변이 너무 짧거나, 질문의 핵심을 다루지 못하고 있는가?

위 기준 중 하나라도 해당하면 "REJECT", 그렇지 않고 답변이 합리적이고 자신감 있어 보이면 "ACCEPT" 라고만 답해라.
오직 "ACCEPT" 또는 "REJECT" 한 단어로만 출력해야 한다.

(질문)
{question}

(모델의 1차 답변)
{answer}
"""

EVAL_PROMPT = """
너는 한국 법률 QA 시스템의 '평가자' 역할이다. (질문), (모델의 답변), (법령/근거 텍스트)를 보고 모델의 답변 품질을 -1.0 ~ +1.0 사이 숫자로 평가해라.
기준:
- +1.0: 근거 텍스트와 거의 완전히 일치. 중요한 누락/오류 없음.
- +0.5 ~ +0.9: 전반적으로 맞지만, 표현/세부에서 작은 문제 있음.
- 0.0: 근거만으로는 확실하지 않거나, "모르겠다"처럼 정직하지만 정보는 부족한 상태.
- -0.1 ~ -0.5: 일부만 맞고, 중요한 부분이 틀리거나 빠져 있음.
- -0.6 ~ -1.0: 질문과 동떨어져 있거나, 근거에 없는 내용을 단정적으로 말함(환각).
특히 '근거 텍스트에 없는 내용을 단정적으로 말하는 것'은 강하게 감점한다.
출력은 반드시 JSON 한 줄만으로: {"reward": 0.7, "reason": "핵심 내용을 잘 요약함."}

(질문)
{question}

(모델의 답변)
{answer}

(법령/근거 텍스트)
{evidence_text}
"""

# --- 유틸 함수 ---
def _trim(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text[:limit] + " ..." if len(text) > limit else text

# --- RAG 인덱스 클래스 ---
class RAGIndex:
    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim
        self.encoder = SentenceTransformer(EMBED_MODEL_NAME)
        self.index: Optional[faiss.Index] = None
        self.texts: List[str] = []
        self.embs: Optional[np.ndarray] = None

    def build_from_folder(self, folder: str, chunk_size: int = 1000):
        folder_path = Path(folder)
        texts: List[str] = []
        for path in folder_path.glob("*.txt"):
            raw = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not raw: continue
            for i in range(0, len(raw), chunk_size):
                chunk = raw[i:i + chunk_size].strip()
                if len(chunk) >= 50: texts.append(chunk)
        
        if not texts: raise ValueError(f"'{folder}'에서 유효한 텍스트를 찾지 못했습니다.")
        
        embs = self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        faiss.normalize_L2(embs)
        
        index = faiss.IndexFlatIP(self.dim)
        index.add(embs)
        
        self.texts, self.embs, self.index = texts, embs, index

    def save(self, index_path: str, texts_path: str, embs_path: Optional[str] = None):
        if self.index is None or not self.texts: raise ValueError("인덱스 또는 텍스트가 비어있습니다.")
        faiss.write_index(self.index, index_path)
        Path(texts_path).write_text("\n<<<DOC_SEP>>>\n".join(self.texts), encoding="utf-8")
        if embs_path and self.embs is not None: np.save(embs_path, self.embs)

    def load(self, index_path: str, texts_path: Optional[str] = None, embs_path: Optional[str] = None):
        self.index = faiss.read_index(index_path)
        if texts_path and Path(texts_path).exists():
            raw = Path(texts_path).read_text(encoding="utf-8", errors="ignore")
            self.texts = [t for t in raw.split("\n<<<DOC_SEP>>>\n") if t.strip()]
        if embs_path and Path(embs_path).exists(): self.embs = np.load(embs_path)

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.4) -> List[Dict]:
        if self.index is None: return []
        
        q_emb = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        
        D, I = self.index.search(q_emb, top_k * 2)
        
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx == -1 or score < score_threshold or idx >= len(self.texts): continue
            results.append({"content": self.texts[idx], "score": float(score)})
            if len(results) >= top_k: break
        return results

# --- RAG + LLM 래퍼 클래스 ---
@dataclass
class RAGLLM:
    index: RAGIndex

    def answer(self, question: str, use_rag: bool = True, extra_context: Optional[str] = None) -> Tuple[str, List[Dict]]:
        docs = []
        if use_rag and self.index:
            docs = self.index.retrieve(question, top_k=5)

        prompt = self._build_prompt(question, docs, extra_context)
        answer = self._call_remote_model(prompt, max_new_tokens=1024)

        if not answer.strip():
            answer = "죄송합니다. 현재 이 질문에 대한 답변을 생성할 수 없습니다. 다른 방식으로 질문해주시겠어요?"
        
        return answer, docs

    def self_correct(self, question: str, answer: str) -> bool:
        print("[Agent] 자기 교정(Self-Correction) 시작...")
        prompt = SELF_CORRECTION_PROMPT.format(question=question, answer=answer)
        result = self._call_remote_model(prompt, max_new_tokens=10).strip().upper()
        print(f"[Agent] 자기 교정 결과: {result}")
        return result == "ACCEPT"

    def evaluate(self, question: str, answer: str, docs: List[Dict], extra_context: Optional[str] = None) -> Dict:
        evidence_chunks = [d.get("content", "") for d in docs]
        if extra_context: evidence_chunks.append(extra_context)
        evidence_text = "\n\n".join(chunk.strip() for chunk in evidence_chunks if chunk.strip())

        prompt = EVAL_PROMPT.format(question=question, answer=answer, evidence_text=_trim(evidence_text, 4000))
        
        raw_eval = self._call_remote_model(prompt, max_new_tokens=256)
        try:
            data = json.loads(raw_eval)
            reward = float(data.get("reward", 0.0))
            reason = str(data.get("reason", "평가 실패"))
            return {"reward": max(-1.0, min(1.0, reward)), "reason": reason}
        except (json.JSONDecodeError, TypeError):
            return {"reward": 0.0, "reason": "평가 JSON 파싱 실패"}

    def _build_prompt(self, question: str, docs: List[Dict], extra_context: Optional[str]) -> str:
        ctx_blocks = []
        used_chars = 0

        for i, d in enumerate(docs):
            content = (d.get("content") or "").strip()
            if not content: continue
            block = f"[RAG 문서 {i+1}]\n{_trim(content, MAX_DOC_CHARS)}\n"
            if used_chars + len(block) > MAX_CONTEXT_CHARS: break
            ctx_blocks.append(block)
            used_chars += len(block)

        if extra_context:
            block = f"[웹 검색 요약]\n{_trim(extra_context, MAX_WEB_CHARS)}\n"
            if used_chars + len(block) <= MAX_CONTEXT_CHARS:
                ctx_blocks.append(block)

        ctx_str = "\n".join(ctx_blocks) if ctx_blocks else "※ 참고 자료 없음"
        
        return f"{SYSTEM_INST}\n\n--- 참고 자료 ---\n{ctx_str}\n\n--- 질문 ---\n{question}\n\n--- 답변 ---\n"

    def _call_remote_model(self, prompt: str, max_new_tokens: int) -> str:
        try:
            resp = requests.post(
                MODEL_URL, 
                json={"prompt": prompt, "max_new_tokens": max_new_tokens}, 
                timeout=600  # 타임아웃 10분
            )
            resp.raise_for_status()
            return (resp.json().get("text") or "").strip()
        except Exception as e:
            print(f"[ERROR] 모델 서버 호출 실패: {e}")
            return ""