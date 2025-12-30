# policy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Dict

import torch
from sentence_transformers import SentenceTransformer
import torch.nn as nn

# 액션 타입을 3가지로 확장
ActionType = Literal["LLM_ONLY", "RAG_ONLY", "WEB_RAG"]

@dataclass
class AgentAction:
    type: ActionType
    reason: str

class SimplePolicy:
    """
    룰 기반 정책 (fallback 및 초기 학습 데이터 생성용).
    """
    def decide(self, question: str) -> AgentAction:
        q = (question or "").strip().lower()
        if not q:
            return AgentAction(type="LLM_ONLY", reason="질문이 비어 있음")

        # 룰 기반도 3가지 액션을 결정하도록 고도화
        if any(k in q for k in ["최신", "2025년", "개정안", "최근 판례"]):
            return AgentAction(type="WEB_RAG", reason="최신 정보 키워드 감지")
        if any(k in q for k in ["제28조", "법 조항", "내용은?"]):
            return AgentAction(type="RAG_ONLY", reason="특정 법 조항 키워드 감지")
        
        return AgentAction(type="LLM_ONLY", reason="일반적인 질문으로 판단됨")

class SmallPolicyNet(nn.Module):
    """
    질문 임베딩 -> (LLM_ONLY / RAG_ONLY / WEB_RAG) 로짓 3차원.
    """
    def __init__(self, emb_dim: int = 384, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            # 출력 차원을 3으로 변경
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, x):
        return self.net(x)

class LearnedPolicy:
    """
    학습된 정책을 로드해서 사용하는 클래스.
    """
    ACTION_MAP: Dict[int, ActionType] = {0: "LLM_ONLY", 1: "RAG_ONLY", 2: "WEB_RAG"}

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        self.device = device
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        emb_dim = self.encoder.get_sentence_embedding_dimension()

        self.model = SmallPolicyNet(emb_dim=emb_dim)
        try:
            state = torch.load(ckpt_path, map_location=self.device)
            if isinstance(state, dict) and "net.0.weight" in state:
                self.model.load_state_dict(state)
            else:
                self.model = state
        except FileNotFoundError:
            raise FileNotFoundError(f"정책 체크포인트 파일을 찾을 수 없습니다: {ckpt_path}")
            
        self.model.to(self.device)
        self.model.eval()

    def decide(self, question: str) -> AgentAction:
        q = (question or "").strip()
        if not q:
            return AgentAction(type="LLM_ONLY", reason="질문이 비어 있어 기본값 사용")

        emb = self.encoder.encode([q], convert_to_tensor=True).to(self.device)

        with torch.no_grad():
            logits = self.model(emb)
            probs = torch.softmax(logits, dim=-1)
            cls = int(torch.argmax(probs, dim=-1))
        
        action_type = self.ACTION_MAP.get(cls, "LLM_ONLY")
        reason = f"학습된 정책 판단 (class: {cls}, prob: {probs[0][cls]:.2f})"
        
        return AgentAction(type=action_type, reason=reason)
