# agent.py
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List

from rag_core import RAGIndex, RAGLLM, _trim
from search_tools import web_search_for_agent
from policy import SimplePolicy, LearnedPolicy, ActionType

RL_LOG_PATH = Path("logs/rl_logs.jsonl")
MAX_DOC_LOG_CHARS = 800

class Agent:
    def __init__(self) -> None:
        self.index = RAGIndex(dim=384)
        self.llm = RAGLLM(self.index)

        try:
            self.index.load("indexes/law_faiss.index", "indexes/law_texts.txt")
        except Exception as e:
            print(f"[WARN] RAG 인덱스 로딩 실패: {e}.")
            
        ckpt_path = Path("policy_ckpt.pt")
        if ckpt_path.exists():
            print("[Agent] 학습된 정책(LearnedPolicy) 사용")
            self.policy = LearnedPolicy(str(ckpt_path))
        else:
            print("[Agent] 룰 기반 정책(SimplePolicy) 사용. 'policy_ckpt.pt'가 없어 fallback합니다.")
            self.policy = SimplePolicy()

    def answer(self, question: str) -> Tuple[str, Dict]:
        action = self.policy.decide(question)
        print(f"[Agent] 정책 결정 액션: {action.type} | 이유: {action.reason}")

        final_action = action.type
        docs: List[Dict] = []
        extra_ctx: Optional[str] = None
        used_web = False

        if action.type == "LLM_ONLY":
            answer, _ = self.llm.answer(question, use_rag=False)
        elif action.type == "RAG_ONLY":
            answer, docs = self.llm.answer(question, use_rag=True)
        else: # WEB_RAG
            extra_ctx = web_search_for_agent(question)
            used_web = bool(extra_ctx)
            answer, docs = self.llm.answer(question, use_rag=True, extra_context=extra_ctx)

        if action.type != "WEB_RAG":
            if not self.llm.self_correct(question, answer):
                print("[Agent] 자기 교정 실패. WEB_RAG로 강제 전환하여 재탐색...")
                final_action = "WEB_RAG"
                extra_ctx = web_search_for_agent(question)
                used_web = bool(extra_ctx)
                answer, docs = self.llm.answer(question, use_rag=True, extra_context=extra_ctx)

        try:
            eval_result = self.llm.evaluate(question, answer, docs, extra_ctx)
            reward = float(eval_result.get("reward", 0.0))
            eval_reason = str(eval_result.get("reason", ""))
        except Exception as e:
            reward, eval_reason = 0.0, f"evaluation error: {e}"

        self._log_rl(
            question=question, initial_action=action.type, final_action=final_action,
            used_web=used_web, answer=answer, reward=reward,
            eval_reason=eval_reason, docs=docs,
        )

        meta = {"initial_action": action.type, "final_action": final_action, "reward": reward}
        return answer, meta

    def _log_rl(self, **kwargs) -> None:
        RL_LOG_PATH.parent.mkdir(exist_ok=True)
        log_item = {k: v for k, v in kwargs.items() if k != "docs"}
        docs = kwargs.get("docs")
        if docs:
            log_item["docs"] = [{"content": _trim(d.get("content", ""), MAX_DOC_LOG_CHARS), "score": d.get("score")} for d in docs]

        with RL_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_item, ensure_ascii=False) + "\n")

    def _update_last_reward(self, new_reward: float) -> None:
        # 마지막으로 기록된 로그가 없으면 아무것도 하지 않음
        if not self.last_log_info:
            print("[Agent] 수정할 이전 로그 정보가 없습니다.")
            return

        position = self.last_log_info['position']
        log_item = self.last_log_info['content']
        
        # 보상 점수 수정
        original_reward = log_item.get('reward', 0.0)
        log_item['reward'] = new_reward
        log_item['eval_reason'] = f"사용자 피드백으로 보상 수정 (원래 점수: {original_reward})"

        try:
            # 파일을 읽고 쓰기 모드('r+')로 열어 해당 위치의 로그를 덮어씀
            with RL_LOG_PATH.open("r+", encoding="utf-8") as f:
                f.seek(position)
                # 덮어쓸 내용 뒤에 줄바꿈 추가
                new_line = json.dumps(log_item, ensure_ascii=False) + "\n"
                f.write(new_line)
            print(f"[Agent] 사용자 피드백 반영: 이전 로그의 보상을 {new_reward}로 수정했습니다.")
            # 한 번 수정한 로그는 다시 수정하지 않도록 초기화
            self.last_log_info = {}
        except Exception as e:
            print(f"[ERROR] 로그 파일 수정 중 오류 발생: {e}")

