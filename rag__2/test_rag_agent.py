# test_rag_agent.py
from agent import Agent
import traceback

if __name__ == "__main__":
    agent = Agent()
    print("법률 AI 에이전트가 준비되었습니다. 질문을 입력해주세요.")

    while True:
        try:
            q = input("\n질문 (종료: exit): ")
            if q.strip().lower() in {"exit", "quit"}:
                break
            if not q.strip():
                continue

            ans, meta = agent.answer(q)

            print("\n=== 에이전트 답변 ===\n")
            print(ans)

            # meta 딕셔너리에 키가 있는지 확인 후 출력
            if not meta.get("feedback_processed"):
                print(
                    "\n[DEBUG]",
                    f"initial_action = {meta.get('initial_action')}",
                    f"| final_action = {meta.get('final_action')}",
                    f"| ai_reward = {meta.get('reward', 0.0):.2f}",
                )

        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"\n[CRITICAL ERROR] 예상치 못한 오류가 발생했습니다: {e}")
            traceback.print_exc()
