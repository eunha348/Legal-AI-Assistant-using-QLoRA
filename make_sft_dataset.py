# make_sft_dataset.py
import json
from pathlib import Path
from collections import defaultdict

LOG_PATH = Path("logs/rl_logs.jsonl")
OUT_DIR = Path("data/finetune")
SFT_OUT_PATH = OUT_DIR / "legal_sft.jsonl"
DPO_OUT_PATH = OUT_DIR / "legal_dpo.jsonl"

def main():
    if not LOG_PATH.exists():
        print(f"로그 파일이 없습니다: {LOG_PATH}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    question_answers = defaultdict(list)

    with LOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                question_answers[item["question"]].append({
                    "answer": item["answer"],
                    "reward": float(item["reward"])
                })
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

    sft_records, dpo_records = [], []

    # SFT 및 DPO 데이터셋 생성
    for q, answers in question_answers.items():
        if not answers: continue
        
        answers.sort(key=lambda x: x["reward"], reverse=True)
        best_answer = answers[0]
        
        # SFT 데이터: 가장 좋은 답변이 특정 점수 이상일 때만 추가
        if best_answer["reward"] > 0.5:
            instruction = "다음 질문에 대해 한국 법률을 기준으로 설명해 주세요."
            sft_records.append({"instruction": instruction, "input": q, "output": best_answer["answer"]})

        # DPO 데이터: 가장 좋은 답변과 나쁜 답변의 쌍을 만듦
        if len(answers) >= 2:
            worst_answer = answers[-1]
            # 보상 차이가 충분히 나야 의미있는 쌍
            if best_answer["reward"] > worst_answer["reward"] + 0.3:
                dpo_records.append({
                    "prompt": q,
                    "chosen": best_answer["answer"],
                    "rejected": worst_answer["answer"],
                })

    # 파일 저장
    with SFT_OUT_PATH.open("w", encoding="utf-8") as f:
        for rec in sft_records: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"SFT 샘플 {len(sft_records)}개 저장 완료: {SFT_OUT_PATH}")

    with DPO_OUT_PATH.open("w", encoding="utf-8") as f:
        for rec in dpo_records: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"DPO 샘플 {len(dpo_records)}개 저장 완료: {DPO_OUT_PATH}")

if __name__ == "__main__":
    main()
