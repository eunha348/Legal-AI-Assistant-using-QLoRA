# train_policy.py
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sentence_transformers import SentenceTransformer

from policy import SmallPolicyNet, ActionType

ACTION_TO_IDX: Dict[ActionType, int] = {"LLM_ONLY": 0, "RAG_ONLY": 1, "WEB_RAG": 2}

class RLLocalDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor, rewards: torch.Tensor):
        self.embeddings = embeddings
        self.labels = labels
        self.rewards = rewards

    def __len__(self): return len(self.embeddings)
    def __getitem__(self, idx): return self.embeddings[idx], self.labels[idx], self.rewards[idx]

def load_logs(log_path: Path) -> Tuple[List[str], List[int], List[float]]:
    questions, labels, rewards = [], [], []
    if not log_path.exists(): return [], [], []

    with log_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                act = item.get("final_action")
                if item.get("question") and act in ACTION_TO_IDX and item.get("reward") is not None:
                    questions.append(item["question"])
                    labels.append(ACTION_TO_IDX[act])
                    rewards.append(float(item["reward"]))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return questions, labels, rewards

def main():
    parser = argparse.ArgumentParser()
    # ... (argparse 설정은 이전과 동일)
    args = parser.parse_args()
    
    log_path = Path(args.log_path)
    questions, labels, rewards = load_logs(log_path)
    
    if len(questions) < args.min_samples:
        print(f"샘플 수가 너무 적어 학습을 건너뜁니다: {len(questions)}개")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    embeddings = encoder.encode(questions, convert_to_tensor=True, show_progress_bar=True)
    
    dataset = RLLocalDataset(embeddings, torch.tensor(labels, dtype=torch.long), torch.tensor(rewards, dtype=torch.float32))
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = SmallPolicyNet(emb_dim=embeddings.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"[train_policy] {len(questions)}개 샘플로 학습 시작")
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        for X, y, r in train_loader:
            X, y, r = X.to(device), y.to(device), r.to(device)
            optimizer.zero_grad()
            
            logits = model(X)
            log_probs = torch.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(1, y.unsqueeze(1)).squeeze(1)
            
            loss = -(action_log_probs * r).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"[Epoch {epoch}/{args.epochs}] Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), args.ckpt_path)
    print(f"정책 모델 저장 완료: {args.ckpt_path}")

if __name__ == "__main__":
    main()
