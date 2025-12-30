# build_index.py
from rag_core import RAGIndex, EMBED_DIM # EMBED_DIM도 import 합니다.

if __name__ == "__main__":
    print(">>> RAG 인덱스 생성 시작")

    # RAGIndex 생성 시 불필요한 파라미터를 모두 제거합니다.
    idx = RAGIndex(dim=EMBED_DIM)

    # 원본 텍스트가 있는 폴더를 지정합니다.
    idx.build_from_folder("data/law_corpus")

    # 생성된 인덱스를 저장합니다.
    idx.save(
        "indexes/law_faiss.index",
        "indexes/law_texts.txt",
        "indexes/law_embs.npy",
    )

    print(">>> 인덱스 생성 완료")
