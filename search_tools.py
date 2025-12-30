# search_tools.py
from __future__ import annotations
from typing import List, Dict
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import socket
import ipaddress


# ─────────────────────────────
#  허용 도메인/범위 정의
# ─────────────────────────────

ALLOWED_DOMAINS = {
    "www.law.go.kr",
    "law.go.kr",
    "www.scourt.go.kr",
    "scourt.go.kr",
    "www.copyright.or.kr",
    "copyright.or.kr",
    "www.kcc.go.kr",
    "kcc.go.kr",
    "www.mcst.go.kr",
    "mcst.go.kr",
    "www.law.go.kr", "law.go.kr",
    "www.copyright.or.kr", "copyright.or.kr",
    "www.mcst.go.kr", "mcst.go.kr",
    "www.easylaw.go.kr", "easylaw.go.kr",
    "www.cros.or.kr", "cros.or.kr",
}

# 필요하면 확장용
ALLOWED_SUFFIXES = (
    ".go.kr",      # 정부기관
    ".or.kr",      # 공공 성격 재단/협회 (저작권위원회 등)
)

# 내부망 / 로컬 주소 방지용
PRIVATE_NETS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
]


def is_official_domain(host: str) -> bool:
    if not host:
        return False
    host = host.lower()

    if host in ALLOWED_DOMAINS:
        return True

    # 필요하면 suffix 기준 허용 (조금 더 느슨한 옵션)
    return any(host.endswith(suf) for suf in ALLOWED_SUFFIXES)


def is_private_host(host: str) -> bool:
    """
    내부망/로컬 IP로 해석되는 호스트는 막기 (SSRF 비슷한 상황 방지)
    """
    try:
        ip_str = socket.gethostbyname(host)
        ip = ipaddress.ip_address(ip_str)
    except Exception:
        # 해석 실패하면 일단 private는 아닌 걸로 취급
        return False

    return any(ip in net for net in PRIVATE_NETS)


# ─────────────────────────────
# 1) 안전한 URL fetch
# ─────────────────────────────
def safe_fetch_url(url: str, timeout: int = 10, max_bytes: int = 2_000_000) -> str:
    """
    - 허용된 도메인 + 프로토콜만 요청
    - 내부망/로컬 IP 차단
    - 리다이렉트 후 최종 도메인도 다시 검사
    - 너무 큰 응답은 잘라냄
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"허용되지 않은 스킴입니다: {parsed.scheme}")

    host = parsed.hostname
    if not is_official_domain(host):
        raise ValueError(f"허용되지 않은 도메인입니다: {host}")

    if is_private_host(host):
        raise ValueError(f"내부망/로컬 IP로 해석되는 도메인입니다: {host}")

    # 요청 (리다이렉트 허용)
    resp = requests.get(url, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()

    # 리다이렉트된 최종 URL도 다시 검증
    final_parsed = urlparse(resp.url)
    final_host = final_parsed.hostname

    if not is_official_domain(final_host):
        raise ValueError(f"리다이렉트 결과 비공식 도메인입니다: {final_host}")

    if is_private_host(final_host):
        raise ValueError(f"리다이렉트 결과 내부망/로컬 IP입니다: {final_host}")

    # 응답 크기 제한 (너무 큰 페이지 방지)
    content = resp.content[:max_bytes]
    text = content.decode(resp.encoding or "utf-8", errors="ignore")
    return text



def html_to_text(html: str, max_chars: int = 5000) -> str:
    """
    HTML에서 메인 본문 텍스트를 최대한 추출해서 반환.
    - script/style/nav/footer/header 제거
    - article/main/#content/.content 우선
    - 없으면 div 중 텍스트가 가장 긴 블록 선택
    - 너무 짧은/메뉴성 라인은 제거
    """
    soup = BeautifulSoup(html, "html.parser")

    # 1) 불필요 태그 제거
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # 2) 본문 후보 선택
    candidates = []

    # article/main/role=main/#content/.content 우선
    for css_sel in ["article", "main", "[role=main]", "#content", ".content"]:
        for node in soup.select(css_sel):
            text = node.get_text(" ", strip=True)
            if len(text) >= 200:  # 너무 짧으면 스킵
                candidates.append(text)

    # 후보가 없으면 div들 중 가장 긴 텍스트 사용
    if not candidates:
        for div in soup.find_all("div"):
            text = div.get_text(" ", strip=True)
            if len(text) >= 200:
                candidates.append(text)

    # 그래도 없으면 전체 페이지 텍스트
    if not candidates:
        raw_text = soup.get_text(" ", strip=True)
    else:
        # 가장 긴 텍스트를 본문으로 간주
        raw_text = max(candidates, key=len)

    # 3) 라인 정제: 너무 짧은 줄/메뉴성 줄 제거
    lines = [ln.strip() for ln in raw_text.splitlines()]
    cleaned_lines = []
    menu_keywords = ("홈", "로그인", "회원가입", "이전", "다음", "사이트맵")

    for ln in lines:
        if len(ln) < 5:
            continue
        if any(mk in ln for mk in menu_keywords):
            continue
        cleaned_lines.append(ln)

    text = "\n".join(cleaned_lines)
    text = text[:max_chars]
    return text



# ─────────────────────────────
# 2) 간단 URL 리스트 기반 검색 (placeholder)
# ─────────────────────────────
def search_predefined_urls(query: str) -> List[Dict]:
    """
    미리 정해둔 공식 사이트 URL 몇 개에서 내용 긁어오는 간단 버전.
    - URL은 candidate_urls에 정의
    - 질문에서 뽑은 키워드 기준으로 '대충 관련 있어 보이면' 결과에 포함
    """
    candidate_urls = [
        # ── 법제처 / 국가법령정보센터: 저작권법 원문 ──
        "https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq=192474",  # 저작권법

        # ── 문화체육관광부: 저작권 정책·상식 ──
        "https://www.mcst.go.kr/web/s_policy/copyright/copyright.jsp",
        "https://mcst.go.kr/site/s_policy/copyright/knowledge/know11.jsp",
        "https://mcst.go.kr/site/s_policy/copyright/knowledge/know12.jsp",

        # ── 한국저작권위원회: FAQ / 연구자료 ──
        "https://www.copyright.or.kr/customer-center/faq/list.do",
        "https://www.copyright.or.kr/search/faq-view.do?counselFaqNo=43935",
        "https://www.copyright.or.kr/information-materials/publication/research-report/view.do?brdctsno=7054",

        # ── 이지로 생활법령 ──
        "https://www.easylaw.go.kr/CSP/CnpClsMain.laf?ccfNo=1&cciNo=1&cnpClsNo=1&csmSeq=695",
        "https://www.easylaw.go.kr/CSP/CnpClsMain.laf?ccfNo=3&cciNo=2&cnpClsNo=4&csmSeq=695",

        # ── 저작권 등록센터 ──
        "https://www.cros.or.kr/",
    ]

    # 질문에서 키워드 후보 추출 (아주 단순하게)
    base_keywords = ["저작권", "이미지", "사진", "블로그", "출처", "인용", "비상업적"]
    question_keywords = [kw for kw in base_keywords if kw in query]

    results: List[Dict] = []

    for url in candidate_urls:
        try:
            html = safe_fetch_url(url)
            text = html_to_text(html)

            # ❗ 필터 로직: 키워드가 하나도 없으면 일단 다 살려두고,
            # 키워드가 있으면 그 중 하나라도 포함된 문서만 선택
            if question_keywords:
                if not any(kw in text for kw in question_keywords):
                    continue

            results.append({"url": url, "text": text})
        except Exception as e:
            print(f"[WARN] URL fetch 실패: {url} ({e})")

    return results


# ─────────────────────────────
# 3) Agent에서 쓸 high-level 함수
# ─────────────────────────────
def web_search_for_agent(question: str, max_sources: int = 2) -> str:
    hits = search_predefined_urls(question)
    print(f"[Agent] 웹 검색 hit 수: {len(hits)}")  # ← 디버그 출력

    if not hits:
        return ""

    pieces = []
    for hit in hits[:max_sources]:
        snippet = hit["text"][:1500]
        pieces.append(f"[웹자료 from {hit['url']}]\n{snippet}")

    return "\n\n".join(pieces)
    
