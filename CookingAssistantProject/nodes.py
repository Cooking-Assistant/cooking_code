# nodes.py
from state import State, Hit, Nutrition, Prefs
from typing import Dict, Any, List, Optional
from pathlib import Path
import faiss
import time, json, re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from state import Hit

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "data" / "test30"
FAISS_PATH = INDEX_DIR / "recipes30.faiss"         # 실제 파일명에 맞춰 조정
ROWMAP_PATH = INDEX_DIR / "rows30.map.csv"         # 실제 파일명에 맞춰 조정
RECIPES_PATH = INDEX_DIR / "recipes30_clean.jsonl" # 여기에 네 jsonl

_embed_model = None
_faiss = None
_rowmap = None
_recipes = None

def _load_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embed_model

def _load_faiss():
    global _faiss
    if _faiss is None:
        _faiss = faiss.read_index(str(FAISS_PATH))
    return _faiss


def _load_rowmap():
    """rows30.map.csv: 인덱스 번호 → title 매핑"""
    global _rowmap
    if _rowmap is None:
        _rowmap = pd.read_csv(ROWMAP_PATH)
    return _rowmap


def _load_recipes():
    """recipes30_clean.jsonl: 인덱스 번호 → 레시피 전체(doc_text 등)"""
    global _recipes
    if _recipes is None:
        import json
        recs = []
        if not RECIPES_PATH.exists():
            raise FileNotFoundError(f"레시피 파일을 찾을 수 없습니다: {RECIPES_PATH}")
        with open(RECIPES_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        _recipes = recs
    return _recipes

def build_query_from_constraints(cons: Dict[str, Any]) -> str:
    q = cons.get("query", "").lower()
    base = []

    if "chicken" in q or "닭" in q:
        base.append("chicken")
    if "breast" in q or "가슴살" in q:
        base.append("breast")
    if cons.get("diet") == "high_protein":
        base.append("high protein")
    if cons.get("time_limit"):
        base.append(f"quick {cons['time_limit']} min")
    if "dessert" in q or "디저트" in q:
        base.append("dessert")

    base.append("easy recipe")
    return " ".join(base)

def rag_search(cons: Dict[str, Any], k: int = 3) -> List[Hit]:
    """FAISS 인덱스를 사용해 query와 가장 유사한 레시피를 검색"""
    query = build_query_from_constraints(cons)

    model = _load_embed_model()
    index = _load_faiss()
    rowmap = _load_rowmap()
    recipes = _load_recipes()

    # 1) 쿼리 임베딩 (인덱스 만들 때와 동일: all-MiniLM-L6-v2 + L2 normalize)
    q = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)

    # 2) 검색
    scores, ids = index.search(q, k)
    scores, ids = scores[0], ids[0]

    hits: List[Hit] = []
    for score, idx in zip(scores, ids):
        if idx == -1:
            continue

        # title: rowmap의 idx번째 행
        if 0 <= idx < len(rowmap):
            title = str(rowmap.iloc[idx]["title"])
        else:
            title = f"recipe_{idx}"

        # text: recipes30_clean.jsonl의 idx번째 레코드의 doc_text
        if 0 <= idx < len(recipes):
            rec = recipes[idx]
            text = rec.get("doc_text", "")
        else:
            text = title  # fallback

        hits.append({
            "id": str(idx),
            "title": title,
            "score": float(score),
            "text": text,
        })

    if not hits:
        hits.append({
            "id": "fallback",
            "title": "기본 레시피",
            "score": 0.0,
            "text": "재료 준비\n조리\n완성",
        })

    return hits

# ========= Mock Services (나중에 GPT/RAG로 교체) =========
def llm_extract_constraints(text: str) -> Dict[str, Any]:
    diet = "high_protein" if ("단백질" in text or "닭가슴살" in text) else None
    m = re.search(r"(\d+)\s*분", text)
    limit = int(m.group(1)) if m else None
    return {"query": text, "diet": diet, "time_limit": limit}

def llm_answer_chef_question(question: str, recipe_text: str) -> str:
    if "마늘" in question and "양파" in question:
        return "맛은 달라지지만 사용 가능해요. 향은 약해지고 단맛이 올라갑니다. 양파를 잘게 썰어 초반에 충분히 볶아주세요."
    return "가능은 하지만, 간·조리 시간은 상황에 맞게 조금씩 조정해주세요."

def compute_nutrition(recipe_text: str) -> Nutrition:
    return {
        "calories": 550.0,
        "protein_g": 20.0,
        "fat_g": 15.0,
        "carbs_g": 70.0,
        "note": "러프 추정"
    }

def append_jsonl(path: str, event: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


# ========= Agents =========
def planner_agent(state: State) -> State:
    # 마지막 user 메시지 찾기
    last_user = ""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, dict) and m.get("role") == "user":
            last_user = m.get("content", "")
            break

    # 1️⃣ 사용자 입력에서 제약 추출
    cons = llm_extract_constraints(last_user)

    # 2️⃣ RAG 검색 쿼리 생성 (build_query_from_constraints 사용)
    query_str = build_query_from_constraints(cons)

    # 3️⃣ 검색 실행
    k = max(1, state.get("topk", 3))
    hits = rag_search(cons, k=k)

    # 4️⃣ 로그 메시지 (쿼리 + 결과)
    msg_query = {
        "role": "system",
        "content": f"🔍 검색 쿼리: {query_str}",
    }

    msg_result = {
        "role": "assistant",
        "content": (
            "요청을 바탕으로 아래 요리를 추천합니다.\n"
            "원하시는 레시피 ID를 선택해 주세요. (예: action=choose:r2)\n" +
            "\n".join([f"- {h['id']}: {h['title']} (score {h['score']:.2f})" for h in hits])
        ),
    }

    # 5️⃣ 결과 반환 (쿼리 로그도 messages에 함께 추가)
    return {
        "constraints": cons,
        "candidates": hits,
        "selection_required": True,
        "next_intent": "need_selection",
        "last_agent": "planner",
        "messages": [msg_query, msg_result],
    }


def choose_agent(state: State) -> State:
    act = state.get("action") or ""
    m = re.match(r"choose:(\w+)", act)

    chosen_id: Optional[str] = None
    if m:
        chosen_id = m.group(1)
    elif state.get("candidates"):
        chosen_id = state["candidates"][0]["id"]  # fallback: top-1

    if not chosen_id:
        return {
            "last_agent": "choose",
            "next_intent": "need_selection",
            "messages": [{"role":"assistant","content":"선택된 레시피가 없어요. action=choose:<id> 형태로 선택해 주세요."}],
        }

    hit = next((h for h in state.get("candidates", []) if h["id"] == chosen_id), None)
    if not hit:
        return {
            "last_agent": "choose",
            "next_intent": "need_selection",
            "messages":[{"role":"assistant","content":"해당 ID의 후보를 찾지 못했어요. 목록에서 다시 선택해 주세요."}],
        }

    steps = [s for s in hit["text"].splitlines() if s.strip()]

    return {
        "selected_id": hit["id"],
        "recipe_text": hit["text"],
        "steps": steps,
        "step_idx": 0,
        "selection_required": False,
        "last_agent": "choose",
        "next_intent": "cook_next",
        "messages":[{"role":"assistant","content":f"'{hit['title']}' 레시피로 진행합니다. action=next_step 으로 조리를 시작해 주세요."}],
    }


def chef_agent(state: State) -> State:
    act = (state.get("action") or "").lower()
    steps = state.get("steps", []) or []
    idx = state.get("step_idx", 0)

    # 1) 질문 처리
    if act.startswith("ask:"):
        q = state["action"][len("ask:"):].strip()
        ans = llm_answer_chef_question(q, state.get("recipe_text", ""))
        return {
            "last_agent": "chef",
            "next_intent": "cook_next",  # 계속 조리 문맥 유지
            "messages":[{"role":"assistant","content":f"Q: {q}\nA: {ans}"}],
        }

    # 2) 다음 스텝 진행
    if act == "next_step" and idx < len(steps):
        step = steps[idx]
        return {
            "step_idx": idx + 1,
            "last_agent": "chef",
            "next_intent": "cook_next",  # 여전히 다음 스텝 가능
            "messages":[{"role":"assistant","content":f"[Step {idx+1}/{len(steps)}] {step}"}],
        }

    # 3) 스텝이 끝났거나, stop이거나, 유효한 next_step이 아닌 경우 → 영양 분석으로
    return {
        "last_agent": "chef",
        "next_intent": "analyze_nutrition",
        "messages":[{"role":"assistant","content":"조리를 마쳤다고 판단했어요. 이제 영양 정보를 계산할게요."}],
    }


def nutrition_agent(state: State) -> State:
    nut = compute_nutrition(state.get("recipe_text", "") or "")
    text = f"대략 {nut['calories']}kcal / 단백질 {nut['protein_g']}g / 지방 {nut['fat_g']}g / 탄수화물 {nut['carbs_g']}g"

    return {
        "nutrition": nut,
        "last_agent": "nutrition",
        "next_intent": "write_memory",
        "messages":[{"role":"assistant","content":f"[Nutrition] {text} (러프 추정입니다.)"}],
    }


def memory_agent(state: State) -> State:
    ev = {
        "ts": time.time(),
        "recipe_id": state.get("selected_id"),
        "prefs": state.get("prefs", {}),
        "nutrition": state.get("nutrition"),
    }
    append_jsonl("data/user_memory.jsonl", ev)

    return {
        "memory_event": ev,
        "last_agent": "memory",
        "next_intent": "finished",
        "messages":[{"role":"assistant","content":"이번 요리 기록을 저장했어요. 이용해 주셔서 감사합니다."}],
    }
