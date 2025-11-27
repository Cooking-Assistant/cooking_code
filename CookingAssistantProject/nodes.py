# nodes.py
from state import State, Hit, Nutrition, Prefs
from typing import Dict, Any, List, Optional
from pathlib import Path
import os
import openai
import faiss
import time, json, re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from state import Hit

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "data" / "test30" 
FAISS_PATH = INDEX_DIR / "recipes30.faiss"
ROWMAP_PATH = INDEX_DIR / "rows30.map.csv"
RECIPES_PATH = INDEX_DIR / "recipes30_clean.jsonl"

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

def _get_chef_llm():
    global _chef_llm
    if _chef_llm is None:
        _chef_llm = ChatOpenAI(
            model="gpt-4.1-mini", 
            temperature=0.3,
        )
    return _chef_llm

def build_query_from_constraints(cons: Dict[str, Any]) -> str:
    """
    추출된 제약을 기반으로 RAG 검색용 쿼리 문자열 구성
    예) chocolate dessert easy recipe quick 15 min
    """
    tokens: List[str] = []

    # 1) 재료
    for ing in cons.get("main_ingredients", []):
        tokens.append(ing)

    # 2) 카테고리 (dessert, cake, salad 등)
    for cat in cons.get("categories", []):
        tokens.append(cat)

    # 3) 식단
    diet = cons.get("diet")
    if diet == "high_protein":
        tokens.append("high protein")
    elif diet == "low_carb":
        tokens.append("low carb")
    elif diet == "diet":
        tokens.append("healthy")

    # 4) 시간 제한
    time_limit = cons.get("time_limit")
    if time_limit:
        tokens.append("quick")
        tokens.append(f"{time_limit} min")

    # 5) 항상 붙이는 기본 토큰
    tokens.append("easy recipe")
    
    # 혹시 아무것도 못 뽑았을 때는 raw_query도 조금 섞어주기
    if len(tokens) <= 2:
        raw = cons.get("raw_query", "").lower()
        if raw:
            tokens.append(raw)

    return " ".join(tokens)


def rag_search(cons: Dict[str, Any], k: int = 5) -> List[Hit]:
    """
    FAISS 인덱스를 사용해 query와 가장 유사한 레시피를 검색하고,
    cons(재료/카테고리)를 이용해서 필터링/재정렬한 뒤 상위 k개만 반환.
    """
    query = build_query_from_constraints(cons)

    model = _load_embed_model()
    index = _load_faiss()
    rowmap = _load_rowmap()
    recipes = _load_recipes()

    # 1) 쿼리 임베딩
    q = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)

    # 2) 넉넉하게 pool_k개 뽑기 (예: k=5 → 최대 25개)
    pool_k = min(max(k * 5, k), index.ntotal)
    scores, ids = index.search(q, pool_k)
    scores, ids = scores[0], ids[0]

    hits: List[Hit] = []
    for score, idx in zip(scores, ids):
        if idx == -1:
            continue
        
        if 0 <= idx < len(rowmap):
            title = str(rowmap.iloc[idx]["title"])
        else:
            title = f"recipe_{idx}"

        if 0 <= idx < len(recipes):
            rec = recipes[idx]
            text = rec.get("doc_text", "")
        else:
            text = title  # fallback

        hits.append({
            "id": f"r{idx}",          # ★ main.py에서 숫자/ID 둘 다 지원
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
    
    # =========================
    # 1️⃣ cons 기반 필터링 (재료/카테고리)
    # =========================
    filters: List[str] = []

    for ing in cons.get("main_ingredients", []):
        filters.append(ing.lower())

    for cat in cons.get("categories", []):
        filters.append(cat.lower())

    if filters:
        filtered_hits: List[Hit] = []
        for h in hits:
            text_l = (h["title"] + " " + h["text"]).lower()
            # 재료/카테고리 키워드가 하나라도 들어가면 통과
            if any(f in text_l for f in filters):
                filtered_hits.append(h)

        if filtered_hits:
            hits = filtered_hits

    # (필요하면 여기서 bonus 점수 로직 추가 가능)

    return hits[:k]

# ========= Mock Services (질문 처리 등) =========

def llm_extract_constraints(text: str) -> Dict[str, Any]:
    """
    간단한 룰 기반 파서:
    - 주요 재료(main_ingredients)
    - 카테고리(디저트, 샐러드 등)
    - 식단 타입(고단백 등)
    - 시간 제한(몇 분)
    """
    t = text.strip()
    t_lower = t.lower()

    main_ingredients: List[str] = []
    categories: List[str] = []
    diet: Optional[str] = None
    time_limit: Optional[int] = None

    # 1) 재료 키워드
    for k, v in INGREDIENT_KEYWORDS.items():
        if k in t or k in t_lower:
            if v not in main_ingredients:
                main_ingredients.append(v)

    # 2) 카테고리 키워드
    for k, v in CATEGORY_KEYWORDS.items():
        if k in t or k in t_lower:
            if v not in categories:
                categories.append(v)

    # 3) 식단 키워드 (단일 값으로만)
    for k, v in DIET_KEYWORDS.items():
        if k in t or k in t_lower:
            diet = v
            break

    # 4) 시간 제한 (예: "15분", "20 분")
    m = re.search(r"(\d+)\s*분", t)
    if m:
        time_limit = int(m.group(1))

    return {
        "raw_query": t,              # 원본 문장도 같이 보관
        "main_ingredients": main_ingredients,
        "categories": categories,
        "diet": diet,
        "time_limit": time_limit,
    }

# ========= 레시피 추출 및 질의응답 =========

def extract_steps(recipe_text: str) -> List[str]:
    """
    레시피 전체 텍스트에서 실제 조리 단계만 뽑아낸다.
    - 기본 규칙:
      1) 줄 단위로 나눈 뒤
      2) 'Directions:' 이후 부분만 보고
      3) '1. ...', '2. ...' 같이 번호 붙은 줄만 스텝으로 사용
      4) 번호 붙은 줄이 하나도 없으면 Directions: 이후 줄 전체를 스텝으로 사용
    """
    # 1) 줄 단위로 나누고 공백 제거
    lines = [l.strip() for l in recipe_text.splitlines() if l.strip()]

    # 2) 'Directions:' 위치 찾기 (대소문자 무시)
    start_idx = 0
    for i, line in enumerate(lines):
        if line.lower().startswith("directions"):
            start_idx = i + 1
            break

    candidate_lines = lines[start_idx:]  # Directions: 이후만 사용

    # 3) '1. ...', '2. ...' 같은 numbered step만 추출
    steps: List[str] = []
    for line in candidate_lines:
        m = re.match(r"\d+\.\s*(.+)", line)
        if m:
            steps.append(m.group(1).strip())

    # 4) numbered step이 하나도 없으면, Directions: 이후 전체를 step으로 취급
    if not steps:
        steps = candidate_lines

    return steps
    
def llm_answer_chef_question(
    question: str,
    recipe_text: str,
    current_step: Optional[str] = None,
) -> str:
    
    llm = _get_chef_llm()

    system = SystemMessage(content=(
        "당신은 요리 조리 과정을 안내하는 셰프 에이전트입니다.\n"
        "- 사용자의 현재 조리 단계와 전체 레시피를 바탕으로 최대한 정확하게 답변해야 합니다.\n"
        "- 레시피에 없는 내용은 일반적인 요리 상식 범위에서만 추론해야 합니다.\n"
        "- 답변은 한국어로, 2~4문장 정도로 간결하게 해야 합니다.\n"
        "- 위험할 수 있는 조리법(상한 재료 사용, 덜 익힌 닭/돼지고기 등)은 명확히 경고해야 합니다.\n"
    ))

    context_parts = []
    if current_step:
        context_parts.append(f"[현재 스텝]\n{current_step}") # 현재 조리 단계
    if recipe_text:
        context_parts.append(f"[전체 레시피]\n{recipe_text}") # 전체 조리 단계

    context = "\n\n".join(context_parts) if context_parts else "(레시피 정보 없음)"

    human = HumanMessage(content=(
        f"이 단계에서 사용자의 질문은 다음과 같습니다:\n{question}\n\n"
        f"전체 레시피 정보는 다음과 같습니다:\n{context}\n\n"
        "위 정보를 바탕으로 질문에 적절히 답변해주세요."
    ))
    
    messages = [system, human]
    resp = llm.invoke(messages) # LLM 호출 및 답변 생성
    return resp.content.strip() # LLM이 응답한 텍스트만 반환

openai_client = None

def _get_openai_client():
    """
    OpenAI 클라이언트를 초기화하거나 기존 인스턴스를 반환
    """
    global openai_client
    if openai_client is None:
        api_key = "put your api_key"
        # api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        openai_client = openai.OpenAI(api_key=api_key)
    return openai_client

# ========= 영양성분 계산 =========

#지수-compute_nutrition
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
    # 마지막 user 메시지 찾기 (팀원 버전 로직 유지)
    messages = state.get("messages", [])
    last_user = ""
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, dict):
            last_user = last_msg.get("content", "")
        elif hasattr(last_msg, "content"):
            last_user = last_msg.content

    # 1️⃣ 사용자 입력에서 제약 추출
    cons = llm_extract_constraints(last_user)

    # 2️⃣ RAG 검색 쿼리 생성
    query_str = build_query_from_constraints(cons)

    # 3️⃣ 검색 실행 (기본 5개)
    k = state.get("topk", 5)
    hits = rag_search(cons, k=k)

    msg_query = {
        "role": "system",
        "content": f"🔍 검색 쿼리: {query_str}",
    }
    
    recommendations = []
    for i, h in enumerate(hits, 1):
        recommendations.append(f"{i}. [{h['id']}] {h['title']}")

    msg_result = {
        "role": "assistant",
        "content": (
            f"요청을 바탕으로 {len(hits)}개의 요리를 추천합니다.\n\n"
            + "\n".join(recommendations)
            + "\n\n원하시는 레시피를 선택해 주세요.\n"
            "예시: action=choose:r2"
        ),
    }

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
        candidates = state.get("candidates", [])
        recommendations = []
        for i, h in enumerate(candidates, 1):
            recommendations.append(f"{i}. [{h['id']}] {h['title']}")

        return {
            "last_agent": "choose",
            "next_intent": "need_selection",
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        "선택된 레시피가 없어요. 아래 목록에서 선택해 주세요:\n\n"
                        + "\n".join(recommendations)
                        + "\n\n예시: action=choose:r2"
                    ),
                }
            ],
        }

    hit = next((h for h in state.get("candidates", []) if h["id"] == chosen_id), None)
    if not hit:
        candidates = state.get("candidates", [])
        recommendations = []
        for i, h in enumerate(candidates, 1):
            recommendations.append(f"{i}. [{h['id']}] {h['title']}")

        return {
            "last_agent": "choose",
            "next_intent": "need_selection",
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        f"'{chosen_id}'는 유효하지 않은 ID예요. 아래 목록에서 다시 선택해 주세요:\n\n"
                        + "\n".join(recommendations)
                        + "\n\n예시: action=choose:r2"
                    ),
                }
            ],
        }

    steps = extract_steps(hit["text"])

    return {
        "selected_id": hit["id"],
        "recipe_text": hit["text"],
        "steps": steps,
        "step_idx": 0,
        "selection_required": False,
        "last_agent": "choose",
        "next_intent": "cook_next",
        "messages": [
            {
                "role": "assistant",
                "content": (
                    f"✅ '{hit['title']}' 레시피로 진행합니다!\n\n"
                    f"총 {len(steps)}단계의 조리 과정이 있습니다.\n"
                    f"'action=next_step'으로 조리를 시작해 주세요."
                ),
            }
        ],
    }


def chef_agent(state: State) -> State:
    act = (state.get("action") or "").lower()
    steps = state.get("steps", []) or []
    idx = state.get("step_idx", 0)

    # 1) 질문 처리
    if act.startswith("ask:"):
        q = state["action"][len("ask:"):].strip() # user의 질문 추출
        steps = state.get("steps", []) or [] # 현재 레시피 step
        idx = state.get("step_idx", 0) # step index

        # 현재 혹은 직전 스텝을 같이 넘겨주면 답변 품질 ↑
        cur_step = None
        if 0 <= idx - 1 < len(steps):
            cur_step = steps[idx - 1]
        elif 0 <= idx < len(steps):
            cur_step = steps[idx]

        ans = llm_answer_chef_question(
            q, 
            state.get("recipe_text", "") or "",
            current_step=cur_step,
        )
        return {
            "last_agent": "chef",
            "next_intent": "cook_next",
            "messages":[{"role":"assistant","content":f"Q: {q}\nA: {ans}"}],
        }
        
    # 2) 현재 스텝 반복 (레시피 다시 보여줌)
    if act == "repeat_step" and idx > 0 and idx <= len(steps):
        step = steps[idx - 1]
        return {
            "last_agent": "chef",
            "next_intent": "cook_next",
            "messages":[{"role":"assistant","content":f"[Step {idx}/{len(steps)} 다시 안내] {step}"}],
        }
    
    # 3) 이전 스텝으로 이동
    if act == "prev_step":
        if idx <= 1:
            # 이미 첫 단계 이전이면 그냥 첫 단계 또는 안내 메시지
            if steps:
                return {
                    "step_idx": 1,
                    "last_agent": "chef",
                    "next_intent": "cook_next",
                    "messages":[{"role":"assistant","content":f"이미 첫 번째 단계입니다. [Step 1/{len(steps)}] {steps[0]}"}],
                }
            else:
                return {
                    "last_agent": "chef",
                    "next_intent": "cook_next",
                    "messages":[{"role":"assistant","content":"진행 중인 조리 단계가 없어요."}],
                }

        new_idx = idx - 1 # 한 단계 앞으로
        step = steps[new_idx - 1]
        return {
            "step_idx": new_idx,
            "last_agent": "chef",
            "next_intent": "cook_next",
            "messages":[{"role":"assistant","content":f"[Step {new_idx}/{len(steps)}] {step}"}],
        }

    # 4) 다음 스텝 진행
    if act == "next_step" and idx < len(steps):
        step = steps[idx]
        return {
            "step_idx": idx + 1,
            "last_agent": "chef",
            "next_intent": "cook_next",  # 여전히 다음 스텝 가능
            "messages":[{"role":"assistant","content":f"[Step {idx+1}/{len(steps)}] {step}"}],
        }
    
    # 5) stop이 들어오면 바로 종료하고 영양 분석으로 이동
    if act == "stop":
        return {
            "last_agent": "chef",
            "next_intent": "analyze_nutrition",
            "messages":[{"role":"assistant","content":"조리를 마친 것으로 처리할게요. 이제 영양 정보를 계산해 보겠습니다."}],
        }
    
    # 스텝이 끝났거나, 유효한 action이 아닌 경우 바로 종료하고 영양 분석으로 이동
    return {
        "last_agent": "chef",
        "next_intent": "analyze_nutrition",
        "messages":[{"role":"assistant","content":"조리를 마쳤다고 판단했어요. 이제 영양 정보를 계산해 보겠습니다."}],
    }


def nutrition_agent(state: State) -> State:
    nut = compute_nutrition(state.get("recipe_text", "") or "")
    text = (
        f"📊 영양 정보 (대략 추정)\n\n"
        f"• 칼로리: {nut['calories']}kcal\n"
        f"• 단백질: {nut['protein_g']}g\n"
        f"• 지방: {nut['fat_g']}g\n"
        f"• 탄수화물: {nut['carbs_g']}g\n\n"
        f"({nut['note']})"
    )

    return {
        "nutrition": nut,
        "last_agent": "nutrition",
        "next_intent": "write_memory",
        "messages": [{"role": "assistant", "content": text}],
    }


MEMORY_PATH = ROOT / "data" / "user_memory.jsonl"

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
