# nodes.py (통합 버전)

from state import State, Hit, Nutrition, Prefs
from typing import Dict, Any, List, Optional
from pathlib import Path
import faiss
import time, json, re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import openai
import os

# ======== 간단 키워드 매핑 (한글/영어) ========

INGREDIENT_KEYWORDS = {
    # 육류
    "닭": "chicken", "닭가슴살": "chicken breast", "치킨": "chicken",
    "소고기": "beef", "스테이크": "steak", "비프": "beef",
    "돼지고기": "pork", "돼지": "pork", "베이컨": "bacon", "햄": "ham",
    "소시지": "sausage", "양고기": "lamb",

    # 해산물
    "새우": "shrimp", "연어": "salmon", "참치": "tuna", "생선": "fish",
    "게": "crab", "조개": "clam",

    # 채소/과일
    "양파": "onion", "마늘": "garlic", "파": "green onion",
    "감자": "potato", "고구마": "sweet potato", "토마토": "tomato",
    "버섯": "mushroom", "당근": "carrot", "시금치": "spinach",
    "옥수수": "corn", "브로콜리": "broccoli", "아보카도": "avocado",
    "사과": "apple", "바나나": "banana", "딸기": "strawberry", "레몬": "lemon",

    # 유제품/알
    "계란": "egg", "달걀": "egg", "치즈": "cheese", "우유": "milk",
    "버터": "butter", "크림": "cream", "요거트": "yogurt",

    # 기타
    "초콜릿": "chocolate", "초코": "chocolate", "쌀": "rice",
    "밥": "rice", "면": "noodle", "파스타": "pasta", "빵": "bread",
    "설탕": "sugar", "소금": "salt", "두부": "tofu", "김치": "kimchi",
}

CATEGORY_KEYWORDS = {
    "디저트": "dessert",
    "디저트류": "dessert",
    "dessert": "dessert",
    "케이크": "cake",
    "cake": "cake",
    "쿠키": "cookie",
    "cookie": "cookie",
    "음료": "drink",
    "drink": "drink",
    "샐러드": "salad",
    "salad": "salad",
}

DIET_KEYWORDS = {
    "다이어트": "diet",
    "저탄수": "low_carb",
    "저탄수화물": "low_carb",
    "고단백": "high_protein",
}

# ======== 경로 설정 ========

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "data" / "test30"
FAISS_PATH = INDEX_DIR / "recipes30.faiss"
ROWMAP_PATH = INDEX_DIR / "rows30.map.csv"
RECIPES_PATH = INDEX_DIR / "recipes30_clean.jsonl"

_embed_model = None
_faiss = None
_rowmap = None
_recipes = None

# ======== OpenAI 클라이언트 (지수 코드) ========

openai_client = None

def _get_openai_client():
    """
    OpenAI 클라이언트를 초기화하거나 기존 인스턴스를 반환
    (지수 버전에서 가져온 코드)
    """
    global openai_client
    if openai_client is None:
        api_key = "put your api_key"
        # api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        openai_client = openai.OpenAI(api_key=api_key)
    return openai_client

# ======== 공통 로더 ========

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

# ======== 쿼리 빌더 / 검색 ========

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


def llm_answer_chef_question(question: str, recipe_text: str) -> str:
    if "마늘" in question and "양파" in question:
        return "맛은 달라지지만 사용 가능해요. 향은 약해지고 단맛이 올라갑니다. 양파를 잘게 썰어 초반에 충분히 볶아주세요."
    return "가능은 하지만, 간·조리 시간은 상황에 맞게 조금씩 조정해주세요."

# ========= 지수 버전 compute_nutrition (OpenAI) =========

def compute_nutrition(recipe_text: str) -> Nutrition:
    """
    OpenAI API를 사용하여 레시피에서 영양 정보를 추출합니다.
    (지수 버전 코드 그대로 사용)
    """
    try:
        client = _get_openai_client()

        prompt = f"""
다음은 요리 레시피입니다. 이 레시피를 바탕으로 1인분 기준의 영양 정보를 정확하게 분석해주세요.

레시피:
{recipe_text}

다음 JSON 형식으로 응답해주세요:
{{
    "calories": 칼로리(float),
    "protein_g": 단백질_그램(float),
    "fat_g": 지방_그램(float),
    "carbs_g": 탄수화물_그램(float),
    "note": "분석_방법_또는_주의사항"
}}

주의사항:
- 1인분 기준으로 계산해주세요
- 일반적인 재료의 양을 가정하여 계산해주세요
- 조리 방법도 고려하여 칼로리를 계산해주세요
- 숫자만 정확히 입력하고, JSON 형식을 정확히 지켜주세요
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 영양학 전문가입니다. 요리 레시피를 분석하여 정확한 영양 정보를 제공합니다.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=500,
        )

        response_text = response.choices[0].message.content.strip()

        try:
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            nutrition_data = json.loads(response_text)

            return {
                "calories": float(nutrition_data.get("calories", 0.0)),
                "protein_g": float(nutrition_data.get("protein_g", 0.0)),
                "fat_g": float(nutrition_data.get("fat_g", 0.0)),
                "carbs_g": float(nutrition_data.get("carbs_g", 0.0)),
                "note": str(nutrition_data.get("note", "OpenAI API로 분석됨")),
            }

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"JSON 파싱 에러: {e}")
            print(f"원본 응답: {response_text}")
            return {
                "calories": 500.0,
                "protein_g": 20.0,
                "fat_g": 15.0,
                "carbs_g": 60.0,
                "note": f"OpenAI 응답 파싱 실패 - 기본값 사용 (에러: {str(e)})",
            }

    except Exception as e:
        print(f"OpenAI API 호출 에러: {e}")
        return {
            "calories": 550.0,
            "protein_g": 20.0,
            "fat_g": 15.0,
            "carbs_g": 70.0,
            "note": f"OpenAI API 호출 실패 - 기본값 사용 (에러: {str(e)})",
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

    steps = [s for s in hit["text"].splitlines() if s.strip()]

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
        q = state["action"][len("ask:") :].strip()
        ans = llm_answer_chef_question(q, state.get("recipe_text", ""))
        return {
            "last_agent": "chef",
            "next_intent": "cook_next",
            "messages": [
                {
                    "role": "assistant",
                    "content": f"💬 Q: {q}\n\n📝 A: {ans}",
                }
            ],
        }

    # 2) 다음 스텝 진행
    if act == "next_step" and idx < len(steps):
        step = steps[idx]
        progress = f"[{idx+1}/{len(steps)}]"
        return {
            "step_idx": idx + 1,
            "last_agent": "chef",
            "next_intent": "cook_next",
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        f"👨‍🍳 {progress} {step}\n\n"
                        f"{'다음 단계로: action=next_step' if idx+1 < len(steps) else '조리 완료! action=stop으로 마무리하세요.'}"
                    ),
                }
            ],
        }

    # 3) 스텝 종료 → 영양 분석
    return {
        "last_agent": "chef",
        "next_intent": "analyze_nutrition",
        "messages": [
            {
                "role": "assistant",
                "content": "🎉 조리를 마쳤다고 판단했어요. 이제 영양 정보를 계산할게요.",
            }
        ],
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
        "messages": [
            {
                "role": "assistant",
                "content": "💾 이번 요리 기록을 저장했어요. 이용해 주셔서 감사합니다!",
            }
        ],
    }
