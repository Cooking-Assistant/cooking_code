from typing import TypedDict, List, Dict, Any, Optional, Annotated, Literal
from operator import add
from langgraph.graph.message import add_messages

class Hit(TypedDict): # RAG 검색 결과 한 건 
    id: str
    title: str
    score: float
    text: str

class Nutrition(TypedDict, total=False): # total=False: 일부만 채워도 ok
    calories: float
    protein_g: float
    fat_g: float
    carbs_g: float
    note: str

class Prefs(TypedDict, total=False): # 사용자 선호 
    disliked: List[str]
    liked: List[str]
    diet: Optional[str]

# Supervisor가 보고 라우팅할 수 있는 의도 타입
NextIntent = Literal[
    "need_selection",     # 후보 추천 후, 선택 필요
    "cook_next",          # 조리 계속 진행
    "user_question",      # 질문 처리 필요
    "analyze_nutrition",  # 영양 분석 단계로
    "write_memory",       # 메모리 기록 단계로
    "finished",           # 전체 세션 종료
    "retry",              # 재시도
    "fallback"            # 대체 경로
]

class State(TypedDict, total=False):
    # 공용 대화 로그(누적)
    messages: Annotated[list, add_messages]

    # Planner / 후보 추천
    constraints: Dict[str, Any] # 사용자의 요구사항 (LLM을 통해 제약조건 추출)
    candidates: Annotated[List[Hit], add] # RAG Top-k 후보
    topk: int # 후보 개수 설정 (Planner가 참고)
    selection_required: bool # 후보 중에서 하나 선택하라는 흐름 제어 플래그
    selected_id: Optional[str] # 사용자가 고른 레시피 -> 나머지 Agent는 이를 기반으로 동작

    # Chef / 조리 진행
    recipe_text: Optional[str]
    steps: List[str]
    step_idx: int
    action: Optional[str]   # "choose:<id>" | "next_step" | "stop" | "ask:<...>"

    # Nutrition
    nutrition: Optional[Nutrition] # 영양 분석 결과 

    # Memory
    prefs: Prefs
    memory_event: Optional[Dict[str, Any]] # 이번 세션에서 기록한 이벤츠 

    # MAS용 메타 정보
    next_intent: Optional[NextIntent]        # 다음에 어디로 갈지에 대한 의도
    last_agent: Optional[str]                # 마지막으로 실행된 에이전트 이름
    errors: Annotated[List[str], add]        # 에러 메시지 누적 (옵션)
