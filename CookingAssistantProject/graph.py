# graph.py
from pathlib import Path
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from state import State
from nodes import (
    planner_agent,
    choose_agent,
    chef_agent,
    nutrition_agent,
    memory_agent,
)

# ========== Supervisor ==========

def supervisor_node(state: State) -> State:
    # 상태를 변경하지 않고, route_supervisor가 읽고 결정할 수 있게만 둠
    return {}

def route_supervisor(state: State) -> str:
    """
    next_intent / 현재 state를 바탕으로
    다음에 실행할 노드 이름을 문자열로 반환.
    add_conditional_edges의 key와 매칭되어야 함.
    """
    intent = state.get("next_intent")
    act = (state.get("action") or "").lower()

    # 1) 명시적 의도 기반 라우팅
    if intent == "need_selection":
        return "choose"
    if intent in ("cook_next", "user_question"):
        return "chef"
    if intent == "analyze_nutrition":
        return "nutrition"
    if intent == "write_memory":
        return "memory"
    if intent == "finished":
        return "END"  # END로 매핑 예정

    # 2) 안전한 기본값들 (초기 진입/의도 누락 케이스)
    if state.get("selection_required", False):
        return "choose"
    if not state.get("selected_id"):
        return "planner"
    if act.startswith("ask:") or act == "next_step":
        return "chef"

    # fallback: 다시 planner
    return "planner"


def build_builder() -> StateGraph:
    g = StateGraph(State)

    # 에이전트 노드
    g.add_node("supervisor", supervisor_node)
    g.add_node("planner", planner_agent)
    g.add_node("choose", choose_agent)
    g.add_node("chef", chef_agent)
    g.add_node("nutrition", nutrition_agent)
    g.add_node("memory", memory_agent)

    # 시작은 Supervisor
    g.set_entry_point("supervisor")

    # Supervisor -> (다음 노드) : 조건부 엣지
    g.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "planner": "planner",
            "choose": "choose",
            "chef": "chef",
            "nutrition": "nutrition",
            "memory": "memory",
            "END": END,           # intent == finished 일 때
        },
    )

    # 각 에이전트는 작업 후 Supervisor로 복귀
    g.add_edge("planner", "supervisor")
    g.add_edge("choose", "supervisor")
    g.add_edge("chef", "supervisor")

    # Nutrition -> Memory -> Supervisor (기록 후 끝/다음 판단)
    g.add_edge("nutrition", "memory")
    g.add_edge("memory", "supervisor")

    return g


# ========== 실행 예시 (동일 thread_id로 연속 호출) ==========

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "user_memory.jsonl").touch(exist_ok=True)

    builder = build_builder()
    db_path = (ROOT / "runs" / "cooking.db")
    with SqliteSaver.from_conn_string(str(db_path)) as cp:
        app = builder.compile(checkpointer=cp)

        cfg = {"configurable": {"thread_id": "demo-user"}}

        # 1) 유저 최초 요청 → supervisor → planner → supervisor → choose
        s = app.invoke({
            "messages":[{"role":"user","content":"닭가슴살로 15분 안에 만들 수 있는 요리 추천해줘"}],
            "topk": 3,
            "prefs": {"disliked":["너무 매운 것"]},
            "next_intent": None,
        }, cfg)
        print(s["messages"][-1]["content"])

        # 2) 유저가 r2 선택 → supervisor → choose → supervisor → chef 준비
        s = app.invoke({"action":"choose:r2"}, cfg)
        print(s["messages"][-1]["content"])

        # 3) 스텝 진행 + 중간 질문들 → 매번 supervisor가 적절히 chef로 라우팅
        s = app.invoke({"action":"next_step"}, cfg); print(s["messages"][-1]["content"])
        s = app.invoke({"action":"ask:마늘 대신 양파를 써도 될까?"}, cfg); print(s["messages"][-1]["content"])
        s = app.invoke({"action":"next_step"}, cfg); print(s["messages"][-1]["content"])
        s = app.invoke({"action":"next_step"}, cfg); print(s["messages"][-1]["content"])

        # 4) stop or 더 이상 step 없음 → nutrition → memory → supervisor → END
        s = app.invoke({"action":"stop"}, cfg)
        print(s["messages"][-1]["content"])
