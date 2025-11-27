# main.py - 대화형 요리 추천 시스템 (Planner + Choose + Chef + Nutrition)

import sys
from pathlib import Path
import uuid

print("=" * 60)
print("🔍 시스템 시작")
print("=" * 60)

try:
    print("\n[1/5] 기본 모듈 로딩 중...")
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.sqlite import SqliteSaver
    print("✅ LangGraph 완료")
    
    print("\n[2/5] 상태 모듈 로딩 중...")
    from state import State
    print("✅ State 완료")
    
    print("\n[3/5] 노드 모듈 로딩 중 (FAISS/모델 로딩 - 1~2분 소요)...")
    # 🔽 기존: planner_agent, choose_agent 만 import
    # 🔽 추가: chef_agent, nutrition_agent, memory_agent 도 함께 import
    from nodes import (
        planner_agent,
        choose_agent,
        chef_agent,
        nutrition_agent,
        memory_agent,
    )
    print("✅ Nodes 완료")
    
except Exception as e:
    print(f"\n❌ Import 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def build_planner_graph() -> StateGraph:
    """Planner만 실행하는 그래프"""
    g = StateGraph(State)
    g.add_node("planner", planner_agent)
    g.set_entry_point("planner")
    g.add_edge("planner", END)
    return g


def build_choose_graph() -> StateGraph:
    """Choose만 실행하는 그래프"""
    g = StateGraph(State)
    g.add_node("choose", choose_agent)
    g.set_entry_point("choose")
    g.add_edge("choose", END)
    return g


def main():
    try:
        print("\n[4/5] 환경 설정 중...")
        ROOT = Path(__file__).resolve().parent
        (ROOT / "runs").mkdir(exist_ok=True)
        (ROOT / "data").mkdir(exist_ok=True)
        
        db_path = ROOT / "runs" / "cooking.db"
        
        print("\n[5/5] 앱 컴파일 중...")
        with SqliteSaver.from_conn_string(str(db_path)) as cp:
            # 두 개의 독립적인 앱 생성 (Planner / Choose)
            planner_app = build_planner_graph().compile(checkpointer=cp)
            choose_app = build_choose_graph().compile(checkpointer=cp)
            
            cfg = {"configurable": {"thread_id": f"user-session-{uuid.uuid4()}"}}
            
            print("\n" + "=" * 60)
            print("✅ 시스템 준비 완료!")
            print("=" * 60)
            print("🍳 요리 추천 시스템")
            print("=" * 60)
            print("종료: 'quit' 또는 'exit'\n")
            
            # === 1단계: 요리 추천 (Planner) ===
            state = None
            while True:
                try:
                    user_input = input("👤 You (요리 요청): ").strip()
                    
                    if user_input.lower() in ["quit", "exit", "종료"]:
                        print("\n👋 종료합니다.")
                        return
                    
                    if not user_input:
                        continue
                    
                    print("\n⏳ 요리 검색 중...\n")
                    
                    # Planner만 실행
                    state = planner_app.invoke(
                        {
                            "messages": [{"role": "user", "content": user_input}],
                            "topk": 5,
                            "prefs": {},
                        },
                        cfg,
                    )
                    
                    # ✅ 추천 결과 출력
                    if state.get("messages"):
                        assistant_texts = []

                        for msg in state["messages"]:
                            # case 1) dict 형태 {"role": "assistant", "content": "..."}
                            if isinstance(msg, dict) and msg.get("role") == "assistant":
                                assistant_texts.append(msg.get("content", ""))

                            # case 2) LangChain AIMessage / ChatMessage 객체
                            elif hasattr(msg, "type") and msg.type in ("ai", "assistant"):
                                assistant_texts.append(getattr(msg, "content", ""))

                        if assistant_texts:
                            last_text = assistant_texts[-1]   # 마지막 assistant 메시지 = 추천 목록
                            print(f"🤖 Assistant:\n{last_text}\n")
                    
                    # 후보가 있으면 선택 단계로
                    if state.get("candidates"):
                        break
                    else:
                        print("❌ 검색 결과가 없습니다. 다시 시도해주세요.\n")
                        
                except KeyboardInterrupt:
                    print("\n\n종료하려면 'quit'를 입력하세요.")
                    continue
                except Exception as e:
                    print(f"\n❌ 오류: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # === 2단계: 레시피 선택 (Choose) ===
            while True:
                try:
                    user_input = input("👤 You (선택): ").strip()
                    
                    if user_input.lower() in ["quit", "exit", "종료"]:
                        print("\n👋 종료합니다.")
                        return
                    
                    if not user_input:
                        continue
                    
                    # 숫자만 입력한 경우 (1~5)
                    if user_input.isdigit():
                        idx = int(user_input) - 1
                        if 0 <= idx < len(state.get("candidates", [])):
                            chosen = state["candidates"][idx]
                            recipe_id = chosen["id"]
                        else:
                            print(f"❌ 1~{len(state['candidates'])} 사이의 숫자를 입력하세요.\n")
                            continue
                    # r0, r1 형태로 입력한 경우
                    elif user_input.lower().startswith("r"):
                        recipe_id = user_input.lower()
                    else:
                        print("❌ 올바른 형식으로 입력하세요 (예: 2 또는 r13)\n")
                        continue
                    
                    print(f"\n⏳ '{recipe_id}' 레시피 확인 중...\n")
                    
                    # 기존 state에 action 추가해서 Choose 실행
                    state["action"] = f"choose:{recipe_id}"
                    result = choose_app.invoke(state, cfg)
                    
                    # 선택 결과 확인
                    if result.get("selected_id"):
                        if result.get("messages"):
                            last_msg = result["messages"][-1]
                            if isinstance(last_msg, dict):
                                print(f"🤖 Assistant:\n{last_msg['content']}\n")
                            elif hasattr(last_msg, "content"):
                                print(f"🤖 Assistant:\n{last_msg.content}\n")
                        
                        # === 3단계: Chef + Nutrition 단계로 바로 진입 ===
                        state = result  # 선택 결과를 이후 단계의 초기 state로 사용

                        print("\n" + "=" * 60)
                        print(f"✅ 선택된 레시피 ID: {state['selected_id']}")
                        print("=" * 60)
                        print("이제 단계별 조리 안내를 시작합니다.")
                        print("• 다음 단계: 'next_step' 또는 'action=next_step'")
                        print("• 질문하기: 'ask:마늘 대신 양파 써도 돼?' 또는 'action=ask:...'\n")
                        print("조리를 모두 마쳤다면 'stop' 이라고 입력하면 영양 분석으로 이동합니다.")
                        print("=" * 60 + "\n")

                        # === 3-1) Chef 루프 ===
                        while True:
                            try:
                                cook_input = input("👤 You (조리/질문): ").strip()

                                if cook_input.lower() in ["quit", "exit", "종료"]:
                                    print("\n👋 종료합니다.")
                                    return
                                if not cook_input:
                                    continue

                                # 'action=' 접두사가 있어도 처리 가능
                                if cook_input.lower().startswith("action="):
                                    action = cook_input.split("=", 1)[1].strip()
                                else:
                                    action = cook_input.strip()

                                # 사용자가 그냥 'stop' 입력하면 → 조리 종료로 간주
                                if action.lower() == "stop":
                                    # chef_agent에서 조리 종료 브랜치 태우기 위해
                                    action = "stop"

                                state["action"] = action

                                chef_result = chef_agent(state)
                                # 기존 state에 업데이트
                                state.update(chef_result)

                                # 메시지 출력
                                msgs = chef_result.get("messages", [])
                                if msgs:
                                    last_msg = msgs[-1]
                                    if isinstance(last_msg, dict):
                                        print(f"🤖 Assistant:\n{last_msg['content']}\n")
                                    elif hasattr(last_msg, "content"):
                                        print(f"🤖 Assistant:\n{last_msg.content}\n")

                                # 다음 의도에 따라 분기
                                intent = chef_result.get("next_intent")
                                if intent == "cook_next":
                                    # 다음 step 계속
                                    continue
                                elif intent == "analyze_nutrition":
                                    # 조리 완료 → 영양 분석 단계로 이동
                                    break
                                else:
                                    # 예외적인 경우에도 일단 루프 종료 후 영양 분석으로 보냄
                                    break

                            except KeyboardInterrupt:
                                print("\n\n종료하려면 'quit'를 입력하세요.")
                                continue
                            except Exception as e:
                                print(f"\n❌ Chef 단계 오류: {e}")
                                import traceback
                                traceback.print_exc()
                                continue

                        # === 3-2) Nutrition 단계 ===
                        print("\n⏳ 영양 정보를 계산하는 중...\n")
                        nut_result = nutrition_agent(state)
                        state.update(nut_result)

                        nut_msgs = nut_result.get("messages", [])
                        if nut_msgs:
                            last_msg = nut_msgs[-1]
                            if isinstance(last_msg, dict):
                                print(f"🤖 Assistant:\n{last_msg['content']}\n")
                            elif hasattr(last_msg, "content"):
                                print(f"🤖 Assistant:\n{last_msg.content}\n")

                        # === 3-3) Memory 저장 단계 ===
                        mem_result = memory_agent(state)
                        state.update(mem_result)

                        mem_msgs = mem_result.get("messages", [])
                        if mem_msgs:
                            last_msg = mem_msgs[-1]
                            if isinstance(last_msg, dict):
                                print(f"🤖 Assistant:\n{last_msg['content']}\n")
                            elif hasattr(last_msg, "content"):
                                print(f"🤖 Assistant:\n{last_msg.content}\n")

                        print("\n🎉 전체 플로우(추천 → 선택 → 조리 → 영양분석 → 기록)가 완료되었습니다.")
                        return

                    else:
                        # 선택 실패 - 재시도
                        if result.get("messages"):
                            last_msg = result["messages"][-1]
                            if isinstance(last_msg, dict):
                                print(f"🤖 Assistant:\n{last_msg['content']}\n")
                            elif hasattr(last_msg, "content"):
                                print(f"🤖 Assistant:\n{last_msg.content}\n")
                        
                except KeyboardInterrupt:
                    print("\n\n종료하려면 'quit'를 입력하세요.")
                    continue
                except Exception as e:
                    print(f"\n❌ 오류: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                    
    except Exception as e:
        print(f"\n❌ 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
