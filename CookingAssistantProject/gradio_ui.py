# cooking_gradio_dynamic.py - 역동적인 요리 추천 시스템 UI

import gradio as gr
import sys
from pathlib import Path
import uuid
import json
from typing import Dict, List, Any, Optional, Tuple

# 백엔드 모듈 임포트
try:
    import os
    
    # .env 파일에서 API 키 로드 시도
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv가 설치되지 않은 경우 무시
    
    # API 키가 설정되지 않은 경우 사용자에게 안내
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OpenAI API 키가 설정되지 않았습니다.")
        print("다음 중 하나의 방법으로 설정해주세요:")
        print("1. 환경 변수: export OPENAI_API_KEY='your-key'")
        print("2. .env 파일에: OPENAI_API_KEY=your-key")
        print("3. 아래 코드에서 직접 설정:")
        
        # 임시로 더미 키 설정 (실제로는 동작하지 않음)
        os.environ["OPENAI_API_KEY"] = ""
        print("4. 현재는 더미 키로 설정되어 실제 동작하지 않을 수 있습니다.")
    
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
except ImportError as e:
    print(f"백엔드 모듈을 찾을 수 없습니다: {e}")
    sys.exit(1)

# 전역 변수
ROOT = Path(__file__).resolve().parent
(ROOT / "runs").mkdir(exist_ok=True)
(ROOT / "data").mkdir(exist_ok=True)

current_state = {}
current_step = "welcome"  # welcome, search, select, cook, nutrition, complete

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

def get_recipe_recommendations(user_input: str, topk: int = 5) -> Tuple[str, str, Dict]:
    """레시피 추천받기"""
    global current_state, current_step
    
    if not user_input.strip():
        return "재료나 요리 요청을 입력해주세요.", "[]", gr.update()
    
    try:
        db_path = ROOT / "runs" / "cooking.db"
        
        with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
            planner_graph = build_planner_graph()
            planner_app = planner_graph.compile(checkpointer=checkpointer)
            cfg = {"configurable": {"thread_id": f"gradio-session-{uuid.uuid4()}"}}
            
            state = planner_app.invoke(
                {
                    "messages": [{"role": "user", "content": user_input}],
                    "topk": topk,
                    "prefs": {},
                },
                cfg,
            )
        
        current_state = state
        current_step = "select"
        
        # 추천 결과 생성
        result = "🔍 **검색 완료! 맞춤 레시피를 찾았습니다**\n\n"
        
        if state.get("candidates"):
            for i, candidate in enumerate(state["candidates"], 1):
                score = int(candidate["score"] * 100)
                result += f"**{i}. {candidate['title']}** (ID: {candidate['id']}, 매칭도: {score}%)\n"
                result += f"   💡 {candidate['text'][:80]}...\n\n"
            
            # 후보 정보를 JSON으로 저장
            candidates_json = json.dumps([
                {"num": i, "id": c["id"], "title": c["title"], "score": c["score"]}
                for i, c in enumerate(state["candidates"], 1)
            ], ensure_ascii=False, indent=2)
            
            # 선택 섹션을 표시하도록 업데이트
            return result, candidates_json, gr.update(visible=True)
        else:
            return "검색 결과가 없습니다.", "[]", gr.update()
            
    except Exception as e:
        return f"오류 발생: {str(e)}", "[]", gr.update()

def select_recipe(selection: str) -> Tuple[str, Dict, Dict]:
    """레시피 선택하기"""
    global current_state, current_step
    
    if not selection.strip():
        return "레시피 번호나 ID를 입력해주세요.", gr.update(), gr.update()
    
    try:
        # 숫자 입력 처리
        if selection.isdigit():
            idx = int(selection) - 1
            if 0 <= idx < len(current_state.get("candidates", [])):
                chosen = current_state["candidates"][idx]
                recipe_id = chosen["id"]
            else:
                return f"1~{len(current_state['candidates'])} 사이의 숫자를 입력하세요.", gr.update(), gr.update()
        # ID 직접 입력 처리
        else:
            recipe_id = selection.lower()
        
        db_path = ROOT / "runs" / "cooking.db"
        
        with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
            choose_graph = build_choose_graph()
            choose_app = choose_graph.compile(checkpointer=checkpointer)
            cfg = {"configurable": {"thread_id": f"gradio-session-{uuid.uuid4()}"}}
            
            current_state["action"] = f"choose:{recipe_id}"
            result = choose_app.invoke(current_state, cfg)
            current_state = result
        
        if result.get("selected_id"):
            current_step = "cook"
            
            # 메시지 추출
            if result.get("messages"):
                last_msg = result["messages"][-1]
                if isinstance(last_msg, dict):
                    content = last_msg.get("content", "")
                elif hasattr(last_msg, "content"):
                    content = last_msg.content
                else:
                    content = "레시피를 선택했습니다."
                
                formatted_content = f"""
<div style="margin: 20px 0; padding: 20px; background: linear-gradient(135deg, #d4edda, #c3e6cb); border-radius: 12px; border-left: 5px solid #28a745;">
    <h3 style="margin: 0 0 10px 0; color: #155724;">🎯 {content}</h3>
</div>
"""
                
                # 조리 섹션을 표시하고 선택 섹션을 숨김
                return formatted_content, gr.update(visible=False), gr.update(visible=True)
            
            return f"✅ 레시피 '{recipe_id}'가 선택되었습니다.", gr.update(visible=False), gr.update(visible=True)
        else:
            return "레시피 선택에 실패했습니다.", gr.update(), gr.update()
            
    except Exception as e:
        return f"오류 발생: {str(e)}", gr.update(), gr.update()

def cooking_step(action: str, question: str = "") -> Tuple[str, Dict]:
    """조리 단계 진행"""
    global current_state, current_step
    
    if not current_state.get("selected_id"):
        return "먼저 레시피를 선택해주세요.", gr.update()
    
    try:
        if action == "next_step":
            current_state["action"] = "next_step"
        elif action == "prev_step":
            current_state["action"] = "prev_step"
        elif action == "stop":
            current_state["action"] = "stop"
        elif action == "ask" and question:
            current_state["action"] = f"ask:{question}"
        else:
            return "올바르지 않은 액션입니다.", gr.update()
        
        # Chef agent 실행
        chef_result = chef_agent(current_state)
        current_state.update(chef_result)
        
        # 메시지 추출
        if chef_result.get("messages"):
            last_msg = chef_result["messages"][-1]
            if isinstance(last_msg, dict):
                content = last_msg.get("content", "")
            elif hasattr(last_msg, "content"):
                content = last_msg.content
            else:
                content = "진행 중..."
            
            # 질문-답변 형식인지 확인하고 HTML로 포맷팅
            if action == "ask" and question and content.startswith("Q: "):
                # Q: 질문\nA: 답변 형식을 HTML로 변환
                parts = content.split("\nA: ", 1)
                if len(parts) == 2:
                    q_part = parts[0].replace("Q: ", "").strip()
                    a_part = parts[1].strip()
                    
                    content = f"""
<div style="margin: 20px 0; padding: 20px; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 12px; border-left: 5px solid #007bff;">
    <div style="margin: 0 0 15px 0;">
        <span style="color: #dc3545; font-weight: bold; font-size: 18px;">Q:</span> 
        <span style="font-size: 16px; margin-left: 10px;">{q_part}</span>
    </div>
    <div style="margin: 0;">
        <span style="color: #0066cc; font-weight: bold; font-size: 18px;">A:</span> 
        <span style="font-size: 16px; line-height: 1.6; margin-left: 10px;">{a_part}</span>
    </div>
</div>
"""
            elif content.startswith("[Step"):
                # 조리 단계 메시지도 HTML로 포맷팅
                content = f"""
<div style="margin: 20px 0; padding: 20px; background: linear-gradient(135deg, #fff3cd, #ffeaa7); border-radius: 12px; border-left: 5px solid #ffc107; animation: slideIn 0.5s ease-in;">
    <div style="font-size: 20px; line-height: 1.6; color: #856404; font-weight: 500;">{content}</div>
</div>
<style>
@keyframes slideIn {{
    from {{ opacity: 0; transform: translateY(-10px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
</style>
"""
            else:
                # 기타 메시지들도 HTML로 포맷팅
                content = f"""
<div style="margin: 20px 0; padding: 20px; background: linear-gradient(135deg, #d1ecf1, #bee5eb); border-radius: 12px; border-left: 5px solid #17a2b8;">
    <div style="font-size: 16px; line-height: 1.6; color: #0c5460;">{content}</div>
</div>
"""
            
            # 다음 의도 확인
            intent = chef_result.get("next_intent")
            if intent == "analyze_nutrition":
                current_step = "nutrition"
                return content, gr.update(visible=True)  # 영양 분석 버튼 표시
            else:
                return content, gr.update()
        else:
            return "조리 단계가 진행되었습니다.", gr.update()
            
    except Exception as e:
        return f"오류 발생: {str(e)}", gr.update()

def get_nutrition_info() -> Tuple[str, Dict]:
    """영양 정보 분석"""
    global current_state, current_step
    
    if not current_state.get("selected_id"):
        return "먼저 레시피를 선택하고 조리를 완료해주세요.", gr.update()
    
    try:
        # Nutrition agent 실행
        nut_result = nutrition_agent(current_state)
        current_state.update(nut_result)
        
        # Memory agent 실행 (자동으로 기록)
        mem_result = memory_agent(current_state)
        current_state.update(mem_result)
        
        current_step = "complete"
        
        # 결과 생성
        result = """
<div style="margin: 20px 0; padding: 25px; background: linear-gradient(135deg, #e1f5fe, #b3e5fc); border-radius: 15px; border-left: 5px solid #0288d1;">
    <h2 style="margin: 0 0 20px 0; color: #01579b;">📊 영양 정보 분석 완료</h2>
"""
        
        # 영양 정보를 직접 구성
        nutrition_data = current_state.get("nutrition", {})
        if nutrition_data:
            # 직접 영양 정보 HTML 구성
            content = f""" 영양 정보 

• 칼로리: {nutrition_data.get('calories', 0)}kcal
• 단백질: {nutrition_data.get('protein_g', 0)}g
• 지방: {nutrition_data.get('fat_g', 0)}g
• 탄수화물: {nutrition_data.get('carbs_g', 0)}g

💡 **건강 팁:** {nutrition_data.get('note', '영양 정보를 확인하세요.')}"""
            
            result += f"<div style='font-size: 16px; line-height: 1.6; margin-bottom: 15px; white-space: pre-line;'>{content}</div>"
        else:
            # nutrition_agent에서 온 메시지 사용 (fallback)
            if nut_result.get("messages"):
                last_msg = nut_result["messages"][-1]
                if isinstance(last_msg, dict):
                    content = last_msg.get("content", "")
                elif hasattr(last_msg, "content"):
                    content = last_msg.content
                else:
                    content = ""
                
                # 괄호 제거 및 줄바꿈 처리
                if content:
                    # 정규식을 사용하지 않고 간단하게 처리
                    if content.strip().endswith(")"):
                        # 마지막 여는 괄호 찾기
                        open_paren = content.rfind("(")
                        if open_paren != -1:
                            main_part = content[:open_paren].strip()
                            note_part = content[open_paren+1:-1].strip()
                            
                            if note_part:  # note가 비어있지 않은 경우에만
                                content = f"{main_part}\n\n💡 건강 팁 : {note_part}"
                            else:
                                content = main_part
                
                result += f"<div style='font-size: 17px; line-height: 1.6; margin-bottom: 15px; white-space: pre-line;'>{content}</div>"
        
        # 기록 메시지
        if mem_result.get("messages"):
            last_msg = mem_result["messages"][-1]
            if isinstance(last_msg, dict):
                result += f"<div style='font-size: 14px; color: #0277bd; font-style: italic;'>{last_msg.get('content', '')}</div>"
            elif hasattr(last_msg, "content"):
                result += f"<div style='font-size: 14px; color: #0277bd; font-style: italic;'>{last_msg.content}</div>"
        
        result += "</div>"
        
        # 완료 섹션을 표시
        return result, gr.update(visible=True)
        
    except Exception as e:
        return f"오류 발생: {str(e)}", gr.update()

def reset_session():
    """세션 초기화"""
    global current_state, current_step
    current_state = {}
    current_step = "welcome"
    return (
        "",  # 추천 결과
        "[]",  # 후보 JSON
        "",  # 선택 결과
        "<div style='font-size: 18px; line-height: 1.6; padding: 20px; text-align: center; color: #6c757d;'>🍳 새로운 요리 여행을 시작해보세요!</div>",  # 조리 결과 (HTML)
        "",   # 영양 정보
        gr.update(visible=False),  # 선택 섹션 숨김
        gr.update(visible=False),  # 조리 섹션 숨김
        gr.update(visible=False),  # 영양 버튼 숨김
        gr.update(visible=False),  # 완료 섹션 숨김
    )

def create_interface():
    """Gradio 인터페이스 생성"""
    
    with gr.Blocks(
        title="🍳 AI 요리 추천 시스템",
        theme=gr.themes.Soft()
    ) as interface:
        
        gr.HTML("""
        <div style="text-align: center; margin: 20px 0; padding: 30px; background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 15px; color: white;">
            <h1 style="margin: 0; font-size: 2.5em;">🍳 AI Cooking Assistant</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9;">당신만을 위한 맞춤 레시피를 찾아드립니다</p>
        </div>
        """)
        
        # 1단계: 레시피 추천 (항상 표시)
        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div style="margin: 20px 0; padding: 20px; background: linear-gradient(135deg, #ffecd2, #fcb69f); border-radius: 12px;">
                    <h2 style="margin: 0 0 10px 0; color: #8b4513;">🔍 STEP 1: 원하는 요리 알려주세요</h2>
                </div>
                """)
                
                with gr.Row():
                    user_input = gr.Textbox(
                        label="재료 또는 요리 요청",
                        placeholder="예: 닭가슴살로 15분 안에 만들 수 있는 요리",
                        lines=2,
                        scale=4
                    )
                    topk = gr.Number(
                        label="추천 개수",
                        minimum=1,
                        maximum=10,
                        value=5,
                        precision=0,
                        scale=1
                    )
                
                recommend_btn = gr.Button("🔍 맞춤 레시피 찾기", variant="primary", size="lg")
        
        recommendations_output = gr.Markdown(label="추천 결과")
        candidates_json = gr.JSON(label="후보 정보 (참고용)", visible=False)
        
        # 2단계: 레시피 선택 (추천 후 표시)
        with gr.Column(visible=False) as selection_section:
            gr.HTML("""
            <div style="margin: 30px 0 20px 0; padding: 20px; background: linear-gradient(135deg, #a8edea, #fed6e3); border-radius: 12px;">
                <h2 style="margin: 0 0 10px 0; color: #2c3e50;">🎯 STEP 2: 마음에 드는 레시피를 선택하세요</h2>
            </div>
            """)
            
            with gr.Row():
                selection_input = gr.Textbox(
                    label="선택할 레시피",
                    placeholder="번호 (예: 2) 또는 ID (예: r13) 입력",
                    lines=1,
                    scale=4
                )
                select_btn = gr.Button("✅ 이 레시피로 요리하기!", variant="primary", scale=2)
        
        selection_output = gr.HTML(label="선택 결과")
        
        # 3단계: 조리 진행 (선택 후 표시)
        with gr.Column(visible=False) as cooking_section:
            gr.HTML("""
            <div style="margin: 30px 0 20px 0; padding: 20px; background: linear-gradient(135deg, #ffd89b, #19547b); border-radius: 12px;">
                <h2 style="margin: 0 0 10px 0; color: white;">👨‍🍳 STEP 3: 함께 요리해봐요!</h2>
            </div>
            """)
            
            # 조리과정 출력 (가장 위에 배치)
            cooking_output = gr.HTML(
                label="조리 과정", 
                value="<div style='font-size: 18px; line-height: 1.6; padding: 20px; text-align: center; color: #6c757d;'>레시피를 선택하면 단계별 조리 안내를 시작합니다.</div>"
            )
            
            # 질문하기 섹션
            gr.HTML("""
            <div style="margin: 25px 0 15px 0; padding: 15px; background: linear-gradient(135deg, #e0f2f1, #b2dfdb); border-radius: 10px;">
                <h3 style="margin: 0; color: #00695c;">❓ 궁금한 점이 있으신가요?</h3>
            </div>
            """)
            with gr.Row():
                question_input = gr.Textbox(
                    label="💬 AI 셰프에게 질문하기",
                    placeholder="예: 마늘 대신 양파를 써도 될까요?",
                    scale=4
                )
                ask_btn = gr.Button("🤔 질문하기", variant="secondary", scale=1)
            
            # 조리 컨트롤 버튼들
            with gr.Row():
                prev_step_btn = gr.Button("◀️ 이전 단계", variant="secondary")
                next_step_btn = gr.Button("▶️ 다음 단계", variant="primary")
                finish_btn = gr.Button("🏁 조리 완료", variant="secondary")
        
        # 영양 분석 버튼 (조리 완료 후 표시)
        with gr.Row():
            nutrition_btn = gr.Button("📊 영양 분석하기", variant="primary", size="lg", visible=False)
        
        nutrition_output = gr.HTML(label="영양 정보")
        
        # 완료 섹션 (영양 분석 후 표시)
        with gr.Column(visible=False) as complete_section:
            gr.HTML("""
            <div style="margin: 30px 0; padding: 30px; background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 15px; text-align: center; color: white;">
                <h2 style="margin: 0 0 15px 0;">🎉 요리 완성!</h2>
                <p style="margin: 0; font-size: 1.1em; opacity: 0.9;">맛있게 드세요! 새로운 요리에 도전해보세요.</p>
            </div>
            """)
            reset_btn = gr.Button("🔄 새로운 요리 시작", variant="primary", size="lg")
        
        # 이벤트 핸들러
        recommend_btn.click(
            fn=get_recipe_recommendations,
            inputs=[user_input, topk],
            outputs=[recommendations_output, candidates_json, selection_section]
        )
        
        select_btn.click(
            fn=select_recipe,
            inputs=[selection_input],
            outputs=[selection_output, selection_section, cooking_section]
        )
        
        next_step_btn.click(
            fn=lambda: cooking_step("next_step"),
            outputs=[cooking_output, nutrition_btn]
        )
        
        prev_step_btn.click(
            fn=lambda: cooking_step("prev_step"),
            outputs=[cooking_output, nutrition_btn]
        )
        
        finish_btn.click(
            fn=lambda: cooking_step("stop"),
            outputs=[cooking_output, nutrition_btn]
        )
        
        ask_btn.click(
            fn=lambda q: cooking_step("ask", q),
            inputs=[question_input],
            outputs=[cooking_output, nutrition_btn]
        ).then(
            # 질문 입력창 초기화
            fn=lambda: "",
            outputs=[question_input]
        )
        
        nutrition_btn.click(
            fn=get_nutrition_info,
            outputs=[nutrition_output, complete_section]
        )
        
        reset_btn.click(
            fn=reset_session,
            outputs=[
                recommendations_output,
                candidates_json,
                selection_output,
                cooking_output,
                nutrition_output,
                selection_section,
                cooking_section,
                nutrition_btn,
                complete_section
            ]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    
    print("🍳 AI 요리 추천 시스템 시작")
    print("Gradio 인터페이스가 시작됩니다...")
    print("브라우저에서 http://localhost:7860 으로 접속하세요")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
