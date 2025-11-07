
"""
LangGraph 기반 요리 AI 어시스턴트
4개의 에이전트가 협력하여 개인 맞춤형 요리 서비스를 제공
"""

import json
import os
import sqlite3
from typing import Dict, List, Optional, TypedDict, Annotated
from datetime import datetime
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

import os
os.environ["OPENAI_API_KEY"] = ""
class RecipeState(TypedDict):
    """요리 어시스턴트의 상태 정의"""
    messages: Annotated[list, add_messages]
    user_query: str
    user_constraints: Dict  # 재료, 시간, 도구, 알레르기 등 제약조건
    recipe_candidates: List[Dict]
    selected_recipe: Optional[Dict]
    current_step: int
    cooking_complete: bool
    nutrition_analysis: Optional[Dict]
    user_feedback: Optional[str]
    next_action: str


class RecipeDatabase:
    """FAISS 기반 레시피 검색 엔진"""
    
    def __init__(self, faiss_path: str, jsonl_path: str):
        # FAISS 인덱스 로드
        self.index = faiss.read_index(faiss_path)
        
        # 레시피 데이터 로드
        self.recipes = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.recipes.append(json.loads(line))
        
        # Sentence Transformer 모델 로드
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def search_recipes(self, query: str, k: int = 5) -> List[Dict]:
        """의미 기반 레시피 검색"""
        query_vector = self.encoder.encode([query])
        distances, indices = self.index.search(query_vector.astype('float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.recipes):
                recipe = self.recipes[idx].copy()
                recipe['similarity_score'] = float(1 - distances[0][i])  # 유사도 점수
                results.append(recipe)
        
        return results
    
    def filter_by_constraints(self, recipes: List[Dict], constraints: Dict) -> List[Dict]:
        """제약조건에 따른 레시피 필터링"""
        filtered = []
        
        for recipe in recipes:
            # 재료 기반 필터링
            if 'available_ingredients' in constraints and constraints['available_ingredients']:
                recipe_ingredients = set(recipe.get('NER', []))
                available_ingredients = set(constraints['available_ingredients'])
                
                # 사용 가능한 재료로 만들 수 있는지 확인 (일부 대체 가능)
                if recipe_ingredients:
                    ingredient_match_ratio = len(recipe_ingredients.intersection(available_ingredients)) / len(recipe_ingredients)
                    if ingredient_match_ratio < 0.3:  # 30% 이상 재료 매칭 필요 (좀 더 관대하게)
                        continue
            
            # 시간 제약
            if 'max_time' in constraints and constraints['max_time']:
                recipe_time = recipe.get('total_time_min_est', 0)
                if recipe_time > constraints['max_time']:
                    continue
            
            # 알레르기 제약
            if 'allergies' in constraints and constraints['allergies']:
                recipe_ingredients = ' '.join(recipe.get('NER', [])).lower()
                if any(allergy.lower() in recipe_ingredients for allergy in constraints['allergies']):
                    continue
            
            filtered.append(recipe)
        
        return filtered


class UserMemory:
    """사용자 정보 및 히스토리 관리"""
    
    def __init__(self, db_path: str = "user_memory.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """사용자 데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 사용자 프로필 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profile (
                user_id TEXT PRIMARY KEY,
                preferences TEXT,  -- JSON 형태로 저장
                allergies TEXT,    -- JSON 형태로 저장
                dietary_goals TEXT -- JSON 형태로 저장
            )
        ''')
        
        # 요리 히스토리 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cooking_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                recipe_title TEXT,
                ingredients_used TEXT,
                nutrition_info TEXT,
                feedback TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_user_profile(self, user_id: str = "default") -> Dict:
        """사용자 프로필 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM user_profile WHERE user_id = ?", (user_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'preferences': json.loads(result[1] or '[]'),
                'allergies': json.loads(result[2] or '[]'),
                'dietary_goals': json.loads(result[3] or '{}')
            }
        else:
            # 기본 프로필 생성
            default_profile = {
                'preferences': [],
                'allergies': [],
                'dietary_goals': {}
            }
            self.save_user_profile(user_id, default_profile)
            return default_profile
    
    def save_user_profile(self, user_id: str, profile: Dict):
        """사용자 프로필 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_profile (user_id, preferences, allergies, dietary_goals)
            VALUES (?, ?, ?, ?)
        ''', (
            user_id,
            json.dumps(profile.get('preferences', [])),
            json.dumps(profile.get('allergies', [])),
            json.dumps(profile.get('dietary_goals', {}))
        ))
        
        conn.commit()
        conn.close()
    
    def add_cooking_record(self, user_id: str, recipe_title: str, ingredients_used: List, 
                          nutrition_info: Dict, feedback: str = ""):
        """요리 기록 추가"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO cooking_history 
            (user_id, recipe_title, ingredients_used, nutrition_info, feedback)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            user_id,
            recipe_title,
            json.dumps(ingredients_used),
            json.dumps(nutrition_info),
            feedback
        ))
        
        conn.commit()
        conn.close()


class PlannerAgent:
    """요리 계획 수립 에이전트"""
    
    def __init__(self, recipe_db: RecipeDatabase, user_memory: UserMemory):
        self.recipe_db = recipe_db
        self.user_memory = user_memory
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    def process_user_input(self, user_query: str, user_id: str = "default") -> Dict:
        """사용자 입력 분석 및 구조화"""
        user_profile = self.user_memory.get_user_profile(user_id)
        
        # 시스템 메시지를 별도로 정의
        system_message = """당신은 요리 계획 전문가입니다. 사용자의 요리 요청을 분석하여 다음 정보를 JSON 형태로 추출하세요.

응답 형식은 다음과 같습니다:
- available_ingredients: 사용 가능한 재료들의 배열
- cooking_tools: 사용할 조리 도구들의 배열  
- max_time: 최대 조리 시간(분 단위)
- cuisine_type: 요리 종류
- allergies: 알레르기 정보 배열
- dietary_preferences: 식단 선호사항 배열
- search_query: 검색에 사용할 쿼리

정보가 명시되지 않은 경우 빈 배열이나 null을 사용하세요. 반드시 유효한 JSON 형식으로만 응답하세요."""

        # 사용자 메시지 구성
        user_message = f"사용자 요청: {user_query}\n사용자 프로필: {json.dumps(user_profile, ensure_ascii=False)}"
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_message),
                HumanMessage(content=user_message)
            ])
            
            # JSON 파싱 시도
            constraints = json.loads(response.content)
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"JSON 파싱 오류: {e}")
            # JSON 파싱 실패 시 기본값 사용
            constraints = {
                "available_ingredients": [],
                "cooking_tools": [],
                "max_time": None,
                "cuisine_type": None,
                "allergies": user_profile.get('allergies', []),
                "dietary_preferences": [],
                "search_query": user_query
            }
        
        return constraints
    
    def search_and_rank_recipes(self, constraints: Dict, k: int = 5) -> List[Dict]:
        """레시피 검색 및 순위 매기기"""
        search_query = constraints.get('search_query', '요리')
        
        # 1차: 의미 기반 검색
        candidates = self.recipe_db.search_recipes(search_query, k * 3)
        
        # 2차: 제약조건 필터링
        filtered_recipes = self.recipe_db.filter_by_constraints(candidates, constraints)
        
        # 결과가 없으면 원본 후보에서 가져오기
        if not filtered_recipes:
            filtered_recipes = candidates[:k]
        
        # 3차: 개인화 점수 계산 및 순위 매기기
        ranked_recipes = self.calculate_personalized_scores(filtered_recipes[:k])
        
        return ranked_recipes
    
    def calculate_personalized_scores(self, recipes: List[Dict]) -> List[Dict]:
        """개인화 점수 계산"""
        for recipe in recipes:
            base_score = recipe.get('similarity_score', 0.5)
            
            # 시간 점수 (짧을수록 높은 점수)
            time_score = 1.0
            total_time = recipe.get('total_time_min_est', 0)
            if total_time > 0:
                time_score = max(0.1, 1 - (total_time / 120))  # 2시간 기준 정규화
            
            # 재료 복잡도 점수 (적을수록 높은 점수)
            ingredients = recipe.get('ingredients', [])
            ingredient_score = max(0.1, 1 - (len(ingredients) / 20))
            
            # 최종 점수 계산
            recipe['final_score'] = (base_score * 0.5 + time_score * 0.3 + ingredient_score * 0.2)
        
        # 점수 순으로 정렬
        recipes.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        return recipes


class ChefAgent:
    """조리 과정 안내 에이전트"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    
    def start_cooking_guidance(self, recipe: Dict) -> Dict:
        """조리 안내 시작"""
        steps = recipe.get('steps_parsed', [])
        if not steps:
            # steps_parsed가 없으면 directions에서 파싱
            directions = recipe.get('directions', [])
            steps = [{"text": step, "duration_min": None, "heat": None} for step in directions]
        
        return {
            "recipe_title": recipe.get('title', ''),
            "total_steps": len(steps),
            "current_step": 0,
            "steps": steps,
            "ingredients": recipe.get('ingredients_std', recipe.get('ingredients', [])),
            "estimated_time": recipe.get('total_time_min_est', 0)
        }
    
    def get_next_step(self, cooking_session: Dict) -> Dict:
        """다음 조리 단계 안내"""
        current_step = cooking_session['current_step']
        steps = cooking_session['steps']
        
        if current_step >= len(steps):
            return {"message": "조리가 완료되었습니다!", "completed": True}
        
        step_info = steps[current_step]
        
        # 시스템 메시지
        system_message = """당신은 전문 요리사입니다. 주어진 조리 단계를 자세하고 친절하게 설명해주세요.
다음 정보를 포함하여 설명하세요:
1. 구체적인 조리 방법
2. 주의사항
3. 예상 소요 시간
4. 시각적/촉각적 완료 신호

친근하고 격려하는 톤으로 설명해주세요."""

        # 사용자 메시지
        user_message = f"조리 단계: {step_info['text']}\n예상 시간: {step_info.get('duration_min') or '미지정'}분\n화력: {step_info.get('heat') or '미지정'}"
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_message),
                HumanMessage(content=user_message)
            ])
            
            instruction = response.content
        except Exception as e:
            print(f"Chef Agent 오류: {e}")
            instruction = f"단계 {current_step + 1}: {step_info['text']}"
        
        return {
            "step_number": current_step + 1,
            "total_steps": len(steps),
            "instruction": instruction,
            "raw_step": step_info['text'],
            "duration": step_info.get('duration_min'),
            "heat": step_info.get('heat'),
            "completed": False
        }


class NutritionAgent:
    """영양 분석 에이전트"""
    
    def __init__(self, user_memory: UserMemory):
        self.user_memory = user_memory
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        
        # 기본 영양소 데이터베이스 (실제로는 더 완전한 DB 사용)
        self.nutrition_db = {
            "chicken": {"calories": 165, "protein": 31, "fat": 3.6, "carbs": 0},
            "beef": {"calories": 250, "protein": 26, "fat": 17, "carbs": 0},
            "rice": {"calories": 130, "protein": 2.7, "fat": 0.3, "carbs": 28},
            "egg": {"calories": 155, "protein": 13, "fat": 11, "carbs": 1.1},
            "butter": {"calories": 717, "protein": 0.9, "fat": 81, "carbs": 0.1},
            "milk": {"calories": 42, "protein": 3.4, "fat": 1, "carbs": 5},
        }
    
    def analyze_nutrition(self, recipe: Dict, user_id: str = "default") -> Dict:
        """레시피 영양소 분석"""
        ingredients = recipe.get('NER', [])
        
        # 기본 영양소 계산
        total_nutrition = {"calories": 0, "protein": 0, "fat": 0, "carbs": 0}
        
        for ingredient in ingredients:
            # 간단한 키워드 매칭으로 영양소 추정
            for food, nutrition in self.nutrition_db.items():
                if food.lower() in ingredient.lower():
                    for nutrient, value in nutrition.items():
                        total_nutrition[nutrient] += value * 0.3  # 평균 분량 가정
                    break
        
        # 기본 분석 결과
        analysis = {
            "nutrition_summary": total_nutrition,
            "health_score": 7,
            "recommendations": ["균형잡힌 식사를 위해 채소를 추가하세요.", "충분한 수분 섭취를 하세요."],
            "dietary_notes": "영양 정보는 추정치입니다."
        }
        
        return analysis
    
    def get_personalized_feedback(self, nutrition_analysis: Dict, user_id: str = "default") -> str:
        """개인 맞춤 영양 피드백"""
        user_profile = self.user_memory.get_user_profile(user_id)
        dietary_goals = user_profile.get('dietary_goals', {})
        
        nutrition_summary = nutrition_analysis['nutrition_summary']
        
        feedback = f"""
오늘 요리하신 음식의 영양 분석 결과입니다!

칼로리: {nutrition_summary['calories']:.0f} kcal
단백질: {nutrition_summary['protein']:.1f}g  
탄수화물: {nutrition_summary['carbs']:.1f}g
지방: {nutrition_summary['fat']:.1f}g

건강한 식사를 위한 조언:
- 채소를 추가하여 식이섬유와 비타민을 보충하세요
- 충분한 수분 섭취를 잊지 마세요
- 규칙적인 식사 시간을 유지하세요
        """
        
        return feedback.strip()


class MemoryAgent:
    """사용자 기억 관리 에이전트"""
    
    def __init__(self, user_memory: UserMemory):
        self.user_memory = user_memory
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    def save_cooking_session(self, state: RecipeState, user_id: str = "default"):
        """요리 세션 저장"""
        if state.get('selected_recipe') and state.get('nutrition_analysis'):
            try:
                self.user_memory.add_cooking_record(
                    user_id=user_id,
                    recipe_title=state['selected_recipe']['title'],
                    ingredients_used=state['selected_recipe'].get('ingredients_std', []),
                    nutrition_info=state['nutrition_analysis'],
                    feedback=state.get('user_feedback', '')
                )
            except Exception as e:
                print(f"요리 세션 저장 오류: {e}")


# LangGraph 노드 함수들
class CookingAssistant:
    """요리 어시스턴트 메인 클래스"""
    
    def __init__(self, faiss_path: str, jsonl_path: str):
        # 데이터베이스 및 에이전트 초기화
        self.recipe_db = RecipeDatabase(faiss_path, jsonl_path)
        self.user_memory = UserMemory()
        
        self.planner = PlannerAgent(self.recipe_db, self.user_memory)
        self.chef = ChefAgent()
        self.nutrition = NutritionAgent(self.user_memory)
        self.memory = MemoryAgent(self.user_memory)
        
        # LangGraph 구성
        self.setup_graph()
    
    def setup_graph(self):
        """LangGraph 워크플로우 설정"""
        workflow = StateGraph(RecipeState)
        
        # 노드 추가
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("recipe_selection", self.recipe_selection_node)
        workflow.add_node("chef", self.chef_node)
        workflow.add_node("nutrition", self.nutrition_node)
        workflow.add_node("memory", self.memory_node)
        
        # 엣지 설정
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "recipe_selection")
        workflow.add_edge("recipe_selection", "chef")
        
        # 조건부 엣지 - chef에서 조리 완료 여부에 따라 분기
        workflow.add_conditional_edges(
            "chef",
            self.should_continue_cooking,
            {
                "continue": "chef",
                "complete": "nutrition"
            }
        )
        
        workflow.add_edge("nutrition", "memory")
        workflow.add_edge("memory", END)
        
        self.graph = workflow.compile()
    
    def planner_node(self, state: RecipeState) -> RecipeState:
        """Planner Agent 노드"""
        try:
            constraints = self.planner.process_user_input(state['user_query'])
            candidates = self.planner.search_and_rank_recipes(constraints)
            
            if not candidates:
                message = "죄송합니다. 요청하신 조건에 맞는 레시피를 찾을 수 없습니다. 다른 조건으로 시도해보시겠어요?"
                return {
                    **state,
                    "user_constraints": constraints,
                    "recipe_candidates": [],
                    "next_action": "retry",
                    "messages": [AIMessage(content=message)]
                }
            
            # 후보 레시피 메시지 생성
            candidates_text = "\n".join([
                f"{i+1}. {recipe.get('title', '제목없음')} (점수: {recipe.get('final_score', 0):.2f})"
                for i, recipe in enumerate(candidates)
            ])
            
            message = f"다음 {len(candidates)}개의 요리를 찾았습니다:\n\n{candidates_text}\n\n첫 번째 레시피로 요리를 시작하겠습니다!"
            
            return {
                **state,
                "user_constraints": constraints,
                "recipe_candidates": candidates,
                "next_action": "select_recipe",
                "messages": [AIMessage(content=message)]
            }
            
        except Exception as e:
            print(f"Planner 오류: {e}")
            message = "레시피 검색 중 오류가 발생했습니다. 다시 시도해주세요."
            return {
                **state,
                "user_constraints": {},
                "recipe_candidates": [],
                "next_action": "retry",
                "messages": [AIMessage(content=message)]
            }
    
    def recipe_selection_node(self, state: RecipeState) -> RecipeState:
        """레시피 선택 처리 노드"""
        if state.get('recipe_candidates'):
            selected_recipe = state['recipe_candidates'][0]
            
            ingredients_text = "\n".join([
                f"- {ing}" for ing in selected_recipe.get('ingredients_std', selected_recipe.get('ingredients', []))[:5]
            ])
            
            message = f"'{selected_recipe.get('title', '선택된 요리')}' 요리를 시작하겠습니다!\n\n필요한 재료 (일부):\n{ingredients_text}\n\n조리를 시작할게요!"
            
            return {
                **state,
                "selected_recipe": selected_recipe,
                "current_step": 0,
                "cooking_complete": False,
                "next_action": "start_cooking",
                "messages": [AIMessage(content=message)]
            }
        
        return state
    
    def chef_node(self, state: RecipeState) -> RecipeState:
        """Chef Agent 노드"""
        if not state.get('selected_recipe'):
            return state
            
        try:
            if state['current_step'] == 0:
                # 조리 시작
                cooking_session = self.chef.start_cooking_guidance(state['selected_recipe'])
                step_info = self.chef.get_next_step(cooking_session)
                
                if step_info.get('completed'):
                    return {
                        **state,
                        "cooking_complete": True,
                        "messages": [AIMessage(content="조리가 완료되었습니다! 영양 분석을 진행하겠습니다.")]
                    }
                
                message = f"조리를 시작하겠습니다!\n\n단계 {step_info['step_number']}/{step_info['total_steps']}:\n{step_info['instruction']}"
                
                return {
                    **state,
                    "current_step": 1,
                    "messages": [AIMessage(content=message)]
                }
            else:
                # 다음 단계 안내
                cooking_session = {
                    "recipe_title": state['selected_recipe'].get('title', ''),
                    "current_step": state['current_step'],
                    "steps": state['selected_recipe'].get('steps_parsed', []),
                    "ingredients": state['selected_recipe'].get('ingredients_std', [])
                }
                
                step_info = self.chef.get_next_step(cooking_session)
                
                if step_info.get('completed'):
                    return {
                        **state,
                        "cooking_complete": True,
                        "messages": [AIMessage(content="조리가 완료되었습니다! 영양 분석을 진행하겠습니다.")]
                    }
                else:
                    message = f"단계 {step_info['step_number']}/{step_info['total_steps']}:\n{step_info['instruction']}"
                    
                    return {
                        **state,
                        "current_step": state['current_step'] + 1,
                        "messages": [AIMessage(content=message)]
                    }
                    
        except Exception as e:
            print(f"Chef 노드 오류: {e}")
            return {
                **state,
                "cooking_complete": True,
                "messages": [AIMessage(content="조리 과정에서 오류가 발생했습니다. 영양 분석으로 넘어가겠습니다.")]
            }
    
    def nutrition_node(self, state: RecipeState) -> RecipeState:
        """Nutrition Agent 노드"""
        if not state.get('selected_recipe'):
            return state
            
        try:
            analysis = self.nutrition.analyze_nutrition(state['selected_recipe'])
            feedback = self.nutrition.get_personalized_feedback(analysis)
            
            message = f"🍽️ 영양 분석 결과:\n\n{feedback}\n\n⭐ 건강 점수: {analysis.get('health_score', 7)}/10"
            
            return {
                **state,
                "nutrition_analysis": analysis,
                "messages": [AIMessage(content=message)]
            }
            
        except Exception as e:
            print(f"Nutrition 노드 오류: {e}")
            message = "영양 분석 중 오류가 발생했습니다."
            return {
                **state,
                "nutrition_analysis": {"health_score": 5, "nutrition_summary": {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}},
                "messages": [AIMessage(content=message)]
            }
    
    def memory_node(self, state: RecipeState) -> RecipeState:
        """Memory Agent 노드"""
        try:
            # 요리 세션 저장
            self.memory.save_cooking_session(state)
            
            message = "요리 기록이 저장되었습니다. 맛있게 드세요! 🍽️\n\n또 다른 요리를 도와드릴까요?"
            
            return {
                **state,
                "messages": [AIMessage(content=message)]
            }
            
        except Exception as e:
            print(f"Memory 노드 오류: {e}")
            message = "기록 저장 중 일부 오류가 있었지만, 요리는 완성되었습니다! 맛있게 드세요! 🍽️"
            return {
                **state,
                "messages": [AIMessage(content=message)]
            }
    
    def should_continue_cooking(self, state: RecipeState) -> str:
        """조리 계속 여부 결정"""
        # 조리 완료 상태이거나 단계가 많이 진행된 경우 완료로 판정
        if state.get('cooking_complete', False) or state.get('current_step', 0) >= 6:
            return "complete"
        else:
            return "continue"
    
    def run(self, user_query: str) -> RecipeState:
        """요리 어시스턴트 실행"""
        initial_state = RecipeState(
            messages=[HumanMessage(content=user_query)],
            user_query=user_query,
            user_constraints={},
            recipe_candidates=[],
            selected_recipe=None,
            current_step=0,
            cooking_complete=False,
            nutrition_analysis=None,
            user_feedback=None,
            next_action="plan"
        )
        
        try:
            result = self.graph.invoke(initial_state)
            return result
        except Exception as e:
            print(f"Graph 실행 오류: {e}")
            # 오류 발생 시 기본 응답
            error_state = initial_state.copy()
            error_state['messages'] = [AIMessage(content=f"죄송합니다. 시스템 오류가 발생했습니다: {str(e)}")]
            return error_state


def main():
    """메인 실행 함수"""
    # 파일 경로 설정 (현재 디렉터리 기준)
    faiss_path = "data/recipes30.faiss"
    jsonl_path = "data/recipes30_clean.jsonl"
    
    try:
        # 요리 어시스턴트 초기화
        print("🔧 요리 어시스턴트를 초기화하고 있습니다...")
        assistant = CookingAssistant(faiss_path, jsonl_path)
        
        # 예시 실행
        user_query = "I have a cho"
        
        print("🍳 요리 AI 어시스턴트를 시작합니다!")
        print(f"사용자 요청: {user_query}")
        print("=" * 50)
        
        result = assistant.run(user_query)
        
        # 결과 출력
        for i, message in enumerate(result.get('messages', [])):
            if hasattr(message, 'content'):
                print(f"\n🤖 어시스턴트: {message.content}")
                print("-" * 30)
                
    except FileNotFoundError as e:
        print(f"❌ 파일을 찾을 수 없습니다: {e}")
        print("파일 경로를 확인하고 recipes30.faiss와 recipes30_clean.jsonl 파일이 존재하는지 확인하세요.")
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {e}")
        print("OpenAI API 키가 설정되어 있는지 확인하세요.")


if __name__ == "__main__":
    main()