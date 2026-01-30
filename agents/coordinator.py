from typing import Dict, Any, List
import json
from pathlib import Path
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from config.settings import settings
from core.state import InterviewState, CandidateInfo
from core.state import StateManager
from core.state import Assessment

class CoordinatorAgent:
    def __init__(self, llm: ChatMistralAI = None):
        self.llm = llm or ChatMistralAI(
            model=settings.MISTRAL_MODEL,
            temperature=0.3,
            api_key=settings.MISTRAL_API_KEY
        )
        
        self.prompt_template = self._load_prompt_template("coordinator.txt")
    
    def _load_prompt_template(self, filename: str) -> ChatPromptTemplate:
        current_dir = Path(__file__).parent
        template_path = current_dir.parent / "prompts" / filename
        
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        
        template_content = template_path.read_text(encoding='utf-8')
        return ChatPromptTemplate.from_template(template_content)
    
    def decide_next_step(self, state: InterviewState) -> Dict[str, Any]:
        """Принимает решение о следующем шаге интервью"""
        
        candidate_info = state["candidate_info"]
        technologies = getattr(candidate_info, 'technologies', [])
        
        if isinstance(technologies, list):
            technologies_str = ", ".join(technologies)
        else:
            technologies_str = str(technologies)
        
        assessment = state.get("assessment", Assessment())
        current_score = getattr(assessment, 'technical_score', 0)
        
        prompt_data = {
            "candidate_name": candidate_info.name,
            "position": candidate_info.position,
            "grade": candidate_info.grade,
            "experience_years": candidate_info.experience_years,
            "technologies": technologies_str,
            "history": self._get_conversation_history(state, 4),
            "current_topic": state.get("current_topic", "Нет темы"),
            "difficulty": state.get("difficulty_level", 2),
            "questions_count": len(state.get("questions_asked", [])),
            "max_turns": settings.MAX_TURNS,
            "current_score": round(current_score, 1),
            "observer_notes": state.get("observer_recommendation", "Нет заметок")
        }
        
        formatted_prompt = self.prompt_template.format(**prompt_data)
        
        try:
            response = self.llm.invoke(formatted_prompt)
            content = response.content.strip()
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # управляющие символы
            content = content.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            
            decision = json.loads(content)
            
            if decision.get("action") not in ["continue", "change_topic", "end_interview"]:
                decision["action"] = "continue"
            
            decision.setdefault("reasoning", "Продолжаем интервью")
            decision.setdefault("instruction_to_interviewer", "Задай следующий вопрос")
            
            return decision
            
        except (json.JSONDecodeError, AttributeError) as e:
            # Fallback решение
            return self._create_fallback_decision(state)
        except Exception as e:
            return self._create_fallback_decision(state)

    def _create_fallback_decision(self, state: InterviewState) -> Dict[str, Any]:
        """Создает fallback решение"""
        questions_count = len(state.get("questions_asked", []))
        
        if questions_count >= settings.MAX_TURNS:
            action = "end_interview"
            reasoning = "Достигнут лимит вопросов"
        elif questions_count >= 5 and state.get("assessment", {}).get("technical_score", 0) > 7:
            action = "end_interview"
            reasoning = "Кандидат демонстрирует высокий уровень"
        else:
            action = "continue"
            reasoning = "Продолжаем интервью (fallback)"
        
        return {
            "action": action,
            "new_topic": state.get("current_topic", "Python"),
            "new_difficulty": state.get("difficulty_level", 2),
            "reasoning": reasoning,
            "instruction_to_interviewer": "Задай следующий вопрос"
        }
    
    def should_end_interview(self, state: InterviewState) -> bool:
        """Проверяет, нужно ли завершить интервью"""
        if state.get("interview_complete", False):
            return True
        
        questions_count = len(state.get("questions_asked", []))
        
        if questions_count >= settings.MAX_TURNS:
            print(f"Достигнут лимит вопросов: {questions_count}/{settings.MAX_TURNS}")
            return True
        
        # Проверяем, достаточно ли данных для оценки (если есть оценка)
        assessment = state.get("assessment")
        if assessment and hasattr(assessment, 'technical_score'):
            if questions_count >= 5 and assessment.technical_score > 7:
                print(f"Кандидат демонстрирует высокий уровень: {assessment.technical_score}/10")
                return True
        
        return False
    
    def _get_conversation_history(self, state: InterviewState, last_n: int = 4) -> str:
        """Получает историю диалога для промпта"""
        messages = state.get("messages", [])
        if not messages:
            return "Нет истории диалога"
        
        history = []
        for msg in messages[-last_n:]:
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                role = "Интервьюер" if msg.role == "interviewer" else "Кандидат"
                history.append(f"{role}: {msg.content}")
            else:
                # Если это словарь -> проверяем структуру
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    role = "Интервьюер" if msg['role'] == "interviewer" else "Кандидат"
                    history.append(f"{role}: {msg['content']}")
        
        return "\n".join(history)