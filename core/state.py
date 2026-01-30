from typing import TypedDict, List, Dict, Optional, Annotated, Any
from pydantic import BaseModel, Field
from datetime import datetime
import operator


class Message(BaseModel):
    role: str  # "user", "interviewer", "observer", "coordinator"
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Optional[Dict] = None

class CandidateInfo(BaseModel):
    name: str = "Анонимный кандидат"
    position: str
    grade: str
    experience_years: float
    technologies: List[str] = []

class Assessment(BaseModel):
    technical_score: float = 0.0
    communication_score: float = 0.0
    confidence_score: float = 0.0
    topics_covered: List[str] = []
    confirmed_skills: List[str] = []
    knowledge_gaps: Dict[str, str] = {}  # тема -> правильный ответ
    soft_skills_notes: List[str] = []

# Главное состояние для LangGraph
# core/state.py
class InterviewState(TypedDict):
    candidate_info: CandidateInfo
    messages: List[Message]
    internal_monologue: List[str]
    current_topic: str
    difficulty_level: int
    assessment: Assessment
    questions_asked: List[str]
    observer_recommendation: Optional[str]
    need_feedback: bool
    interview_complete: bool
    current_agent: str
    current_question: Optional[str]
    current_answer: Optional[str]
    coordinator_instruction: Optional[str]
    log_data: Dict[str, Any]  # Добавьте это!
    current_turn_thoughts: List[str]
    scenario_number: int

# Для аннотации в LangGraph
def add_message(messages: List[Message], message: Message) -> List[Message]:
    return messages + [message]

class StateManager:
    @staticmethod
    def get_conversation_history(state: InterviewState, last_n: int = 6) -> str:
        """Получить историю диалога для промпта"""
        history = []
        for msg in state["messages"][-last_n:]:
            if msg.role == "user":
                history.append(f"Кандидат: {msg.content}")
            elif msg.role == "interviewer":
                history.append(f"Интервьюер: {msg.content}")
        return "\n".join(history)
    
    @staticmethod
    def get_internal_thoughts(state: InterviewState) -> str:
        """Получить внутренние мысли для логов"""
        return " | ".join(state["internal_monologue"][-3:])

def validate_state(state: InterviewState) -> bool:
    """Проверяет, что состояние содержит все необходимые поля"""
    required_fields = [
        "candidate_info", "messages", "assessment", 
        "current_topic", "difficulty_level", "questions_asked"
    ]
    
    for field in required_fields:
        if field not in state:
            print(f"В состоянии отсутствует поле: {field}")
            return False
    
    return True