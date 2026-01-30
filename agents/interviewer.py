from typing import Dict, Any, List
import random
from pathlib import Path
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from config.settings import settings
from core.state import InterviewState, Message
from core.state import StateManager

class InterviewerAgent:
    def __init__(self, llm: ChatMistralAI = None):
        self.llm = llm or ChatMistralAI(
            model=settings.MISTRAL_MODEL,
            temperature=0.7,  # Более "креативные" вопросы
            api_key=settings.MISTRAL_API_KEY
        )
        
        self.prompt_template = self._load_prompt_template("interviewer.txt")
        
         # Минимальный fallback банк (только для крайних случаев)
        self.fallback_questions = {
            "python": "Расскажите о своем опыте работы с Python?",
            "database": "Как вы проектируете схемы баз данных?",
            "algorithm": "Какой алгоритм вы считаете самым полезным в вашей работе?",
            "default": "Расскажите о самом сложном техническом задаче в вашем опыте?"
        }
    
    def _load_prompt_template(self, filename: str) -> ChatPromptTemplate:
        current_dir = Path(__file__).parent
        template_path = current_dir.parent / "prompts" / filename
        
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        
        template_content = template_path.read_text(encoding='utf-8')
        return ChatPromptTemplate.from_template(template_content)
    
    def _get_conversation_history(self, state: InterviewState, last_n: int = 4) -> str:
        """Получить историю диалога для промпта"""
        messages = state.get("messages", [])
        if not messages:
            return "Нет истории"
        
        history = []
        for msg in messages[-last_n:]:
            if msg.role == "user":
                history.append(f"Кандидат: {msg.content}")
            elif msg.role == "interviewer":
                history.append(f"Интервьюер: {msg.content}")
        
        return "\n".join(history)
    
    def generate_question(self, state: InterviewState) -> str:
        """Генерирует следующий вопрос"""
        topic = state.get("current_topic", "")
        difficulty = state.get("difficulty_level", 2)
        asked_questions = state.get("questions_asked", [])
        recent_questions = asked_questions[-5:] if len(asked_questions) > 5 else asked_questions # Формирую список последних 5 вопросов для промпта
        
        # Подготавливаем данные для промпта
        prompt_data = {
            "position": state["candidate_info"].position,
            "grade": state["candidate_info"].grade,
            "technologies": ", ".join(state["candidate_info"].technologies[:5]),
            "current_topic": topic,
            "difficulty": difficulty,
            "coordinator_instruction": state.get("coordinator_instruction", "Задай следующий вопрос"),
            "observer_feedback": state.get("observer_recommendation", "Нет обратной связи"),
            "history": self._get_conversation_history(state, 4),
            "asked_questions": recent_questions
        }
        
        try:
            formatted_prompt = self.prompt_template.format(**prompt_data)
            print(f"\n Генерация вопроса по теме '{topic}' (сложность {difficulty}/5)...")
            response = self.llm.invoke(formatted_prompt)
            question = response.content.strip()
            
            # Очищаем ответ
            for prefix in ["Вопрос:", "Question:", "Q:", "Интервьюер:"]:
                if question.startswith(prefix):
                    question = question[len(prefix):].strip()
    
            if not question.endswith('?'):
                question = question + '?'
            
            # Проверяем, не повторяется ли вопрос
            if question in asked_questions:
                print(f"Вопрос уже был задан, генерирую другой...")
                # Генерируем альтернативный вопрос
                question = self._generate_alternative_question(topic, difficulty, asked_questions)
            
            print(f"Новый вопрос: {question[:100]}...")
            return question
            
        except Exception as e:
            print(f"Ошибка генерации вопроса: {e}")
            return self._get_fallback_question(topic, difficulty, asked_questions)

    def _generate_alternative_question(self, topic: str, difficulty: int, asked_questions: list) -> str:
        """Генерирует альтернативный вопрос при повторе"""
        try:
            alt_prompt = f"""
            Придумай ДРУГОЙ технический вопрос по теме "{topic}" (сложность {difficulty}/5).
            
            Уже заданные вопросы (НЕ ИСПОЛЬЗУЙ их!):
            {chr(10).join(f'- {q}' for q in asked_questions[-3:])}
            
            Сфокусируйся на другом аспекте темы. Например:
            - Если были вопросы про синтаксис, спроси про практическое применение
            - Если были теоретические вопросы, спроси про оптимизацию
            - Если были вопросы про базовые концепции, спроси про продвинутые темы
            
            Только вопрос, без пояснений.
            """
            
            response = self.llm.invoke(alt_prompt)
            question = response.content.strip()
            
            if question and question not in asked_questions:
                return question
        
        except Exception:
            pass
        
        # Если не получилось, используем fallback
        return self._get_fallback_question(topic, difficulty, asked_questions)

    def _get_fallback_question(self, topic: str, difficulty: int, asked_questions: list) -> str:
        """Крайний fallback - минимальный набор"""
        fallback_questions = {
            "python": [
                "Расскажите о своем опыте работы с Python?",
                "Какой проект на Python вы считаете самым сложным и почему?",
                "С какими Python библиотеками вы работали?",
                "Как вы отлаживаете Python код?",
                "Расскажите о паттернах проектирования в Python?"
            ],
            "базы данных": [
                "Как вы проектируете схемы баз данных?",
                "Как оптимизируете медленные SQL запросы?",
                "Расскажите о вашем опыте работы с транзакциями?",
                "Как обеспечиваете безопасность баз данных?",
                "Как работаете с миграциями схемы?"
            ]
        }
        
        # Находим вопросы по теме
        topic_lower = topic.lower()
        for key, questions in fallback_questions.items():
            if key in topic_lower:
                for q in questions:
                    if q not in asked_questions:
                        return q
        
        # Общий fallback
        return f"Расскажите о своем опыте работы с {topic}?"
    
    def adapt_difficulty(self, state: InterviewState, 
                        last_answer_quality: float) -> int:
        """Адаптирует сложность вопросов на основе качества ответов"""
        current_difficulty = state.get("difficulty_level", 2)
        
        if last_answer_quality > 0.8:  # Отличный ответ
            new_difficulty = min(5, current_difficulty + 1)
            state["internal_monologue"].append(
                f"Interviewer: Повышаю сложность с {current_difficulty} до {new_difficulty}"
            )
        elif last_answer_quality < 0.4:  # Плохой ответ
            new_difficulty = max(1, current_difficulty - 1)
            state["internal_monologue"].append(
                f"Interviewer: Понижаю сложность с {current_difficulty} до {new_difficulty}"
            )
        else:
            new_difficulty = current_difficulty
        
        return new_difficulty