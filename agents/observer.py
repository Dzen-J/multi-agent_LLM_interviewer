import time
from typing import Dict, Any, Tuple
import json
from pathlib import Path
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from config.settings import settings
from core.state import InterviewState, Assessment
from core.rag import KnowledgeBase

class ObserverAgent:
    def __init__(self, llm: ChatMistralAI = None, knowledge_base: KnowledgeBase = None):
        self.llm = llm or ChatMistralAI(
            model=settings.MISTRAL_MODEL,
            temperature=0.2,
            api_key=settings.MISTRAL_API_KEY,
            timeout=30, 
            max_retries=3 
        )
        
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.prompt_template = self._load_prompt_template()
    
    def _load_prompt_template(self) -> ChatPromptTemplate:
        current_dir = Path(__file__).parent
        template_path = current_dir.parent / "prompts" / "observer.txt"
        
        if not template_path.exists():
            # Fallback на простой промпт
            print(f"Файл промпта не найден: {template_path}")
            return ChatPromptTemplate.from_template("""
            Проанализируй ответ кандидата на технический вопрос.
            
            Тема: {current_topic}
            Ожидаемый уровень: {expected_level}
            Вопрос: {question}
            Ответ кандидата: {answer}
            
            Проверка знаний: {knowledge_check_result}
            
            Верни ответ в формате JSON:
            {{
                "technical_score": число от 0 до 10,
                "completeness_score": число от 0 до 10,
                "confidence_score": число от 0 до 10,
                "communication_score": число от 0 до 10,
                "has_errors": true/false,
                "errors_list": ["ошибка1", "ошибка2"],
                "is_evasive": true/false,
                "depth_of_knowledge": "shallow/adequate/deep",
                "recommendation_for_next_question": "текст рекомендации",
                "suggested_correction": "правильный ответ если есть ошибки"
            }}
            """)
        
        template_content = template_path.read_text(encoding='utf-8')
        return ChatPromptTemplate.from_template(template_content)
    
    def analyze_answer(self, state: InterviewState, 
              question: str, answer: str) -> Tuple[Dict, Assessment]:
        """Анализирует ответ кандидата и обновляет оценку"""
        
        print(f"\n Observer: Начинаю анализ...")
        print(f"   Получен вопрос: {'ДА' if question else 'НЕТ'} (длина: {len(question) if question else 0})")
        print(f"   Получен ответ: {'ДА' if answer else 'НЕТ'} (длина: {len(answer) if answer else 0})")
        
        if not question or not answer:
            print("Пропускаю анализ: нет вопроса или ответа")
            return self._create_empty_analysis(), state["assessment"]
        
        # Убедимся, что ответ не слишком длинный для API
        if len(answer) > 4000:
            answer = answer[:4000] + "... [ответ обрезан]"
        
        # Проверяем ответ через RAG (если включено)
        if settings.RAG_ENABLED:
            try:
                verification = self.knowledge_base.verify_technical_answer(
                    question=question,
                    answer=answer,
                    topic=state.get("current_topic", "python")
                )
            except Exception as e:
                pass
        else:
            verification = {
                "is_correct": None,
                "confidence": 0.5,
                "correct_info": "RAG отключен",
                "suggested_topics": []
            }
        
        # Пытаемся получить анализ от LLM
        analysis = None
        try:
            print(f"Анализ через LLM...")
            
            # Форматируем промпт
            formatted_prompt = self.prompt_template.format(
                current_topic=state.get("current_topic", "Неизвестно"),
                expected_level=state["candidate_info"].grade,
                question=question,
                answer=answer,
                knowledge_check_result=f"Проверка: {verification['confidence']:.2f} уверенности. {verification['correct_info']}"
            )
            
            # Упрощенный вызов LLM
            response = self.llm.invoke(formatted_prompt)
            
            # Очищаем и парсим JSON ответ
            content = self._clean_json_response(response.content)
            analysis = json.loads(content)
            print(f"Анализ получен. Оценка: {analysis.get('technical_score', 0)}/10")
            
        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга JSON: {e}")
            print(f"Сырой ответ: {response.content[:200] if 'response' in locals() else 'Нет ответа'}...")
            analysis = self._fallback_analysis(answer, verification)
            
        except Exception as e:
            print(f"Ошибка запроса к LLM: {e}")
            analysis = self._fallback_analysis(answer, verification)
        
        # Обновляем общую оценку
        updated_assessment = self._update_assessment(state["assessment"], analysis, verification)
        
        return analysis, updated_assessment

    def _clean_json_response(self, text: str) -> str:
        """Очищает JSON ответ от управляющих символов и извлекает JSON"""
        if not text:
            return '{"technical_score": 5, "completeness_score": 5, "confidence_score": 5, "communication_score": 5, "has_errors": false, "errors_list": [], "is_evasive": false, "depth_of_knowledge": "adequate", "recommendation_for_next_question": "Продолжить тему", "suggested_correction": ""}'
        
        import re
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        # Убедимся, что это валидный JSON
        try:
            json.loads(text)
            return text
        except:
            import json
            json_pattern = r'\{.*\}'
            matches = re.search(json_pattern, text, re.DOTALL)
            if matches:
                return matches.group(0)
            
            # Fallback - создаем простой JSON
            return json.dumps({
                "technical_score": 5,
                "completeness_score": 5,
                "confidence_score": 5,
                "communication_score": 5,
                "has_errors": False,
                "errors_list": [],
                "is_evasive": False,
                "depth_of_knowledge": "adequate",
                "recommendation_for_next_question": "Продолжить тему",
                "suggested_correction": ""
            })
    
    def _create_empty_analysis(self) -> Dict:
        """Создает пустой анализ при отсутствии данных"""
        return {
            "technical_score": 0,
            "completeness_score": 0,
            "confidence_score": 0,
            "communication_score": 5,
            "has_errors": False,
            "errors_list": [],
            "is_evasive": False,
            "depth_of_knowledge": "unknown",
            "recommendation_for_next_question": "Продолжить тему",
            "suggested_correction": ""
        }
    
    def _fallback_analysis(self, answer: str, verification: Dict) -> Dict:
        """Fallback анализ если LLM не сработал"""
        answer_lower = answer.lower().strip()
        answer_length = len(answer.split())
        
        # Проверяем короткие/пустые ответы
        short_responses = ["ничем", "не знаю", "не помню", "нет", "не", "no", "don't know"]
        
        if answer_lower in short_responses or answer_length <= 2:
            return {
                "technical_score": 1,
                "completeness_score": 1,
                "confidence_score": 2,
                "communication_score": 3,
                "has_errors": True,
                "errors_list": ["Слишком краткий ответ", "Не показано понимание темы"],
                "is_evasive": True,
                "depth_of_knowledge": "shallow",
                "recommendation_for_next_question": "Задать более простой вопрос по той же теме или уточняющий вопрос",
                "suggested_correction": "Списки изменяемы (mutable), кортежи неизменяемы (immutable). Списки используют [], кортежи используют (). Пример использования списка: для хранения элементов, которые могут изменяться. Пример использования кортежа: для координат (x, y) или константных данных."
            }
        
        # Обычная эвристика для более длинных ответов
        if answer_length < 15:
            score = 3
            confidence = 3
            is_evasive = True
        elif answer_length > 100:
            score = 7
            confidence = 8
            is_evasive = False
        else:
            score = 5
            confidence = 5
            is_evasive = False
        
        return {
            "technical_score": score,
            "completeness_score": score,
            "confidence_score": confidence,
            "communication_score": 5,
            "has_errors": verification.get("confidence", 0.5) < 0.4,
            "errors_list": ["Неполный ответ"] if answer_length < 25 else [],
            "is_evasive": is_evasive,
            "depth_of_knowledge": "adequate" if score > 6 else "shallow",
            "recommendation_for_next_question": "Продолжить тему" if score > 5 else "Упростить вопрос",
            "suggested_correction": verification.get("correct_info", "")
        }
    
    def _update_assessment(self, assessment: Assessment, 
                          analysis: Dict, verification: Dict) -> Assessment:
        """Обновляет общую оценку на основе анализа ответа"""
        
        # Обновляем технические баллы
        assessment.technical_score = (
            assessment.technical_score * 0.7 + analysis["technical_score"] * 0.3
        )
        
        # Обновляем оценку коммуникации
        assessment.communication_score = (
            assessment.communication_score * 0.7 + analysis["communication_score"] * 0.3
        )
        
        # Обновляем уверенность
        assessment.confidence_score = (
            assessment.confidence_score * 0.7 + analysis["confidence_score"] * 0.3
        )
        
        # Добавляем пробелы в знаниях если есть ошибки
        if analysis["has_errors"] and analysis.get("suggested_correction"):
            topic = "текущая тема"
            assessment.knowledge_gaps[topic] = analysis["suggested_correction"][:200]
        
        # Добавляем заметки по софт скиллам
        if analysis["is_evasive"]:
            assessment.soft_skills_notes.append("Кандидат был уклончив в ответе")
        
        if analysis["depth_of_knowledge"] == "deep":
            assessment.soft_skills_notes.append("Показал глубокое понимание темы")
        
        return assessment
    
    def detect_off_topic(self, answer: str, current_topic: str) -> bool:
        """Определяет, пытается ли кандидат уйти от темы"""
        off_topic_keywords = [
            "не по теме", "другая тема", "погода", "отвлечемся",
            "в другой компании", "вообще", "в принципе"
        ]
        
        answer_lower = answer.lower()
        topic_keywords = current_topic.lower().split()
        has_topic_keywords = any(keyword in answer_lower for keyword in topic_keywords)
        # Проверяем наличие off-topic фраз
        has_off_topic = any(phrase in answer_lower for phrase in off_topic_keywords)
        
        return has_off_topic and not has_topic_keywords