from typing import Dict, Any, List
from pathlib import Path
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from config.settings import settings
from core.state import InterviewState, Assessment, CandidateInfo
from core.rag import KnowledgeBase

class FeedbackGenerator:
    def __init__(self, llm: ChatMistralAI = None, knowledge_base: KnowledgeBase = None):
        self.llm = llm or ChatMistralAI(
            model=settings.MISTRAL_MODEL,
            temperature=0.4,
            api_key=settings.MISTRAL_API_KEY
        )
        
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.prompt_template = self._load_prompt_template("feedback.txt")
    
    def _load_prompt_template(self, filename: str) -> ChatPromptTemplate:
        """Загружает промпт из Jinja2 файла"""
        current_dir = Path(__file__).parent
        template_path = current_dir.parent / "prompts" / filename
        
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        
        template_content = template_path.read_text(encoding='utf-8')
        return ChatPromptTemplate.from_template(template_content)
    
    def generate_feedback(self, state: InterviewState, 
                     duration_minutes: float) -> Dict[str, Any]:
        """Генерирует финальный фидбэк по результатам интервью"""
        
        # Безопасно получаем assessment
        assessment = state.get("assessment")
        if assessment is None:
            # Создаем пустой assessment
            from core.state import Assessment
            assessment = Assessment()
        
        candidate = state["candidate_info"]
        
        # Безопасно получаем атрибуты assessment
        def safe_get(obj, attr, default=None):
            if hasattr(obj, attr):
                return getattr(obj, attr)
            return default
        
        candidate_level = self._determine_level(assessment, candidate) # уровень кандидата
        hiring_recommendation = self._determine_hiring_recommendation(assessment, candidate) # рекомендация по найму
        confidence = self._calculate_confidence(assessment) # Уверенность
        
        # Форматируем данные для промпта
        confirmed_skills = safe_get(assessment, 'confirmed_skills', [])
        confirmed_skills_str = "\n".join([f"- {skill}" for skill in confirmed_skills]) if confirmed_skills else "- Нет данных"
        
        knowledge_gaps = safe_get(assessment, 'knowledge_gaps', {})
        knowledge_gaps_str = "\n".join([f"- **{topic}**: {correction[:100]}..." for topic, correction in knowledge_gaps.items()]) if knowledge_gaps else "- Нет пробелов"
        
        roadmap = self._generate_roadmap(assessment, candidate)
        roadmap_str = "\n".join([f"- {item}" for item in roadmap])
        
        learning_resources = []
        if hasattr(assessment, 'knowledge_gaps') and assessment.knowledge_gaps:
            for topic in assessment.knowledge_gaps.keys():
                try:
                    resources = self.knowledge_base.get_learning_resources(
                        topic=topic,
                        difficulty=state.get("difficulty_level", 2)
                    )
                    learning_resources.extend(resources[:2])
                except:
                    pass
        
        learning_resources_str = "\n".join([f"- {r.get('topic', 'Тема')}: {r.get('content', '')[100]}..." for r in learning_resources]) if learning_resources else "- Нет рекомендаций"
        
        assessment_summary = self._format_assessment_summary(assessment)
        
        prompt_data = {
            "candidate_name": candidate.name,
            "position": candidate.position,
            "grade": candidate.grade,
            "duration": f"{duration_minutes:.1f}",
            "total_questions": len(state.get("questions_asked", [])),
            "assessment_summary": assessment_summary,
            "candidate_level": candidate_level,
            "hiring_recommendation": hiring_recommendation,
            "confidence": confidence,
            "confirmed_skills_str": confirmed_skills_str,
            "knowledge_gaps_str": knowledge_gaps_str,
            "clarity_score": safe_get(assessment, 'communication_score', 5.0),
            "honesty_score": 8.0 if not knowledge_gaps else 5.0,
            "engagement_score": 7.0,
            "soft_skills_notes": ", ".join(safe_get(assessment, 'soft_skills_notes', [])),
            "roadmap_str": roadmap_str,
            "learning_resources_str": learning_resources_str
        }
        
        try:
            print(f"\n Генерация фидбэка...")
            formatted_prompt = self.prompt_template.format(**prompt_data)
            response = self.llm.invoke(formatted_prompt)
            
            feedback = {
                "verdict": {
                    "grade": candidate_level,
                    "hiring_recommendation": hiring_recommendation,
                    "confidence_score": confidence
                },
                "hard_skills_analysis": {
                    "confirmed_skills": confirmed_skills,
                    "knowledge_gaps": dict(knowledge_gaps)
                },
                "soft_skills_analysis": {
                    "clarity": safe_get(assessment, 'communication_score', 5.0),
                    "honesty": 8.0 if not knowledge_gaps else 5.0,
                    "engagement": 7.0,
                    "notes": safe_get(assessment, 'soft_skills_notes', [])
                },
                "roadmap": roadmap,
                "learning_resources": learning_resources,
                "full_text_feedback": response.content
            }
            
            print(f" Фидбэк сгенерирован успешно")
            return feedback
            
        except Exception as e:
            print(f" Ошибка генерации фидбэка: {e}")
            return self._generate_fallback_feedback(assessment, candidate)

    def _determine_level(self, assessment: Assessment, candidate: CandidateInfo) -> str:
        """Определяет уровень кандидата (Junior/Middle/Senior)"""
        tech_score = assessment.technical_score
        
        if candidate.grade.lower() == "junior":
            if tech_score >= 7:
                return "Middle-ready Junior"
            return "Junior"
        elif candidate.grade.lower() == "middle":
            if tech_score >= 8:
                return "Senior-ready Middle"
            elif tech_score >= 6:
                return "Middle"
            else:
                return "Junior+"
        else:  # Senior
            if tech_score >= 9:
                return "Senior"
            elif tech_score >= 7:
                return "Middle+"
            else:
                return "Под вопросом"
    
    def _determine_hiring_recommendation(self, assessment: Assessment, 
                                        candidate: CandidateInfo) -> str:
        """Определяет рекомендацию по найму"""
        tech_score = assessment.technical_score
        comm_score = assessment.communication_score
        
        if tech_score >= 8 and comm_score >= 7:
            return "Strong Hire"
        elif tech_score >= 6 and comm_score >= 5:
            return "Hire"
        elif tech_score >= 4:
            return "Hire with reservations"
        else:
            return "No Hire"
    
    def _calculate_confidence(self, assessment: Assessment) -> float:
        """Рассчитывает уверенность в оценке"""
        base_confidence = min(assessment.confidence_score * 10, 100)
        return round(base_confidence, 1)
    
    def _format_assessment_summary(self, assessment: Assessment) -> str:
        """Форматирует сводку оценки для промпта"""
        def safe_get_score(attr, default=0.0):
            return getattr(assessment, attr, default) if hasattr(assessment, attr) else default
        
        tech_score = safe_get_score('technical_score', 0.0)
        comm_score = safe_get_score('communication_score', 0.0)
        conf_score = safe_get_score('confidence_score', 0.0)
        
        return f"""
        Технический балл: {tech_score:.1f}/10
        Коммуникация: {comm_score:.1f}/10
        Уверенность: {conf_score:.1f}/10
        """
    
    def _generate_roadmap(self, assessment: Assessment, 
                         candidate: CandidateInfo) -> List[str]:
        """Генерирует персонализированный roadmap"""
        roadmap = []
        
        # Для пробелов в знаниях
        for topic in assessment.knowledge_gaps.keys():
            roadmap.append(f"Изучить {topic}: {assessment.knowledge_gaps[topic][:100]}...")
        
        # Общие рекомендации
        if assessment.technical_score < 6:
            roadmap.append("Углубить понимание основных концепций")
        
        if assessment.communication_score < 5:
            roadmap.append("Потренироваться в объяснении технических концепций")
        
        # Рекомендации по грейду
        if candidate.grade == "Junior" and assessment.technical_score > 7:
            roadmap.append("Начать изучение продвинутых тем для перехода на Middle")
        
        if candidate.grade == "Middle" and assessment.technical_score > 8:
            roadmap.append("Изучить архитектурные паттерны и проектирование систем")
        
        return roadmap[:5]
    
    def _generate_fallback_feedback(self, assessment: Assessment, 
                                  candidate: CandidateInfo) -> Dict:
        """Fallback фидбэк если LLM не сработал"""
        return {
            "verdict": {
                "grade": "Не определен",
                "hiring_recommendation": "Требуется ручная проверка",
                "confidence_score": 50.0
            },
            "hard_skills_analysis": {
                "confirmed_skills": assessment.confirmed_skills,
                "knowledge_gaps": assessment.knowledge_gaps
            },
            "soft_skills_analysis": {
                "clarity": assessment.communication_score,
                "honesty": 5.0,
                "engagement": 5.0,
                "notes": ["Автоматическая оценка недоступна"]
            },
            "roadmap": ["Пройти повторное интервью после подготовки"],
            "learning_resources": [],
            "full_text_feedback": "Извините, не удалось сгенерировать подробный фидбэк."
        }