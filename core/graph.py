from typing import TypedDict, Annotated, List, Dict, Optional
import operator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import json

from agents.coordinator import CoordinatorAgent
from agents.interviewer import InterviewerAgent
from agents.observer import ObserverAgent
from agents.feedback_generator import FeedbackGenerator
from core.state import InterviewState, Message, Assessment, CandidateInfo
from core.logger import InterviewLogger
from core.rag import KnowledgeBase
from config import settings
from core.display import DisplayManager

class InterviewWorkflow:
    def __init__(self):
        # Инициализируем агентов
        self.knowledge_base = KnowledgeBase()
        
        self.coordinator = CoordinatorAgent()
        self.interviewer = InterviewerAgent()
        self.observer = ObserverAgent(knowledge_base=self.knowledge_base)
        self.feedback_gen = FeedbackGenerator(knowledge_base=self.knowledge_base)
        
        # Создаем граф
        self.workflow = StateGraph(InterviewState)
        
        # Добавляем ноды
        self.workflow.add_node("start_interview", self.start_interview)
        self.workflow.add_node("coordinator_decision", self.coordinator_decision)
        self.workflow.add_node("generate_question", self.generate_question)
        self.workflow.add_node("get_user_answer", self.get_user_answer)
        self.workflow.add_node("analyze_answer", self.analyze_answer)
        self.workflow.add_node("generate_feedback", self.generate_feedback)
        self.workflow.add_node("end_interview", self.end_interview)
        
        # Определяем edges
        self.workflow.set_entry_point("start_interview")
        
        self.workflow.add_edge("start_interview", "coordinator_decision")
        self.workflow.add_conditional_edges(
            "coordinator_decision",
            self.check_should_continue,
            {
                "continue": "generate_question",
                "end": "generate_feedback"
            }
        )
        self.workflow.add_edge("generate_question", "get_user_answer")
        self.workflow.add_edge("get_user_answer", "analyze_answer")
        self.workflow.add_edge("analyze_answer", "coordinator_decision")
        self.workflow.add_edge("generate_feedback", "end_interview")
        self.workflow.add_edge("end_interview", END)
        
        # Добавляем checkpoint memory
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)

        self.display = DisplayManager()
    
    def start_interview(self, state: InterviewState) -> InterviewState:
        """Инициализация интервью"""
        self.display.print_section("НАЧАЛО ИНТЕРВЬЮ")
        
        # Инициализируем лог в состоянии
        if "log_data" not in state:
            state["log_data"] = InterviewLogger.init_log_data(state["candidate_info"])
            print(f"📝 Лог инициализирован в start_interview")
        else:
            print(f"📝 Лог уже существует в start_interview, записей: {len(state['log_data'].get('turns', []))}")
        
        # Инициализируем остальные поля
        if "current_topic" not in state:
            state["current_topic"] = ""
        
        if "difficulty_level" not in state:
            state["difficulty_level"] = settings.DEFAULT_DIFFICULTY
        
        if "questions_asked" not in state:
            state["questions_asked"] = []
        
        if "internal_monologue" not in state:
            state["internal_monologue"] = []
        
        if "assessment" not in state:
            state["assessment"] = Assessment()
        
        if "messages" not in state:
            state["messages"] = []
        
        if "observer_recommendation" not in state:
            state["observer_recommendation"] = None
        
        if "current_turn_thoughts" not in state:
            state["current_turn_thoughts"] = []
        
        state["interview_complete"] = False
        state["need_feedback"] = False
        state["current_question"] = None
        state["current_answer"] = None
        state["coordinator_instruction"] = None
        
        self.display.print_agent_action("System", f"Интервью начато для {state['candidate_info'].name}")
        
        return state
    
    def coordinator_decision(self, state: InterviewState) -> InterviewState:
        """Координатор принимает решение о следующем шаге"""
        # Проверяем наличие log_data в состоянии
        if "log_data" not in state:
            print("❌ ОШИБКА: log_data отсутствует в coordinator_decision!")
            # Восстанавливаем log_data
            state["log_data"] = InterviewLogger.init_log_data(state["candidate_info"])
        else:
            print(f"📊 log_data передан в coordinator_decision, записей: {len(state['log_data'].get('turns', []))}")
        
        # Проверяем, не пора ли завершить
        if self.coordinator.should_end_interview(state):
            state["interview_complete"] = True
            self.display.print_agent_action("Coordinator", "Завершаем интервью")
            return state
        
        # Получаем решение координатора
        decision = self.coordinator.decide_next_step(state)
        
        # Сохраняем инструкцию для интервьюера
        state["coordinator_instruction"] = decision.get("instruction_to_interviewer", "")
        
        # Обновляем тему и сложность если нужно
        if decision.get("action") == "change_topic" and decision.get("new_topic"):
            old_topic = state.get("current_topic", "нет темы")
            state["current_topic"] = decision["new_topic"]
            self.display.print_agent_action("Coordinator", f"Меняем тему на '{decision['new_topic']}'")
        
        # Инициализируем current_turn_thoughts для нового хода
        if "current_turn_thoughts" not in state:
            state["current_turn_thoughts"] = []
        else:
            # Начинаем новый ход - очищаем мысли предыдущего хода
            state["current_turn_thoughts"] = []
        
        # Добавляем мысли координатора
        coordinator_thought = f"Решение: {decision.get('action', 'continue')}"
        if decision.get("reasoning"):
            coordinator_thought += f", обоснование: {decision.get('reasoning')[:100]}..."
        
        state["current_turn_thoughts"].append(f"[Coordinator]: {coordinator_thought}")
        
        # Если решение - завершить интервью
        if decision.get("action") == "end_interview":
            state["interview_complete"] = True
            self.display.print_agent_action("Coordinator", "Завершаем интервью по решению LLM")
        
        return state
    
    def generate_question(self, state: InterviewState) -> InterviewState:
        """Генерация вопроса интервьюером"""
        # Проверяем наличие log_data
        if "log_data" not in state:
            print("❌ ОШИБКА: log_data отсутствует в generate_question!")
            state["log_data"] = InterviewLogger.init_log_data(state["candidate_info"])
        
        # Генерируем вопрос
        self.display.print_agent_action("Interviewer", "Генерация вопроса...")
        question = self.interviewer.generate_question(state)
        
        # Добавляем мысли интервьюера
        if "current_turn_thoughts" not in state:
            state["current_turn_thoughts"] = []
        
        interviewer_thought = f"Сгенерирован вопрос по теме '{state.get('current_topic', 'общая')}'"
        if state.get("coordinator_instruction"):
            interviewer_thought += f", инструкция: {state.get('coordinator_instruction')[:100]}..."
        
        state["current_turn_thoughts"].append(f"[Interviewer]: {interviewer_thought}")
        
        # Добавляем в историю сообщений
        messages = state.get("messages", [])
        messages = messages + [Message(
            role="interviewer",
            content=question
        )]
        
        # Сохраняем текущий вопрос
        state["messages"] = messages
        state["current_question"] = question
        state["last_question"] = question
        state["questions_asked"] = state.get("questions_asked", []) + [question]
        state["current_answer"] = ""
        
        return state
    
    def get_user_answer(self, state: InterviewState) -> InterviewState:
        """Получение ответа от пользователя"""
        # Проверяем наличие log_data
        if "log_data" not in state:
            print("❌ ОШИБКА: log_data отсутствует в get_user_answer!")
            state["log_data"] = InterviewLogger.init_log_data(state["candidate_info"])
        
        # Получаем текущий вопрос из состояния
        current_question = state.get("current_question", "")
        
        if not current_question:
            # Используем последний вопрос из истории
            messages = state.get("messages", [])
            for msg in reversed(messages):
                if msg.role == "interviewer":
                    current_question = msg.content
                    break
        
        if current_question:
            self.display.print_question(current_question)
        else:
            print("\n❌ Вопрос не найден в состоянии")
        
        # Получаем ответ от пользователя
        try:
            user_input = input("Ваш ответ: ").strip()
        except EOFError:
            user_input = "стоп"
        
        if user_input.lower() in ["стоп", "stop", "завершить", "конец", "exit", "quit"]:
            self.display.print_agent_action("System", "Интервью завершено по запросу пользователя")
            state["interview_complete"] = True
            state["current_answer"] = user_input
            return state
        
        self.display.print_answer(user_input)
        
        # Добавляем сообщение пользователя
        messages = state.get("messages", [])
        messages = messages + [Message(
            role="user",
            content=user_input
        )]
        
        # Добавляем системные мысли
        if "current_turn_thoughts" not in state:
            state["current_turn_thoughts"] = []
        
        state["current_turn_thoughts"].append(f"[System]: Получен ответ длиной {len(user_input)} символов")
        
        # Сохраняем ответ в состоянии
        state["messages"] = messages
        state["current_answer"] = user_input
        state["last_answer"] = user_input
        
        return state
    
    def analyze_answer(self, state: InterviewState) -> InterviewState:
        """Анализ ответа наблюдателем"""
        # Проверяем наличие log_data
        if "log_data" not in state:
            print("❌ ОШИБКА: log_data отсутствует в analyze_answer!")
            state["log_data"] = InterviewLogger.init_log_data(state["candidate_info"])
        
        # Получаем вопрос и ответ из состояния
        question = state.get("current_question", "")
        answer = state.get("current_answer", "")
        
        if not question or not answer:
            print("❌ Observer: Нет вопроса или ответа для анализа")
            print(f"  Вопрос: {question}")
            print(f"  Ответ: {answer}")
            return state
        
        try:
            # Анализируем ответ
            analysis, updated_assessment = self.observer.analyze_answer(state, question, answer)
            
            # Формируем internal_thoughts
            internal_thoughts = ""
            
            # Добавляем мысли из current_turn_thoughts
            if "current_turn_thoughts" in state:
                for thought in state["current_turn_thoughts"]:
                    internal_thoughts += f"{thought}\n"
            
            # Добавляем мысли наблюдателя
            if analysis.get("reasoning"):
                internal_thoughts += f"[Observer]: {analysis.get('reasoning')}\n"
            
            if analysis.get("technical_score") is not None:
                internal_thoughts += f"[Observer]: Техническая оценка: {analysis.get('technical_score')}/10\n"
            
            if analysis.get("communication_score") is not None:
                internal_thoughts += f"[Observer]: Коммуникационная оценка: {analysis.get('communication_score')}/10\n"
            
            if analysis.get("recommendation_for_next_question"):
                internal_thoughts += f"[Observer]: Рекомендация: {analysis.get('recommendation_for_next_question')}\n"
            
            # Логируем
            print(f"📝 Добавление записи в лог: вопрос длиной {len(question)}, ответ длиной {len(answer)}")
            
            # Добавляем ход в лог
            state["log_data"] = InterviewLogger.add_turn(
                state["log_data"],
                question,
                answer,
                internal_thoughts.strip()
            )
            
            print(f"✅ Запись добавлена. Всего записей: {len(state['log_data']['turns'])}")
            
            # Обновляем состояние
            state["assessment"] = updated_assessment
            state["observer_recommendation"] = analysis.get("recommendation_for_next_question", "")
            
            # Очищаем временные поля для следующего хода
            state["current_turn_thoughts"] = []
            state["coordinator_instruction"] = None
            
        except Exception as e:
            print(f"❌ Observer: Ошибка анализа: {e}")
            import traceback
            traceback.print_exc()
        
        return state
    
    def generate_feedback(self, state: InterviewState) -> InterviewState:
        """Генерация финального фидбэка"""
        print("\n" + "="*60)
        print("📊 ГЕНЕРАЦИЯ ФИНАЛЬНОГО ФИДБЭКА")
        print("="*60)
        
        # Проверяем наличие log_data
        if "log_data" not in state:
            print("⚠️  Нет log_data, создаем новый...")
            state["log_data"] = InterviewLogger.init_log_data(state["candidate_info"])
        
        # Рассчитываем длительность
        from datetime import datetime
        if "start_time" in state["log_data"]:
            start_time = datetime.fromisoformat(state["log_data"]["start_time"])
            duration_minutes = (datetime.now() - start_time).total_seconds() / 60
        else:
            duration_minutes = 10.0
        
        print(f"⏱️  Длительность интервью: {duration_minutes:.1f} минут")
        
        # Генерируем фидбэк
        feedback = self.feedback_gen.generate_feedback(state, duration_minutes)
        
        # Сохраняем фидбэк в лог
        print("📝 Сохраняем фидбэк в лог...")
        state["log_data"] = InterviewLogger.save_final_feedback(state["log_data"], feedback)
        
        # Выводим фидбэк
        print("\n" + "="*60)
        print("🎯 ИТОГОВЫЙ ФИДБЭК")
        print("="*60)
        
        if isinstance(feedback, dict) and "full_text_feedback" in feedback:
            print(feedback["full_text_feedback"])
        elif isinstance(feedback, str):
            print(feedback)
        else:
            print("Фидбэк не в текстовом формате")
        
        # Сохраняем лог в файл
        try:
            scenario_num = state.get("scenario_number", 1)
            filename = f"interview_log_{scenario_num}.json"
            
            log_file = InterviewLogger.save_to_file(
                state["log_data"],
                state["candidate_info"].name,
                filename
            )
            print(f"\n📁 Лог сохранен в: {log_file}")
            
            # Проверяем структуру файла
            with open(log_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                print(f"✅ Проверка сохраненного файла:")
                print(f"   • Участник: {saved_data.get('participant_name')}")
                print(f"   • Записей (turns): {len(saved_data.get('turns', []))}")
                print(f"   • Фидбэк: {'Да' if saved_data.get('final_feedback') else 'Нет'}")
                
        except Exception as e:
            print(f"⚠️  Ошибка сохранения лога: {e}")
            import traceback
            traceback.print_exc()
        
        state["final_feedback"] = feedback
        return state
    
    def end_interview(self, state: InterviewState) -> InterviewState:
        """Завершение интервью"""
        print("\n" + "="*60)
        print("✅ ИНТЕРВЬЮ ЗАВЕРШЕНО")
        print("="*60)
        
        # Статистика
        turns = len(state["log_data"]["turns"]) if "log_data" in state else 0
        
        print(f"\n📈 Статистика:")
        print(f"   • Вопросов задано: {turns}")
        
        return state
    
    def check_should_continue(self, state: InterviewState) -> str:
        """Проверяет, нужно ли продолжать интервью"""
        if state.get("interview_complete", False):
            return "end"
        
        # Проверяем максимальное количество вопросов
        max_questions = settings.MAX_TURNS
        questions_asked = len(state.get("questions_asked", []))
        
        if questions_asked >= max_questions:
            print(f"📊 Достигнуто максимальное количество вопросов ({max_questions})")
            return "end"
        
        return "continue"
    
    def run(self, candidate_info: CandidateInfo, scenario_number: int = 1, config: Dict = None):
        """Запуск интервью"""
        # Инициализируем состояние с log_data сразу
        initial_state: InterviewState = {
            "candidate_info": candidate_info,
            "messages": [],
            "internal_monologue": [],
            "current_topic": "",
            "difficulty_level": settings.DEFAULT_DIFFICULTY,
            "assessment": Assessment(),
            "questions_asked": [],
            "observer_recommendation": None,
            "need_feedback": False,
            "interview_complete": False,
            "current_agent": "coordinator",
            "current_question": None,
            "current_answer": None,
            "coordinator_instruction": None,
            "log_data": InterviewLogger.init_log_data(candidate_info),  # Инициализируем здесь!
            "current_turn_thoughts": [],
            "scenario_number": scenario_number
        }
        
        print(f"\n🎭 Запуск мультиагентной системы интервью... (Сценарий {scenario_number})")
        print(f"📝 Лог инициализирован, записей: {len(initial_state['log_data'].get('turns', []))}")
        
        try:
            thread_id = f"interview_{candidate_info.name}_{candidate_info.position}_{scenario_number}"
            
            final_state = self.app.invoke(
                initial_state,
                config={"configurable": {"thread_id": thread_id}}
            )
            
            print(f"\n✨ Интервью успешно завершено! (Сценарий {scenario_number})")
            
            # Проверяем финальное состояние
            if "log_data" in final_state:
                print(f"📊 Финальный лог содержит {len(final_state['log_data'].get('turns', []))} записей")
            else:
                print("❌ Финальный лог не найден в состоянии!")
            
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Интервью прервано пользователем (Сценарий {scenario_number})")
            # Сохраняем лог при прерывании
            if "log_data" in initial_state and initial_state["log_data"]:
                try:
                    filename = f"interview_interrupted_{scenario_number}.json"
                    InterviewLogger.save_to_file(
                        initial_state["log_data"],
                        candidate_info.name,
                        filename
                    )
                    print(f"📁 Лог прерванного интервью сохранен в {filename}")
                except Exception as e:
                    print(f"⚠️  Ошибка сохранения лога при прерывании: {e}")
        
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()