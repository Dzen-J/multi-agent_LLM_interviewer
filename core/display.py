class DisplayManager:
    """Менеджер для красивого вывода в консоль"""
    
    @staticmethod
    def print_section(title: str):
        """Выводит разделитель с заголовком"""
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")
    
    @staticmethod
    def print_agent_action(agent: str, action: str, details: str = ""):
        """Выводит действие агента"""
        colors = {
            "Coordinator": "",
            "Interviewer": "", 
            "Observer": "",
            "System": "",
            "Feedback": ""
        }
        emoji = colors.get(agent, "🔹")
        print(f"\n{emoji} {agent}: {action}")
        if details:
            print(f"   {details[:100]}...")
    
    @staticmethod
    def print_question(question: str):
        """Выводит вопрос"""
        print(f"\n{'─'*40}")
        print(f"ВОПРОС:")
        print(f"{question}")
        print(f"{'─'*40}")
    
    @staticmethod
    def print_answer(answer: str):
        """Выводит ответ пользователя"""
        print(f"ОТВЕТ: {answer[:100]}...")
    
    @staticmethod
    def print_analysis(result: dict):
        """Выводит результат анализа"""
        score = result.get('technical_score', 0)
        if score >= 8:
            rating = "Отлично"
        elif score >= 6:
            rating = "Хорошо"
        elif score >= 4:
            rating = "Удовлетворительно"
        else:
            rating = "Требует улучшения"
        
        print(f"АНАЛИЗ: {rating} ({score}/10)")
        if result.get('recommendation_for_next_question'):
            print(f"   Рекомендация: {result['recommendation_for_next_question']}")