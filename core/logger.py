import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from config.settings import settings
from .state import CandidateInfo

class InterviewLogger:
    """Утилита для логирования. Хранит данные в состоянии, а не в себе."""
    
    @staticmethod
    def init_log_data(candidate_info: CandidateInfo) -> Dict[str, Any]:
        """Инициализирует структуру лога в состоянии"""
        return {
            "participant_name": candidate_info.name,
            "turns": [],
            "final_feedback": "",
            "start_time": datetime.now().isoformat()
        }
    
    @staticmethod
    def format_agent_thoughts(thoughts_list: list[str]) -> str:
        """Форматирует мысли агентов согласно ТЗ"""
        formatted = ""
        for thought in thoughts_list:
            if not thought.startswith("["):
                continue
            formatted += f"{thought}\n"
        return formatted.strip()
    
    @staticmethod
    def add_turn(log_data: Dict[str, Any], 
                 agent_visible_message: str, 
                 user_message: str, 
                 internal_thoughts: str) -> Dict[str, Any]:
        """Добавляет ход в лог"""
        turn = {
            "turn_id": len(log_data["turns"]) + 1,
            "agent_visible_message": agent_visible_message,
            "user_message": user_message,
            "internal_thoughts": internal_thoughts
        }
        
        log_data["turns"].append(turn)
        print(f"ЗАПИСЬ В ЛОГ: Turn {turn['turn_id']} добавлен")
        
        print(f"Вопрос: {agent_visible_message[:80]}...")
        print(f"Ответ: {user_message[:80]}...")
        print(f"Мысли агентов: {internal_thoughts[:100]}...")
        
        return log_data
    
    @staticmethod
    def save_final_feedback(log_data: Dict[str, Any], feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Сохраняет финальный фидбэк"""
        if isinstance(feedback, dict) and "full_text_feedback" in feedback:
            log_data["final_feedback"] = feedback["full_text_feedback"]
        elif isinstance(feedback, str):
            log_data["final_feedback"] = feedback
        else:
            log_data["final_feedback"] = str(feedback)
        
        if "end_time" not in log_data:
            log_data["end_time"] = datetime.now().isoformat()
        
        if "start_time" in log_data:
            start = datetime.fromisoformat(log_data["start_time"])
            end = datetime.fromisoformat(log_data["end_time"])
            log_data["duration_minutes"] = round((end - start).total_seconds() / 60, 2)
        else:
            log_data["duration_minutes"] = 0
        
        return log_data
    
    @staticmethod
    def save_to_file(log_data: Dict[str, Any], 
                     candidate_name: str,
                     filename: Optional[str] = None) -> Path:
        """Сохраняет лог в файл"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(c for c in candidate_name if c.isalnum() or c in ' _-')
            filename = f"interview_{safe_name}_{timestamp}.json"
        
        filepath = Path(settings.LOG_DIR) / filename
        
        output_data = {
            "participant_name": log_data["participant_name"],
            "turns": log_data["turns"],
            "final_feedback": log_data["final_feedback"]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Лог сохранен: {filepath}")
        
        InterviewLogger._verify_log(output_data)
        
        return filepath
    
    @staticmethod
    def _verify_log(data: Dict[str, Any]):
        """Проверяет структуру лога"""
        print(f"\n ПРОВЕРКА СТРУКТУРЫ ЛОГА:")
        print(f"Имя участника: {data.get('participant_name')}")
        print(f"Количество turns: {len(data.get('turns', []))}")
        
        if data.get('turns'):
            print(f"   • Пример turn:")
            turn = data['turns'][0]
            print(f" - ID: {turn.get('turn_id')}")
            print(f" - Вопрос: {turn.get('agent_visible_message', '')[:60]}...")
            print(f" - Ответ: {turn.get('user_message', '')[:60]}...")
            print(f" - Мысли: {turn.get('internal_thoughts', '')[:60]}...")
        else:
            print(f"ВНИМАНИЕ: turns пустые!")