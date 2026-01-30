# main.py
#!/usr/bin/env python3
"""
Multi-Agent Interview Coach System
Система технического интервью с несколькими AI агентами
"""

import os
import sys
from pathlib import Path
import warnings

# Отключаем предупреждения HF Hub
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from core.state import CandidateInfo
from core.graph import InterviewWorkflow
from config.settings import settings
import argparse

def collect_candidate_info() -> CandidateInfo:
    """Сбор информации о кандидате"""
    print("\n" + "="*60)
    print("ВВЕДИТЕ ИНФОРМАЦИЮ О КАНДИДАТЕ")
    print("="*60)
    
    name = input("Имя кандидата (или Enter для анонимного): ").strip() or "Анонимный кандидат"
    position = input("Позиция (например, Python Developer): ").strip() or "Python Developer"
    
    print("\nУровень (grade):")
    print("  1. Junior")
    print("  2. Middle")
    print("  3. Senior")
    
    grade_choice = input("Выберите уровень (1-3): ").strip()
    grade_map = {"1": "Junior", "2": "Middle", "3": "Senior"}
    grade = grade_map.get(grade_choice, "Middle")
    
    try:
        experience = float(input("Опыт работы (лет): ").strip() or "2.0")
    except ValueError:
        experience = 2.0
    
    technologies_input = input("Технологии (через запятую): ").strip()
    technologies = [tech.strip() for tech in technologies_input.split(",") if tech.strip()]
    
    if not technologies:
        technologies = ["Python", "SQL", "Git"]
    
    return CandidateInfo(
        name=name,
        position=position,
        grade=grade,
        experience_years=experience,
        technologies=technologies
    )

def check_environment():
    """Проверка настроек окружения"""
    print("\n" + "="*60)
    print("ПРОВЕРКА НАСТРОЕК")
    print("="*60)
    
    # Выводим информацию о настройках
    print(settings)
    
    if not settings.MISTRAL_API_KEY:
        print("\n MISTRAL_API_KEY не найден в .env файле!")
        print("Добавьте в файл .env:")
        print("MISTRAL_API_KEY=ваш_ключ_здесь")
        print("\nПолучите API ключ на: https://console.mistral.ai/api-keys/")
        
        # Пробуем найти ключ в переменных окружения
        api_key_from_env = os.getenv("MISTRAL_API_KEY")
        if api_key_from_env:
            print(f"\n Найден ключ в переменных окружения, обновляю настройки...")
            settings.MISTRAL_API_KEY = api_key_from_env
            print(" Ключ загружен из переменных окружения")
            return True
        
        use_demo = input("\nПродолжить в демо-режиме (с ограниченной функциональностью)? (y/n): ").lower()
        if use_demo != 'y':
            return False
        
        print(" Продолжаем в демо-режиме...")
        return True
    
    print(" Все настройки корректны")
    return True

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Multi-Agent Interview Coach System')
    parser.add_argument('--demo', action='store_true', help='Демо-режим (без реального AI)')
    args = parser.parse_args()
    
    print("Система технического интервью с несколькими AI агентами")
    print("\nАгенты:")
    print("  • Coordinator - управляет процессом")
    print("  • Interviewer - задает вопросы")
    print("  • Observer - анализирует ответы")
    print("  • Feedback Generator - дает финальный фидбэк")
    
    if not check_environment():
        print("\n Не удалось инициализировать настройки")
        print("Создайте файл .env с вашим API ключом")
        return
    
    if args.demo:
        print("\n  Включен демо-режим - AI функции будут ограничены")
    
    # Собираю информацию о кандидате
    candidate_info = collect_candidate_info()
    
    print("\n" + "="*60)
    print("СВОДКА НАСТРОЕК")
    print("="*60)
    print(f"Кандидат: {candidate_info.name}")
    print(f"Позиция: {candidate_info.position}")
    print(f"Уровень: {candidate_info.grade}")
    print(f"Опыт: {candidate_info.experience_years} лет")
    print(f"Технологии: {', '.join(candidate_info.technologies)}")
    print(f"Режим: {'Демо' if args.demo or not settings.MISTRAL_API_KEY else 'Полный'}")
    print("="*60)
    
    confirm = input("\nНачать интервью? (y/n): ").lower()
    if confirm != 'y':
        print("\n Интервью отменено")
        return
    
    # Запускаем workflow
    workflow = InterviewWorkflow()
    
    try:
        workflow.run(candidate_info)
    except KeyboardInterrupt:
        print("\n\n Интервью прервано. До свидания!")
    except Exception as e:
        print(f"\n Критическая ошибка во время интервью: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()