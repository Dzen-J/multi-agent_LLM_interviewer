import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class Settings:
    """Класс настроек приложения"""
    
    def __init__(self):
        # MistralAI
        self.MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
        self.MISTRAL_MODEL: str = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
        
        # RAG
        self.RAG_ENABLED: bool = os.getenv("RAG_ENABLED", "True").lower() == "true"
        self.EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        
        # Интервью
        self.MAX_TURNS: int = int(os.getenv("MAX_TURNS", "10"))
        self.DEFAULT_DIFFICULTY: int = int(os.getenv("DEFAULT_DIFFICULTY", "2"))
        self.MIN_CONFIDENCE_SCORE: float = float(os.getenv("MIN_CONFIDENCE_SCORE", "0.7"))
        
        # Логирование
        self.LOG_DIR: str = os.getenv("LOG_DIR", "./interview_logs")
        self.LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json")

        self._create_directories()
        self._print_settings()
        
    def _create_directories(self):
        """Создает необходимые директории"""
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.CHROMA_PERSIST_DIR, exist_ok=True)
        
    def _print_settings(self):
        """Выводит настройки"""
        print(f"\n{'='*60}")
        print("🔧 НАСТРОЙКИ СИСТЕМЫ")
        print(f"{'='*60}")
        print(f"API ключ: {'✓ установлен' if self.MISTRAL_API_KEY else '✗ отсутствует'}")
        print(f"Модель: {self.MISTRAL_MODEL}")
        print(f"Макс. вопросов: {self.MAX_TURNS}")
        print(f"Сложность по умолчанию: {self.DEFAULT_DIFFICULTY}")
        print(f"RAG: {'включен' if self.RAG_ENABLED else 'выключен'}")
        print(f"Директория логов: {self.LOG_DIR}")
        print(f"{'='*60}")
    
    def validate(self):
        """Проверяет настройки"""
        if not self.MISTRAL_API_KEY:
            print("ВНИМАНИЕ: MISTRAL_API_KEY не установлен!")
            print("Система будет использовать демо-режим с ограниченной функциональностью")
            return False
        return True
    
settings = Settings()

MISTRAL_API_KEY = settings.MISTRAL_API_KEY
MISTRAL_MODEL = settings.MISTRAL_MODEL
DEFAULT_DIFFICULTY = settings.DEFAULT_DIFFICULTY
MAX_TURNS = settings.MAX_TURNS
RAG_ENABLED = settings.RAG_ENABLED
EMBEDDING_MODEL = settings.EMBEDDING_MODEL
CHROMA_PERSIST_DIR = settings.CHROMA_PERSIST_DIR
LOG_DIR = settings.LOG_DIR
LOG_FORMAT = settings.LOG_FORMAT
MIN_CONFIDENCE_SCORE = settings.MIN_CONFIDENCE_SCORE
HF_TOKEN = os.getenv("HF_TOKEN", None)
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR", "./models")