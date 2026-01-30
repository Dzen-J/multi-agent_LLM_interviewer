import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import os
import warnings

# Отключаем предупреждения HF Hub
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

from pathlib import Path
from config.settings import settings
import json

class KnowledgeBase:
    """RAG система для проверки технических знаний"""
    
    def __init__(self, collection_name: str = "tech_interview_kb"):
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"Загружена существующая коллекция: {collection_name}")
        except:
            self.collection = self.client.create_collection(collection_name)
            print(f"Создана новая коллекция: {collection_name}")
            self._load_default_knowledge()
    
    def _load_default_knowledge(self):
        """Загружаем базовые знания если коллекция пуста"""
        default_knowledge = [
            {
                "text": "Python - интерпретируемый язык программирования высокого уровня с динамической типизацией.",
                "metadata": {"topic": "python", "subtopic": "basics", "difficulty": 1}
            },
            {
                "text": "Список (list) в Python - изменяемая последовательность элементов. Сложность доступа по индексу O(1).",
                "metadata": {"topic": "python", "subtopic": "data_structures", "difficulty": 1}
            },
            {
                "text": "Словарь (dict) - структура данных ключ-значение. В среднем O(1) для операций поиска, вставки, удаления.",
                "metadata": {"topic": "python", "subtopic": "data_structures", "difficulty": 1}
            },
            {
                "text": "Декоратор в Python - функция, которая принимает другую функцию и расширяет её функциональность, не изменяя её код.",
                "metadata": {"topic": "python", "subtopic": "advanced", "difficulty": 3}
            },
            {
                "text": "GIL (Global Interpreter Lock) в CPython позволяет выполнять только один поток Python одновременно.",
                "metadata": {"topic": "python", "subtopic": "concurrency", "difficulty": 4}
            },
            {
                "text": "SQL JOIN объединяет строки из двух или более таблиц на основе связанного столбца между ними. Типы: INNER, LEFT, RIGHT, FULL.",
                "metadata": {"topic": "databases", "subtopic": "sql", "difficulty": 2}
            },
            {
                "text": "Индекс в базе данных ускоряет поиск, но замедляет вставку и обновление. Используется B-дерево или hash индексы.",
                "metadata": {"topic": "databases", "subtopic": "performance", "difficulty": 3}
            },
            {
                "text": "REST API использует HTTP методы: GET (получить), POST (создать), PUT (обновить), DELETE (удалить).",
                "metadata": {"topic": "web", "subtopic": "api", "difficulty": 2}
            },
            {
                "text": "Docker контейнер - изолированная среда для запуска приложений. Образ - шаблон для создания контейнеров.",
                "metadata": {"topic": "devops", "subtopic": "containers", "difficulty": 2}
            },
        ]
        
        documents = [item["text"] for item in default_knowledge]
        metadatas = [item["metadata"] for item in default_knowledge]
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        embeddings = self.embedding_model.encode(documents).tolist()
        
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Загружено {len(documents)} документов в базу знаний")


    def verify_technical_answer(self, question: str, answer: str, topic: str = None) -> Dict:
        """Проверяет технический ответ через RAG"""
        
        try:
            # Если RAG отключен или нет базы знаний
            if not self.documents or len(self.documents) == 0:
                return {
                    "is_correct": None,
                    "confidence": 0.5,
                    "correct_info": "База знаний пуста",
                    "suggested_topics": []
                }
            
            # Получаем релевантные документы
            query = f"Вопрос: {question}. Ответ: {answer}"
            
            # Безопасное получение документов
            relevant_docs = []
            try:
                relevant_docs = self.get_relevant_documents(query, limit=3)
            except Exception as e:
                print(f"Ошибка при поиске документов: {e}")
            
            if not relevant_docs:
                return {
                    "is_correct": None,
                    "confidence": 0.3,
                    "correct_info": "Не найдено релевантной информации в базе знаний",
                    "suggested_topics": []
                }
            
            # Анализируем соответствие
            correct_answer = relevant_docs[0]["content"][:500] if relevant_docs else "Нет данных"
            
            # Простая проверка на соответствие (можно улучшить)
            score = 0
            if answer and len(answer) > 10:
                # Проверяем ключевые слова
                keywords = ["список", "list", "кортеж", "tuple", "изменяемый", "immutable", "изменять", "мутабельный"]
                found_keywords = sum(1 for kw in keywords if kw.lower() in answer.lower())
                score = min(1.0, found_keywords / 4)
            
            return {
                "is_correct": score > 0.5,
                "confidence": score,
                "correct_info": f"Согласно базе знаний: {correct_answer[:200]}...",
                "suggested_topics": [topic] if topic else []
            }
            
        except Exception as e:
            print(f"Ошибка в RAG системе: {e}")
            return {
                "is_correct": None,
                "confidence": 0.3,
                "correct_info": f"Ошибка при проверке: {str(e)[:100]}",
                "suggested_topics": []
            }
    
    def add_custom_knowledge(self, documents: List[str], metadatas: List[Dict]):
        """Добавить кастомные знания в базу"""
        embeddings = self.embedding_model.encode(documents).tolist()
        start_id = len(self.collection.get()["ids"]) if self.collection.get()["ids"] else 0
        ids = [f"custom_{start_id + i}" for i in range(len(documents))]
        
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Добавлено {len(documents)} кастомных документов")