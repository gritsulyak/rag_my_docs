# RAG по нормативным документам — отчёт о качестве и производительности

## 1. Стенд
- **ОС**: Linux 7.0.0-34-generic
- **Python**: 3.13.9
- **CPU**: x86_64 x16
- **RAM, ГБ**: 27.1
- **Провайдер LLM**: openai
- **LLM**: bonsai
- **Коллекция по умолчанию**: normdoc_e5_small
- **Qdrant**: http://localhost:6333
- **GPU**: нет (CPU)

## 2. Корпус
- **Документов**: 13
- **Чанков**: 907
- **Типы**: form: 1, law: 4, regulation: 1, release_notes: 1, standard: 6
- **Дублей исключено**: 1

## 3. Качество ответов (RAGAS)
Судья — локальная LLM, эмбеддинги — локальные. Faithfulness — опора ответа на контекст, ContextPrecision — доля релевантных чанков, ContextRecall — полнота контекста относительно эталона.

| Конфигурация | LLM | Эмбеддинги | Реранкер | Чанкинг | Faithfulness | ContextPrecision | ContextRecall | Ср. латентность, с |
|---|---|---|---|---|---|---|---|---|
| `baseline` | bonsai | e5_small | нет | semantic | 0.975 | 0.634 | 0.833 | 34.94 |
| `e5_rerank_bonsai` | bonsai | e5_small | да | semantic | 0.976 | 0.667 | 0.833 | 41.28 |
| `bge_rerank_bonsai` | bonsai | bge_m3 | да | semantic | 0.933 | 0.639 | 1.000 | 52.22 |

## 4. Производительность (perf-тесты)
Замеры латентности ретривера и генерации, скорости генерации (ток/с) и длины ответа.

| Конфигурация | LLM | Коллекция | Реранкер | Ретрив p50, с | Ретрив p95, с | Генерация p50, с | Итого p50, с | ток/с | Ответ, симв. |
|---|---|---|---|---|---|---|---|---|---|
| `baseline` | bonsai | normdoc_e5_small | да | 9.372 | 9.876 | 46.927 | 56.385 | 2.57 | 229 |

## 5. Нагрузочное тестирование инференса
HTTP-нагрузка на `/answer` под конкурентностью: RPS, перцентили латентности, ошибки, токены/с. Графики строятся `scripts/load_test.py`.

| Конфигурация | Запросов | Конкурентность | RPS | p50, с | p95, с | p99, с | Ошибок | ток/с |
|---|---|---|---|---|---|---|---|---|
| `local` | 12 | 3 | 0.03 | 78.41 | 154.40 | 156.25 | 0 | 2.39 |

Графики латентности (латентность по запросам / распределение / перцентили):

![Нагрузочный тест local](load_local.png)

## 6. Мониторинг Langfuse (трейсы и latency)
Хост: `file:///home/rtg/github/gritsulyak/rag_my_docs/results/traces_local.jsonl`, трейсов: **12**, средняя латентность: **92.28 с**.

| Trace | Вопрос | Латентность, с | Спаны (retrieve / generate, с) | Чанков |
|---|---|---|---|---|
| `local-1791708496086` | Какие требования к паролю пользователя средства контейнеризации 4 класса защиты  | 52.52 | retrieve: 35.50 / generate: 17.02 | 5 |
| `local-1791708506138` | Сколько классов защиты устанавливается для средств контейнеризации Требованиями  | 62.57 | retrieve: 35.58 / generate: 27.00 | 5 |
| `local-1791708519036` | На какой срок выдаётся сертификат соответствия процессов безопасной разработки п | 75.47 | retrieve: 35.67 / generate: 39.79 | 5 |
| `local-1791708534571` | В какой срок ФСТЭК России рассматривает материалы сертификации, если экспертное  | 38.48 | retrieve: 22.10 / generate: 16.38 | 5 |
| `local-1791708555216` | С какой даты заказчики вправе осуществлять закупки для обеспечения государственн | 49.06 | retrieve: 32.05 / generate: 17.01 | 5 |
| `local-1791708566699` | Какая организация определена оператором единого агрегатора торговли распоряжение | 47.64 | retrieve: 29.62 / generate: 18.02 | 5 |
| `local-1791708615912` | В какой срок получатель средств федерального бюджета формирует сведения о бюджет | 81.32 | retrieve: 27.52 / generate: 53.80 | 5 |
| `local-1791708667525` | С какой даты, согласно письму Федерального казначейства от 27.02.2019, вступают  | 112.29 | retrieve: 31.02 / generate: 81.27 | 5 |
| `local-1791708719208` | Как в версии ЕАТ 2.80.1 переименована вкладка «Доп. соглашения» в карточке закуп | 152.49 | retrieve: 33.87 / generate: 118.62 | 5 |
| `local-1791708772632` | Каким приказом утверждён и введён в действие ГОСТ Р 57100—2025 и с какой даты он | 156.71 | retrieve: 19.61 / generate: 137.10 | 5 |
| `local-1791708816514` | Что такое «описание архитектуры» по ГОСТ Р 57100—2025? | 148.98 | retrieve: 21.75 / generate: 127.23 | 5 |
| `local-1791708849062` | Каким приказом утверждён ГОСТ Р 56939—2024 и взамен какого стандарта он введён? | 129.85 | retrieve: 10.02 / generate: 119.83 | 5 |

## 7. Тесты
Итог последнего прогона: **42 passed in 58.10s**

## 8. Каталог рекомендованных РФ-моделей для локального RAG
| Модель | Организация | Тип | Параметры | Контекст | Лицензия | HuggingFace | Запуск |
|---|---|---|---|---|---|---|---|
| GigaChat3.1-10B-A1.8B | Sber / ai-sage | llm | 10B (активных 1.8B) | 262144 | см. HF | [ai-sage/GigaChat3.1-10B-A1.8B](https://huggingface.co/ai-sage/GigaChat3.1-10B-A1.8B) | llama.cpp / ollama |
| GigaChat-20B-A3B-instruct-v1.5 | Sber / ai-sage | llm | 21B (активных 3B) | 131072 | см. HF | [ai-sage/GigaChat-20B-A3B-instruct-v1.5](https://huggingface.co/ai-sage/GigaChat-20B-A3B-instruct-v1.5) | llama.cpp / ollama |
| GigaChat (облачный API) | Sber | llm | —B | 131072 | проприетарная (API) | — | api |
| Avito A-vibe | Avito (AvitoTech) | llm | —B | 32768 | проприетарная (веса не опубликованы) | — | api / OpenAI-совместимый эндпоинт |
| Meno-Lite-0.1 | Meno AI | llm | 7B | 32768 | см. HF | [Menlo/Meno-Lite-0.1](https://huggingface.co/Menlo/Meno-Lite-0.1) | llama.cpp / ollama |
| T-lite-it-2.1 | T-Bank (t-tech) | llm | 8B | 40960 | см. HF | [t-tech/T-lite-it-2.1](https://huggingface.co/t-tech/T-lite-it-2.1) | llama.cpp / ollama |
| QVikhr-3-8B-Instruction | Vikhrmodels | llm | 8B | 40960 | Apache-2.0 | [Vikhrmodels/QVikhr-3-8B-Instruction](https://huggingface.co/Vikhrmodels/QVikhr-3-8B-Instruction) | llama.cpp / ollama |
| RuadaptQwen3-8B-Hybrid | RefalMachine | llm | 8B | 40960 | Apache-2.0 | [RefalMachine/RuadaptQwen3-8B-Hybrid](https://huggingface.co/RefalMachine/RuadaptQwen3-8B-Hybrid) | llama.cpp / ollama |
| ruadapt_qwen2.5_7B_instruct | RefalMachine | llm | 7B | 32768 | Apache-2.0 | [RefalMachine/ruadapt_qwen2.5_7B_ext_u48_instruct](https://huggingface.co/RefalMachine/ruadapt_qwen2.5_7B_ext_u48_instruct) | llama.cpp / ollama |
| Saiga Llama3 8B | IlyaGusev | llm | 8B | 8192 | см. HF | [IlyaGusev/saiga_llama3_8b_gguf](https://huggingface.co/IlyaGusev/saiga_llama3_8b_gguf) | llama.cpp / ollama |
| Vikhr-YandexGPT-5-Lite-8B-it | Vikhrmodels | llm | 8B | 32768 | см. HF | [Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it](https://huggingface.co/Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it) | llama.cpp / ollama |
| Qwen2.5-7B-Instruct | Alibaba | llm | 7B | 32768 | Apache-2.0 | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | llama.cpp / ollama |
| Ternary Bonsai 2 27B | PrismML | llm | 27B | 65536 | см. HF | — | llama.cpp |
| BAAI/bge-m3 | BAAI | embedding | 0.57B | 8192 | MIT | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | sentence-transformers |
| multilingual-e5-small | intfloat | embedding | 0.12B | 512 | MIT | [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | sentence-transformers |
| Giga-Embeddings-instruct | Sber / ai-sage | embedding | —B | 8192 | см. HF | [ai-sage/Giga-Embeddings-instruct](https://huggingface.co/ai-sage/Giga-Embeddings-instruct) | sentence-transformers (trust_remote_code) |
| USER-bge-m3 | deepvk | embedding | 0.57B | 8194 | MIT | [deepvk/USER-bge-m3](https://huggingface.co/deepvk/USER-bge-m3) | sentence-transformers |
| ru-en-RoSBERTa | ai-forever | embedding | 0.4B | 512 | MIT | [ai-forever/ru-en-RoSBERTa](https://huggingface.co/ai-forever/ru-en-RoSBERTa) | sentence-transformers |
| BAAI/bge-reranker-v2-m3 | BAAI | reranker | 0.57B | 8192 | Apache-2.0 | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | sentence-transformers CrossEncoder |
| Giga-Retrieval-instruct | Sber / ai-sage | reranker | —B | 8192 | см. HF | [ai-sage/Giga-Retrieval-instruct](https://huggingface.co/ai-sage/Giga-Retrieval-instruct) | transformers (trust_remote_code) |

## 9. Инфраструктура и сервинг (Docker)
Приложение и инфраструктура упакованы в `docker-compose.yml`. Профили:

| Профиль | Назначение | Бэкенд llama.cpp |
|---|---|---|
| `ai395` | AMD Ryzen AI MAX+ 395 / Radeon 8060S | Vulkan (`/dev/dri`,`/dev/kfd`) |
| `rtx3060` | NVIDIA RTX 3060 (отдельный бокс), sm_86 | CUDA (nvidia runtime) |
| `rocm` | AMD через ROCm/HIP (gfx1151) | HIP/ROCm |
| `cpu` | Любая машина без GPU | CPU |
| `monitoring` | Self-hosted Langfuse (Postgres, ClickHouse, Redis, MinIO) | — |
| `tools` | Разовые задачи: индексация, RAGAS, load-тесты | — |

Запуск (пример для AI MAX+ 395 с мониторингом):

```bash
docker compose --profile ai395 --profile monitoring up -d --build
```

Активные сервисы: `langfuse-clickhouse`, `langfuse-minio`, `langfuse-minio-init`, `langfuse-postgres`, `langfuse-redis`, `langfuse-web`, `langfuse-worker`, `qdrant`, `app`, `llama-vulkan`.

## 10. Воспроизведение
```bash
# 1) поднять стек (выберите профиль железа)
docker compose --profile ai395 --profile monitoring up -d --build

# 2) корпус и индекс
docker compose --profile tools run --rm app python scripts/prepare_docs.py
docker compose --profile tools run --rm app python scripts/build_index.py --reset \
    --collection normdoc_e5_small

# 3) качество (RAGAS) + производительность по конфигурациям
docker compose --profile tools run --rm app python scripts/compare_configs.py --limit 6

# 4) нагрузочный тест с графиками
docker compose --profile tools run --rm app python scripts/load_test.py \
    --url http://app:8000 --requests 40 --concurrency 4

# 5) экспорт трейсов Langfuse и сборка отчёта
docker compose --profile tools run --rm app python scripts/langfuse_report.py
docker compose --profile tools run --rm app python scripts/report.py
```