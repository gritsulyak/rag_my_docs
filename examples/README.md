# Примеры входных данных и запросов

Эта папка помогает проверить систему «как есть», без подготовки своего корпуса.

## Документы (входные данные)

Примеры нормативных документов лежат в [`../data/raw/sources/`](../data/raw/sources/):

| Документ | Тип | О чём |
|---|---|---|
| Требования ФСТЭК России №118 | PDF | классы защиты средств контейнеризации, парольная политика |
| Приказ ФСТЭК России №240 | PDF | сертификация процессов безопасной разработки ПО |
| Распоряжение Правительства РФ №824-р | RTF→PDF | единый агрегатор торговли (ЕАТ), оператор |
| Письмо Федерального казначейства от 27.02.2019 | PDF (скан, OCR) | бюджетные обязательства по договорам ЕАТ |
| Регламент ЕАТ (редакция №19) | PDF | закупочные процедуры |
| Обновление ЕАТ 2.80.1 | PDF | изменения интерфейса |
| ГОСТ Р 57100—2025 | PDF | описание архитектуры |
| ГОСТ Р 58412—2019 | PDF | разработка безопасного ПО |

Полный список — в `data/raw/manifest.json` после `scripts/prepare_docs.py`.

## Тестовые запросы

- [`../data/qa/golden_qa.jsonl`](../data/qa/golden_qa.jsonl) — 12 пар
  `{"id","question","ground_truth"}` на русском; используются и для RAGAS, и
  для нагрузочных тестов.
- [`requests.http`](requests.http) — те же запросы в виде готовых HTTP-вызовов
  (VS Code REST Client или `curl`).

## Как проверить

```bash
# 1) сервис
uv run python scripts/serve_api.py --port 8000
#    или: docker compose --profile cpu up -d app

# 2) один вопрос
curl -s localhost:8000/answer -H 'Content-Type: application/json' \
  -d '{"question":"Каким приказом утверждён ГОСТ Р 57100—2025?"}' | jq

# 3) автоматическая оценка качества и производительности
uv run python scripts/evaluate_ragas.py --limit 12
uv run python scripts/load_test.py --url http://localhost:8000 --requests 40 --concurrency 4
```

## Ожидаемый формат ответа

```json
{
  "answer": "… ответ строго по контексту, со ссылками [1], [2] …",
  "sources": [{"source": "…", "page": 3, "score": 0.84, "snippet": "…"}],
  "latency_s": 12.3,
  "retrieval_s": 0.4,
  "generation_s": 11.9,
  "usage": {"completion_tokens": 58},
  "dropped_citations": []
}
```

`dropped_citations` — ссылки, которые модель выдумала и которые система
отбросила при граундинге (см. `src/normdoc/prompts.py`).
