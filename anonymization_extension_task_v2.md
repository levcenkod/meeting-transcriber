# ТЗ: Добавление локальной анонимизации перед онлайн LLM

## Контекст

Текущий pipeline уже реализован:

```text
WhisperX + diarization
↓
*_speakers.txt
↓
smart chunking
↓
structured extraction
↓
merge + deduplication
↓
evidence check
↓
final summary
↓
summary.md + actions.json + decisions.json
```

Structured extraction и summary сейчас выполняются через онлайн LLM API.

Основная проблема: transcript и extracted data могут содержать чувствительные данные:

- имена сотрудников;
- названия клиентов;
- email;
- телефоны;
- домены;
- project names;
- transaction IDs;
- страны;
- внутренние обсуждения;
- коммерческие детали;
- обсуждение поставщиков, клиентов и регионов продаж.

Нужно добавить слой локальной анонимизации перед отправкой данных в онлайн LLM.

---

# Главная идея

Онлайн LLM НЕ должна видеть реальные сущности.

Правильный flow:

```text
raw transcript
↓
local anonymization
↓
anonymized transcript
↓
online LLM
↓
structured extraction
↓
local de-anonymization
↓
final artifacts
```

---

# Основной принцип

Анонимизация должна происходить локально.

Mapping между реальными сущностями и токенами должен храниться только локально и никогда не отправляться наружу.

---

# Новый flow

```text
WhisperX
↓
*_speakers.txt
↓
speaker_map.json
↓
smart chunking
↓
LOCAL anonymization
↓
anonymized chunks
↓
online LLM structured extraction
↓
LLM returns anonymized JSON
↓
LOCAL de-anonymization
↓
merge + deduplication
↓
evidence check
↓
final summary
↓
summary.md
actions.json
decisions.json
```

---

# Что нужно анонимизировать

Минимальный набор:

- PERSON
- COMPANY
- PROJECT
- EMAIL
- PHONE
- DOMAIN
- URL
- COUNTRY
- TRANSACTION_ID

---

# Пример

## До анонимизации

```text
Георгий, проверь payout callback на conceptpay.org.
Клиент из Германии жалуется на задержку выплат.
```

---

## После анонимизации

```text
PERSON_001, проверь PAYMENT_PROCESS_001 на DOMAIN_001.
CLIENT_001 из COUNTRY_001 жалуется на задержку выплат.
```

---

# Главная цель

Онлайн модель должна понимать структуру разговора и контекст, но не видеть реальные идентификаторы и коммерчески чувствительные сущности.

---

# Важное ограничение

Анонимизация НЕ скрывает смысл.

Пример:

```text
"клиент угрожает судом"
"задержка зарплат"
"прод упал"
```

Даже без имён это остаётся чувствительным контекстом.

Цель этой системы — снизить риск утечки PII и internal identifiers, а не сделать transcript полностью безопасным.

---

# Формат anonymization map

Нужно создать:

```text
meeting_anonymization_map.json
```

---

# Пример

```json
{
  "PERSON_001": "Георгий",
  "PERSON_002": "Алексей",
  "COMPANY_001": "ConceptPay",
  "DOMAIN_001": "conceptpay.org",
  "COUNTRY_001": "Germany"
}
```

---

# Важные правила

## 1. Map хранится только локально

Никогда:
- не отправлять в LLM;
- не логировать в remote services;
- не класть в prompts.

---

## 2. Анонимизация должна быть deterministic

Если:

```text
Георгий → PERSON_001
```

то во всём transcript это должно оставаться:

```text
PERSON_001
```

---

## 3. Reverse mapping должен быть возможен

После получения ответа от LLM система должна уметь:

```text
PERSON_001 → Георгий
```

---

# Этап anonymization

Нужно создать:

```text
scripts/anonymize.py
```

---

# Вход

```text
*_speakers.txt
```

---

# Выход

```text
*_anonymized.txt
meeting_anonymization_map.json
```

---

# Что делает anonymize.py

1. Загружает transcript.
2. Находит чувствительные сущности.
3. Создаёт deterministic tokens.
4. Делает replace.
5. Сохраняет anonymization map.

---

# Деанонимизация

После получения structured extraction от LLM нужно сделать de-anonymization.

---

# Нужно создать

```text
scripts/deanonymize.py
```

---

# Вход

```text
anonymized JSON response
meeting_anonymization_map.json
```

---

# Выход

```text
real entities restored
```

---

# Пример

## LLM response

```json
{
  "action_items": [
    {
      "task": "Проверить PAYMENT_PROCESS_001",
      "owner": "PERSON_001"
    }
  ]
}
```

---

## После de-anonymization

```json
{
  "action_items": [
    {
      "task": "Проверить payout callback",
      "owner": "Георгий"
    }
  ]
}
```

---

# Что НЕ нужно анонимизировать

Не нужно заменять:
- PostgreSQL
- Docker
- Kubernetes
- Redis
- HTTP
- generic business words
- common technologies

Иначе transcript потеряет смысл.

---

# Где встроить anonymization

## Было

```text
chunk
↓
LLM extraction
```

---

## Должно стать

```text
chunk
↓
anonymize
↓
LLM extraction
↓
de-anonymize
```

---

# Новый flow в summarize.py

```text
main()
↓
read *_speakers.txt
↓
speaker mapping
↓
parse speaker blocks
↓
build chunks
↓
for each chunk:
    anonymize chunk
    save anonymization map
    send anonymized chunk to LLM
    validate JSON
    de-anonymize response
    save intermediate result
↓
merge chunk results
↓
deduplicate
↓
evidence check
↓
generate final summary
```

---

# Environment settings

```env
ENABLE_ANONYMIZATION=true

ANONYMIZE_PERSONS=true
ANONYMIZE_COMPANIES=true
ANONYMIZE_DOMAINS=true
ANONYMIZE_EMAILS=true
ANONYMIZE_PHONES=true
ANONYMIZE_URLS=true
ANONYMIZE_COUNTRIES=true
ANONYMIZE_TRANSACTION_IDS=true
```

---

# Важное замечание

Анонимизация НЕ делает transcript полностью безопасным.

Она:
- снижает риск утечки PII;
- скрывает internal identifiers;
- скрывает реальные имена/домены/клиентов/страны;
- но не скрывает сам смысл обсуждения.

---

# Рекомендуемая стратегия использования

```text
local anonymization
↓
online LLM
```

---

# Критерии успеха

Система считается реализованной, если:

1. Transcript анонимизируется локально.
2. Online LLM никогда не получает реальные имена/домены/email.
3. Mapping хранится только локально.
4. Анонимизация deterministic.
5. Возможна полная de-anonymization.
6. Final artifacts содержат реальные сущности после de-anonymization.
7. Structured extraction продолжает нормально работать после anonymization.
