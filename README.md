# LLMAgentExample

Агент на Python для запросов к DeepSeek API.

- `agent.py` — класс `Agent`: обычный запрос и запрос с мета-промптингом.
- `main.py` — консольная обёртка: аргументы, загрузка токена и вывод через Rich.

Установите зависимости: `pip install -r requirements.txt`. Создайте рядом с
`main.py` файл `.env` с переменной `API_KEY=ваш_токен`. Значение `API_KEY`, уже
заданное в окружении, имеет приоритет.

```shell
python main.py --user "Объясни принцип работы ИИ-агента"
python main.py --user "Объясни принцип работы ИИ-агента" --meta-prompt
python main.py --help
```

Агента можно использовать без консольной обёртки:

```python
from agent import Agent

agent = Agent(token="ваш_токен")
result = agent.request("Объясни SOLID", temperature=0.5)
print(result.content)

meta_result, result = agent.request_with_meta_prompt(
    "Объясни SOLID",
    system="Отвечай кратко и на русском языке",
    max_tokens=1000,
)
print(meta_result.content)
print(result.content)
```

Конструктор принимает только токен. Оба метода принимают параметры конкретного
запроса: `user`, `model`, `system`, `max_tokens`, `temperature`, `stop_sequences`
и `response_format`. Результат `RequestResult` содержит текст `content`, исходный
ответ API `response` со статистикой токенов и время запроса `elapsed` в секундах.

Метод `request_with_meta_prompt()` возвращает пару результатов: сгенерированный
промпт и итоговый ответ. Первый этап всегда использует текстовый формат, второй —
формат и системный промпт, переданные вызывающим кодом. В CLI результаты обоих
этапов выводятся после завершения всего вызова.

Проверки без обращений к API: `python -m unittest -v`.
