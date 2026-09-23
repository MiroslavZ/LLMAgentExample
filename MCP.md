# Подключение MCP — день 16

Клиент подключается к серверу, выполняет `initialize`, отправляет
`notifications/initialized` и получает `tools/list`. Если сервер возвращает
`nextCursor`, клиент запрашивает следующие страницы. После загрузки каталог
остаётся в интерфейсе, а сессия и HTTP-клиент закрываются.

Используется официальный Python SDK `mcp==2.2.0`, уже включённый в
`requirements.txt`. В этой версии транспорт принимает `httpx2.AsyncClient`,
возвращает два потока, а Python-модели используют имена `input_schema`,
`next_cursor`, `server_info`. При выводе JSON используются протокольные имена.

## GitHub MCP Server

Подключение использует [официальный удалённый сервер GitHub](https://github.com/github/github-mcp-server)
с endpoint `https://api.githubcopilot.com/mcp/`. Docker и локальный сервер
для этого варианта не нужны. Поддерживается авторизация Personal Access Token
через заголовок `Authorization: Bearer …`.
См. [документацию удалённого сервера](https://github.com/github/github-mcp-server/blob/main/docs/remote-server.md)
и [создание GitHub PAT](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).
Доступный набор инструментов зависит от настроек сервера и разрешений токена.

Добавьте в `.env` в корне проекта:

```dotenv
GITHUB_PERSONAL_ACCESS_TOKEN=ваш_токен
```

Значение, уже заданное в окружении процесса, имеет приоритет над `.env`.
После изменения токена перезапустите веб-приложение. `.env` исключён из Git.

```shell
python -m pip install -r requirements.txt
python -m llm_agent.web
```

В верхней панели откройте **MCP → Добавить MCP**. Форма содержит:

| Поле | Значение для GitHub |
| --- | --- |
| Название | `GitHub` |
| URL MCP-сервера | `https://api.githubcopilot.com/mcp/` |
| Переменная окружения с токеном | `GITHUB_PERSONAL_ACCESS_TOKEN` |

Нажмите **Сохранить MCP**, затем **Get tools**. При успехе отображаются имя и
версия сервера, версия протокола и число инструментов. Раскройте инструмент,
чтобы прочитать описание и JSON Schema входных параметров. Пустой список
показывается как успешный результат с нулевым числом инструментов.

Серверы можно редактировать и удалять. Настройки общие для всех диалогов и
сохраняются в `data/conversations/mcp.sqlite3`; при `--data-dir` файл находится
в выбранном каталоге. Сохраняется только имя переменной с токеном. Сам токен
не передаётся в интерфейс, историю диалога или базу настроек.

## Терминал и Python

Вывести сведения о GitHub-сервере и каталог инструментов в JSON:

```shell
python -m llm_agent.mcp_client
```

Другой endpoint и другое имя переменной:

```shell
python -m llm_agent.mcp_client --url https://example.com/mcp --token-env MY_MCP_TOKEN
```

Пустое значение `--token-env` отключает Bearer-авторизацию; в веб-форме для
этого оставьте соответствующее поле пустым. Допускается HTTPS, а для локальной
разработки — HTTP на `localhost`, `127.0.0.1` или `[::1]`. URL с логином,
паролем, query-параметрами и fragment не принимаются.

Клиент доступен отдельно от NiceGUI и LLM:

```python
import asyncio
from llm_agent.mcp_client import get_tools
from llm_agent.mcp_config import MCPServer

# Переменная GITHUB_PERSONAL_ACCESS_TOKEN должна быть в окружении процесса.
server = MCPServer("github", "GitHub", "https://api.githubcopilot.com/mcp/",
                   "GITHUB_PERSONAL_ACCESS_TOKEN")
result = asyncio.run(get_tools(server))
for tool in result.tools:
    print(tool.name, tool.description, tool.input_schema)
```

`get_tools` асинхронный; общий таймаут по умолчанию — 30 секунд. Повторный
Get tools открывает новую сессию. Закрытие MCP-окна отменяет текущий запрос.
При отсутствии токена, отказе авторизации, ошибке сети или протокола интерфейс
показывает сообщение; CLI завершает работу с ненулевым кодом.

## Границы задания и проверка

В дне 16 реализован просмотр каталога. `tools/call`, передача инструментов
в контекст LLM, автоматический выбор инструментов, OAuth, stdio и старый SSE
транспорт пока не реализованы. Собственный рабочий MCP-сервер относится к дню 17.

```shell
python -m unittest tests.test_mcp_config tests.test_mcp_client tests.test_mcp_web -v
python -m unittest discover -s tests -q
```

Автоматические проверки используют временные настройки и тестовый MCP-сервер;
они не обращаются к GitHub или DeepSeek. Успешную авторизацию именно на GitHub
нужно проверить кнопкой **Get tools** или командой CLI с действующим PAT.
