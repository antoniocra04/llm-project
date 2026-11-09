"""Инструменты агента. Требование FR-6."""

import re
from collections.abc import Callable

from assistant.graphrag import search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

ERP_NO_DATA = "В ERP нет данных"
_WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]
_PART_ID = {"type": "string", "pattern": r"^P-[0-9]{4}[A-Z]?$"}
_WAREHOUSE = {"type": "string", "enum": _WAREHOUSES}
_TEXT = {"type": "string", "minLength": 1}


def _tool_schema(name: str, description: str, **properties: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS: list[dict] = [
    _tool_schema("search_knowledge", "Найти факты, коды деталей и их замены в графе знаний.", question=_TEXT),
    _tool_schema("summarize_topic", "Получить обзор темы по сводкам базы знаний.", question=_TEXT),
    _tool_schema(
        "check_stock",
        "Проверить остаток детали на указанном складе по данным ERP.",
        part_id=_PART_ID,
        warehouse=_WAREHOUSE,
    ),
    _tool_schema("get_price", "Узнать цену детали в рублях по данным ERP.", part_id=_PART_ID),
    _tool_schema(
        "create_service_order",
        "Создать сервисный заказ с количеством из вопроса пользователя.",
        part_id=_PART_ID,
        warehouse=_WAREHOUSE,
        qty={"type": "integer", "minimum": 1},
        model=_TEXT,
    ),
]
_PARAMETERS = {schema["function"]["name"]: schema["function"]["parameters"]["properties"] for schema in TOOL_SCHEMAS}


def argument_error(name: str, arguments: dict) -> str | None:
    parameters = _PARAMETERS.get(name)
    if parameters is None:
        return "Неизвестный инструмент."
    if arguments.keys() != parameters.keys():
        return f"Ожидаемые аргументы: {', '.join(parameters)}."
    for key, schema in parameters.items():
        value = arguments[key]
        if schema["type"] == "integer":
            if type(value) is not int or value < schema["minimum"]:
                return "Количество должно быть положительным целым числом."
            continue
        if not isinstance(value, str) or not value.strip():
            return f"Аргумент {key} должен быть непустой строкой."
        if "pattern" in schema and not re.fullmatch(schema["pattern"], value):
            return "Укажите номер детали в формате P-1042 или P-1042B."
        if "enum" in schema and value not in schema["enum"]:
            return f"Неизвестный склад. Допустимые склады: {', '.join(schema['enum'])}."
    return None


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """

    def search_knowledge(question: str) -> dict:
        if error := argument_error("search_knowledge", {"question": question}):
            return {"error": error}
        result = search.local_search(question, kb, llm)
        return {"answer": result.text, "facts": result.facts, "sources": result.sources, "refused": result.refused}

    def summarize_topic(question: str) -> dict:
        if error := argument_error("summarize_topic", {"question": question}):
            return {"error": error}
        result = search.global_search(question, kb, llm)
        return {"answer": result.text, "sources": result.sources, "refused": result.refused}

    def call_erp(name: str, method: Callable[..., dict], **arguments) -> dict:
        if error := argument_error(name, arguments):
            return {"error": error}
        try:
            return method(**arguments)
        except ErpNotFound as exc:
            return {"error": f"{ERP_NO_DATA}: {exc}"}
        except ErpRejected as exc:
            return {"error": f"ERP отклонила запрос: {exc}"}

    def check_stock(part_id: str, warehouse: str) -> dict:
        return call_erp("check_stock", erp.get_stock, part_id=part_id, warehouse=warehouse)

    def get_price(part_id: str) -> dict:
        return call_erp("get_price", erp.get_price, part_id=part_id)

    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> dict:
        return call_erp(
            "create_service_order", erp.create_order, part_id=part_id, warehouse=warehouse, qty=qty, model=model
        )

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }
