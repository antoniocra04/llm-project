"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
from itertools import combinations, pairwise

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase

REFUSED_TEXT = "в базе знаний не найдено"
FACTS_CHAR_LIMIT = 8000
PATH_CUTOFF = 3
NEIGHBOURHOOD_RADIUS = 2

LOCAL_SYSTEM = (
    "Ты ассистент сервисной службы компании «ПолярТех». "
    "Отвечай на вопрос только по пронумерованному списку фактов. "
    "Коды моделей, деталей, ошибок и регламентов пиши точно как в фактах. "
    "Если среди фактов есть цепочка замен REPLACED_BY, в ответе называй последнюю деталь цепочки. "
    "Если ответа в фактах нет, напиши: «в базе знаний не найдено»."
)

MAP_SYSTEM = (
    "Ты ассистент сервисной службы «ПолярТех». "
    "По одной сводке сообщества дай частичный ответ на вопрос и оценку полезности сводки от 0 до 10. "
    "0 означает, что сводка не помогает ответить на вопрос."
)

REDUCE_SYSTEM = (
    "Ты ассистент сервисной службы «ПолярТех». "
    "Объедини частичные ответы в один итоговый ответ на вопрос. "
    "Опирайся только на частичные ответы, не добавляй новых фактов."
)


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    anchors = _resolve_anchors(question, kb)
    if anchors is None or not anchors:
        return _refused()

    if len(anchors) == 1:
        edges = _neighbourhood_edges(kb.graph, anchors[0])
    else:
        edges = _path_edges(kb.graph, anchors)
        parts = {node for edge in edges for node in edge[:2] if kb.graph.nodes[node].get("type") == "Part"}
        edges = _unique_edges([*edges, *_replacement_edges(kb.graph, parts)])

    if not edges:
        return _refused()

    facts = [_format_fact(kb.graph, source, target, data) for source, target, data in edges]
    facts_block = _numbered_facts(facts)
    reply = llm.chat(
        [
            {"role": "system", "content": LOCAL_SYSTEM},
            {"role": "user", "content": f"Факты:\n{facts_block}\n\nВопрос: {question}"},
        ]
    )
    text = reply.content.strip()
    refused = "не найдено" in text.lower()
    return Answer(text, sources=_sources_from_edges(edges), facts=facts, refused=refused)


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    partials: list[str] = []
    for community in kb.communities:
        reply = llm.chat(
            [
                {"role": "system", "content": MAP_SYSTEM},
                {
                    "role": "user",
                    "content": f"Сводка сообщества:\n{community.summary}\n\nВопрос: {question}",
                },
            ],
            json_schema=MAP_SCHEMA,
        )
        parsed = _parse_map_answer(reply.content)
        if parsed is None:
            continue
        answer, score = parsed
        if score > 0:
            partials.append(answer)

    if not partials:
        return _refused()

    numbered = "\n".join(f"{index}. {text}" for index, text in enumerate(partials, start=1))
    reply = llm.chat(
        [
            {"role": "system", "content": REDUCE_SYSTEM},
            {"role": "user", "content": f"Частичные ответы:\n{numbered}\n\nВопрос: {question}"},
        ]
    )
    return Answer(reply.content.strip(), refused=False)


def _refused() -> Answer:
    return Answer(REFUSED_TEXT, refused=True)


def _resolve_anchors(question: str, kb: KnowledgeBase) -> list[str] | None:
    anchors: list[str] = []
    seen: set[str] = set()
    for code in find_codes(question):
        node_id = f"{code_type(code)}:{code}"
        if node_id not in kb.graph:
            return None
        if node_id not in seen:
            anchors.append(node_id)
            seen.add(node_id)
    for entity_type, name in find_names(question, kb.vocabulary):
        node_id = f"{entity_type}:{name}"
        if node_id in kb.graph and node_id not in seen:
            anchors.append(node_id)
            seen.add(node_id)
    return anchors


def _path_edges(graph: nx.DiGraph, anchors: list[str]) -> list[tuple[str, str, dict]]:
    undirected = graph.to_undirected(as_view=True)
    edges: list[tuple[str, str, dict]] = []
    for start, end in combinations(anchors, 2):
        if start not in undirected or end not in undirected:
            continue
        for path in nx.all_simple_paths(undirected, start, end, cutoff=PATH_CUTOFF):
            for left, right in pairwise(path):
                edges.extend(_directed_edges(graph, left, right))
    return _unique_edges(edges)


def _neighbourhood_edges(graph: nx.DiGraph, anchor: str) -> list[tuple[str, str, dict]]:
    undirected = graph.to_undirected(as_view=True)
    if anchor not in undirected:
        return []
    nodes = set(nx.ego_graph(undirected, anchor, radius=NEIGHBOURHOOD_RADIUS))
    edges = [
        (source, target, data) for source, target, data in graph.edges(data=True) if source in nodes and target in nodes
    ]
    return _unique_edges(edges)


def _replacement_edges(graph: nx.DiGraph, parts: set[str]) -> list[tuple[str, str, dict]]:
    edges: list[tuple[str, str, dict]] = []
    for part in parts:
        current = part
        seen: set[str] = set()
        while current not in seen:
            seen.add(current)
            replacements = [
                (current, target, data)
                for _, target, data in graph.out_edges(current, data=True)
                if data.get("type") == "REPLACED_BY"
            ]
            if not replacements:
                break
            edges.extend(replacements)
            current = replacements[0][1]
    return edges


def _directed_edges(graph: nx.DiGraph, left: str, right: str) -> list[tuple[str, str, dict]]:
    edges: list[tuple[str, str, dict]] = []
    if graph.has_edge(left, right):
        edges.append((left, right, graph.edges[left, right]))
    if graph.has_edge(right, left):
        edges.append((right, left, graph.edges[right, left]))
    return edges


def _unique_edges(edges: list[tuple[str, str, dict]]) -> list[tuple[str, str, dict]]:
    unique: list[tuple[str, str, dict]] = []
    seen: set[tuple[str, str, str]] = set()
    for source, target, data in edges:
        key = (source, target, data.get("type", ""))
        if key in seen:
            continue
        seen.add(key)
        unique.append((source, target, data))
    return unique


def _format_fact(graph: nx.DiGraph, source: str, target: str, data: dict) -> str:
    return f"{graph.nodes[source]['name']} -{data['type']}-> {graph.nodes[target]['name']}"


def _numbered_facts(facts: list[str]) -> str:
    lines: list[str] = []
    size = 0
    for index, fact in enumerate(facts, start=1):
        line = f"{index}. {fact}"
        addition = len(line) + (1 if lines else 0)
        if size + addition > FACTS_CHAR_LIMIT:
            break
        lines.append(line)
        size += addition
    return "\n".join(lines)


def _sources_from_edges(edges: list[tuple[str, str, dict]]) -> list[str]:
    sources: list[str] = []
    seen: set[str] = set()
    for _, _, data in edges:
        for chunk_id in data.get("sources", []):
            doc_id = chunk_id.split("#", 1)[0]
            if doc_id not in seen:
                seen.add(doc_id)
                sources.append(doc_id)
    return sources


def _parse_map_answer(content: str) -> tuple[str, int] | None:
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    answer = payload.get("answer")
    score = payload.get("score")
    if not isinstance(answer, str) or not isinstance(score, int):
        return None
    if score < 0 or score > 10:
        return None
    return answer, score
