"""Сообщества графа и их сводки. Требование FR-3."""

import networkx as nx
from networkx.algorithms.community import louvain_communities

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community

LOUVAIN_SEED = 42
MAX_SUMMARY_WORDS = 150


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    undirected = _undirected_weighted(graph)
    if undirected.number_of_nodes() == 0:
        return []
    partitions = louvain_communities(undirected, weight="weight", seed=LOUVAIN_SEED)
    ordered = sorted((sorted(nodes) for nodes in partitions), key=lambda nodes: (len(nodes), nodes))
    return [Community(community_id, nodes) for community_id, nodes in enumerate(ordered)]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    members = set(community.nodes)
    entities = [f"{graph.nodes[node]['type']}: {graph.nodes[node]['name']}" for node in community.nodes]
    relations: list[str] = []
    for source, target, data in graph.edges(data=True):
        if source not in members or target not in members:
            continue
        source_name = graph.nodes[source]["name"]
        target_name = graph.nodes[target]["name"]
        relations.append(f"{source_name} -{data['type']}-> {target_name} (вес {data['weight']})")

    prompt = (
        "Напиши сводку сообщества на русском языке не длиннее 150 слов. "
        "Опиши, какие сущности входят, как они связаны и что встречается чаще. "
        "Числа уже подсчитаны по весам рёбер — используй их как есть, не пересчитывай.\n\n"
        f"Сущности:\n{chr(10).join(entities) or '(нет)'}\n\n"
        f"Связи:\n{chr(10).join(relations) or '(нет)'}"
    )
    reply = llm.chat(
        [
            {
                "role": "system",
                "content": "Ты составляешь краткие сводки сообществ графа знаний сервисной службы «ПолярТех».",
            },
            {"role": "user", "content": prompt},
        ]
    )
    words = reply.content.split()
    return " ".join(words[:MAX_SUMMARY_WORDS])


def _undirected_weighted(graph: nx.DiGraph) -> nx.Graph:
    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes())
    for source, target, data in graph.edges(data=True):
        weight = data.get("weight", 1)
        if undirected.has_edge(source, target):
            undirected[source][target]["weight"] += weight
        else:
            undirected.add_edge(source, target, weight=weight)
    return undirected
