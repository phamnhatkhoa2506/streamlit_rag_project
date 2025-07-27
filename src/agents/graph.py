from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from agents.nodes import (
    reponse_to_user,
    execute_tools,
    summarize_conversation
)
from agents.states import RuntimeState
from agents.agents import redis_saver


def decide_next_steps(state: RuntimeState) -> str:
    latest_ai_message = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
    if latest_ai_message and latest_ai_message.tool_calls:
        return "execute_tools"
    
    return "summarize_conversation"


def build_graph():
    graph_builder = StateGraph(RuntimeState)

    graph_builder.add_node("response_to_user", reponse_to_user)
    graph_builder.add_node("execute_tools", execute_tools)
    graph_builder.add_node("summarize_conversation", summarize_conversation)

    graph_builder.add_edge(START, "response_to_user") # or graph_builder.set_entry_point("response_to_user")
    graph_builder.add_conditional_edges(
        "response_to_user",
        decide_next_steps,
        {"execute_tools": "execute_tools", "summarize_conversation": "summarize_conversation"}
    )
    graph_builder.add_edge("execute_tools", "response_to_user")
    graph_builder.add_edge("summarize_conversation", END)

    graph = graph_builder.compile(checkpointer=redis_saver)

    return graph


# graph = build_graph()