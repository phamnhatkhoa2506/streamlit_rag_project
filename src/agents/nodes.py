from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, RemoveMessage
from langchain_core.runnables.config import RunnableConfig
from agents.agents import travel_agent
from agents.prompts import SUMMARY_SYSTEM_PROMPT
from agents.states import RuntimeState
from llms.summerizer_model import summarizer
from utils.constants import MESSAGE_SUMMARIZATION_THRESHOLD


def reponse_to_user(state: RuntimeState, config: RunnableConfig) -> RuntimeState:
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not human_messages:
        return state
    
    try:
        result = travel_agent.invoke(
            {"messages": human_messages},
            config=config
        )

        ai_message = result["messages"][-1] 
        state["messages"].append(ai_message)
    except Exception as e:

        ai_message = AIMessage(content="I'm sorry, I encountered an error processing your request.")
        state["messages"].append(ai_message)

    return state


def execute_tools(state: RuntimeState, config: RunnableConfig) -> RuntimeState:
    messages = state["messages"]
    latest_ai_message = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage) and m.tool_calls),
        None
    )

    if not latest_ai_message:
        return state
    
    tool_messages = []
    for tool_call in latest_ai_message.tool_calls:
        tool_name = tool_call["name"]
        tool_id = tool_call["id"]
        tool_args = tool_call["args"]

        tool = next((tc for tc in latest_ai_message.tool_calls if tc["name"] == tool_name), None)

        if not tool:
            continue
        
        try:
            result = tool.invoke(tool_args, config=config)
            tool_message = ToolMessage(
                content=str(result),
                name=tool_name,
                tool_call_id=tool_id
            )

            tool_messages.append(tool_message)
        except Exception as e:
            error_message = ToolMessage(
                content=f"Error executing tool '{tool_name}': {str(e)}",
                name=tool_name,
                tool_call_id=tool_id
            )

            tool_messages.append(error_message)

    messages.extend(tool_messages)
    state["messages"] = messages

    return state


def summarize_conversation(state: RuntimeState, config: RunnableConfig) -> RuntimeState:
    messages = state["messages"]
    current_message_count = len(messages)
    if current_message_count < MESSAGE_SUMMARIZATION_THRESHOLD:
        return state
    
    message_content = "\n".join(
        [
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in messages
        ]
    )

    summary_messages = [
        SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
        HumanMessage(content=message_content)
    ]

    summary_response = summarizer.invoke(summary_messages, config=config)

    summary_message = SystemMessage(
        content=f"""
        Summary of the conversation so far:

        {summary_response.content}

        Please continue the conversation based on this summary and the recent messages.
        """
    )

    remove_messages = [RemoveMessage(id=msg.id) for msg in messages if msg.id]

    state["messages"] = [
        *remove_messages,
        summary_message,
        state["messages"][-1]
    ]

    return state.copy()