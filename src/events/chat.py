import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agents.states import RuntimeState


def handle_bot_chat(question: str) -> None:
    with st.spinner():
        st.session_state.chat_state["messages"].append(HumanMessage(content=question))

        for result in st.session_state.graph.stream(st.session_state.chat_state, config=st.session_state.chat_config, stream_mode="values"):
            st.session_state.chat_state = RuntimeState(**result)

        ai_messages = [m for m in st.session_state.chat_state["messages"] if isinstance(m, AIMessage)]
        if ai_messages:
            message = ai_messages[-1].content
        else:
            message = "I'm sorry, I couldn't process your request properly."
            
            st.session_state.chat_state["messages"].append(AIMessage(content=message))

        with st.chat_message(name="ai"):
            st.write(message)
        st.session_state.messages.append({"role": "ai", "content": message})


def handle_bot_chat(question: str) -> None:
    with st.spinner(): 
        answer =  st.session_state.graph.invoke(question)

        with st.chat_message(name="ai"):
            st.write(answer)
        st.session_state.messages.append({"role": "ai", "content": answer})