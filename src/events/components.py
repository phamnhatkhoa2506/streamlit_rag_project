import streamlit as st
from events.upload_file import  handle_file_uploaded_files


def _st_upload_documents() -> None:
    st.session_state.uploaded_files = st.file_uploader(
        "Choose files", 
        accept_multiple_files=True,
    )


def _st_import_urls() -> None:
    st.session_state.urls = st.text_area(
        "Enter URLs (one per line)",
        placeholder="https://example.com/doc1.pdf\nhttps://example.com/doc2.pdf",
        height=150,
    )


def st_document_receiving_area() -> None:
    
    st.session_state.upload_choice = st.radio(
        "Choose how to import documents:",
        ("Upload Files", "Import URLs"),
        horizontal=True,
    )

    left_col, right_col = st.sidebar.columns(2)
    with left_col:
        if st.button("Clear all", key="clear_documents"):
            st.session_state["clear_documents_flag"] = True

    st.session_state.title_dir_name = st.text_input("Enter your title")

    # Tại luồng chính, sau khi layout xong
    if st.session_state.get("clear_documents_flag"):
        del st.session_state["clear_documents_flag"]  # Xóa flag để không rerun mãi
        st.rerun()
    
    with right_col:
        st.button(
            "Load files", 
            key="confirm_documents",
            on_click=handle_file_uploaded_files, 
        )
    
    if st.session_state.upload_choice == "Upload Files":
        _st_upload_documents()
    elif st.session_state.upload_choice == "Import URLs":
        _st_import_urls()

    
