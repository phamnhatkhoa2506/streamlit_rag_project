import streamlit as st
from utils.download import download, save_documents
from rag.chain import build_rag_chain


def _handle_upload_documents() -> None:
    if not st.session_state.uploaded_files:
        st.sidebar.warning("No documents to upload")
        return
    
    save_documents(
        dir_name=st.session_state.title_dir_name, 
        files=st.session_state.uploaded_files
    )

    st.success("Upload documents successfully!")


def _handle_retrieve_urls() -> None:
    if not st.session_state.urls:
        st.sidebar.warning("No urls to access")
        return

    urls = st.session_state.urls.strip().splitlines()
    links = [
        {
            "url": url,
            "title": f"file{i}"
        }
        for i, url in enumerate(urls)
    ]

    download(dir_name=st.session_state.title_dir_name, links=links)

    st.success("Access urls successfully!")
             
                
def handle_file_uploaded_files() -> None:
    if not st.session_state.title_dir_name:
        st.sidebar.warning("No title provided")
        return 
    
    try:
        with st.sidebar:
            with st.spinner("Processing..."):
                if st.session_state.upload_choice == "Upload Files":
                    _handle_upload_documents()
                else:
                    _handle_retrieve_urls()

            st.session_state.llm_model = build_rag_chain(
                st.session_state.llm_model,
                data_dir=f"data_source/{st.session_state.title_dir_name}",
                data_type="pdf"
            )

            st.success("Build RAG model succesfully")

    
    except Exception as e:
        st.error("Error when processing files: " + str(e))
    finally:
        pass
        