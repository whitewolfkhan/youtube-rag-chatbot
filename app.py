"""
YouTube RAG Chatbot - Streamlit Application
"""

import streamlit as st
from config import APP_TITLE, APP_ICON, TOP_K_RESULTS
from transcript import transcript_extractor
from vectorstore import vector_store
from rag_chain import get_rag_chain
from utils import extract_video_id, get_youtube_thumbnail, format_timestamp


# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initialize session state variables."""
    if 'video_loaded' not in st.session_state:
        st.session_state.video_loaded = False
    if 'video_url' not in st.session_state:
        st.session_state.video_url = ""
    if 'video_id' not in st.session_state:
        st.session_state.video_id = ""
    if 'transcript_data' not in st.session_state:
        st.session_state.transcript_data = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processing' not in st.session_state:
        st.session_state.processing = False


def display_header():
    """Display the app header."""
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.markdown("""
    **Ask questions about any YouTube video!** 
    Just paste a YouTube URL, and I'll extract the transcript and answer your questions.
    """)
    st.divider()


def display_video_input():
    """Display video URL input section."""
    st.subheader("📺 Load a YouTube Video")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed"
        )
    
    with col2:
        load_button = st.button("Load Video", type="primary", use_container_width=True)
    
    return url, load_button


def display_video_info():
    """Display information about the loaded video."""
    if st.session_state.video_loaded:
        with st.container():
            st.success("✅ Video loaded successfully!")
            
            col1, col2, col3 = st.columns([1, 2, 3])
            
            with col1:
                # Display thumbnail
                thumbnail_url = get_youtube_thumbnail(st.session_state.video_id)
                st.image(thumbnail_url, use_container_width=True)
            
            with col2:
                st.markdown(f"**Video ID:** `{st.session_state.video_id}`")
                
                if st.session_state.transcript_data:
                    duration = st.session_state.transcript_data.get('total_duration', 0)
                    language = st.session_state.transcript_data.get('language', 'Unknown')
                    total_chunks = vector_store.video_metadata.get('total_chunks', 0)
                    
                    st.markdown(f"**Duration:** {format_timestamp(duration)}")
                    st.markdown(f"**Language:** {language}")
                    st.markdown(f"**Chunks:** {total_chunks}")
            
            with col3:
                if st.button("🗑️ Clear Video", type="secondary"):
                    clear_video()
                    st.rerun()


def clear_video():
    """Clear the loaded video and reset state."""
    st.session_state.video_loaded = False
    st.session_state.video_url = ""
    st.session_state.video_id = ""
    st.session_state.transcript_data = None
    st.session_state.chat_history = []
    vector_store.clear()
    
    # Clear RAG chain history
    try:
        rag = get_rag_chain()
        rag.clear_history()
    except:
        pass


def process_video(url: str) -> bool:
    """Process a YouTube video URL."""
    # Validate URL
    video_id = extract_video_id(url)
    if not video_id:
        st.error("❌ Invalid YouTube URL. Please enter a valid URL.")
        return False
    
    # Show processing status
    with st.status("Processing video...", expanded=True) as status:
        # Step 1: Extract transcript
        st.write("📥 Extracting transcript...")
        transcript_data = transcript_extractor.extract(url)
        
        if not transcript_data.get('success'):
            st.error(f"❌ {transcript_data.get('error', 'Failed to extract transcript')}")
            return False
        
        # Step 2: Create embeddings
        st.write("🔢 Creating embeddings...")
        
        # Step 3: Store in vector database
        st.write("💾 Storing in vector database...")
        success = vector_store.create_from_transcript(transcript_data, url)
        
        if not success:
            st.error("❌ Failed to create vector store.")
            return False
        
        status.update(label="✅ Video processed successfully!", state="complete")
    
    # Update session state
    st.session_state.video_loaded = True
    st.session_state.video_url = url
    st.session_state.video_id = video_id
    st.session_state.transcript_data = transcript_data
    
    return True


def display_chat():
    """Display the chat interface."""
    st.subheader("💬 Chat")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.markdown(f"**[{source['timestamp']}]** {source['text']}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the video..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    rag = get_rag_chain()
                    result = rag.ask(prompt)
                    
                    if result.get('success'):
                        answer = result['answer']
                        sources = result.get('sources', [])
                        
                        st.markdown(answer)
                        
                        if sources:
                            with st.expander("📚 Sources"):
                                for source in sources:
                                    timestamp = source['timestamp']
                                    start_seconds = source['start_seconds']
                                    youtube_link = f"https://www.youtube.com/watch?v={st.session_state.video_id}&t={int(start_seconds)}s"
                                    st.markdown(f"**[{timestamp}]({youtube_link})** {source['text']}")
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                    else:
                        error_msg = result.get('error', 'An error occurred')
                        st.error(f"❌ {error_msg}")
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"Error: {error_msg}"
                        })
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")


def display_sidebar():
    """Display sidebar with instructions and info."""
    with st.sidebar:
        st.header("ℹ️ How it works")
        st.markdown("""
        1. **Paste a YouTube URL** in the input field
        2. **Click "Load Video"** to process the transcript
        3. **Ask questions** about the video content
        4. **Get answers** with source timestamps
        """)
        
        st.divider()
        
        st.header("⚙️ Settings")
        st.markdown(f"""
        - **Model:** Llama 3 (8B) via Groq
        - **Embeddings:** all-MiniLM-L6-v2
        - **Top K Results:** {TOP_K_RESULTS}
        """)
        
        st.divider()
        
        st.header("📋 Tips")
        st.markdown("""
        - Ask specific questions for better answers
        - Click on timestamps to jump to that part
        - Works best with videos that have captions
        """)


def main():
    """Main application function."""
    # Initialize
    init_session_state()
    
    # Display UI components
    display_header()
    display_sidebar()
    
    # Video input section
    url, load_button = display_video_input()
    
    # Handle video loading
    if load_button and url:
        st.session_state.processing = True
        success = process_video(url)
        st.session_state.processing = False
        if success:
            st.rerun()
    
    # Display video info if loaded
    display_video_info()
    
    # Display chat if video is loaded
    if st.session_state.video_loaded:
        st.divider()
        display_chat()


if __name__ == "__main__":
    main()