from typing import List, Dict, Optional
from groq import Groq
import config
from vectorstore import vector_store
from utils import format_timestamp


class RAGChain:
    """
    RAG Chain for answering questions about YouTube videos.
    """
    
    def __init__(self):
        """
        Initialize the RAG chain.
        """
        self.client = None
        self.conversation_history = []
        self._init_client()
    
    def _init_client(self):
        """
        Initialize Groq client.
        """
        if config.GROQ_API_KEY:
            self.client = Groq(api_key=config.GROQ_API_KEY)
        else:
            raise ValueError("GROQ_API_KEY not found in environment variables")
    
    def _build_system_prompt(self, context: str, video_info: Dict) -> str:
        """
        Build the system prompt with context.
        """
        return f"""You are a friendly and knowledgeable tutor helping students understand video content. Your goal is to explain concepts clearly and thoroughly.

IMPORTANT INSTRUCTIONS:
1. Answer questions naturally as if you are a helpful teacher explaining to a student
2. Be conversational, warm, and engaging in your tone
3. Provide DETAILED and THOROUGH explanations - don't be brief
4. Use simple language that students can easily understand
5. Use examples and analogies when helpful to clarify concepts
6. Break down complex ideas into smaller, easy-to-understand parts
7. If the information is available, explain it fully - don't just give short answers
8. DO NOT mention phrases like "according to the video", "the transcript says", "since the video doesn't provide" etc.
9. Just explain the concepts directly and naturally
10. If something is not covered, simply say "I don't have enough information about that specific point" without blaming the source
11. When relevant, mention timestamps naturally like "around 2:30..." or "at about 5 minutes in..."
12. Structure your answers with clear explanations, not just facts

CONTEXT FROM VIDEO:
{context}

Remember: You are teaching, not just retrieving information. Explain things in a way that helps the student truly understand."""


    def _build_messages(self, query: str, context: str, video_info: Dict) -> List[Dict]:
        """
        Build the message list for the LLM.
        """
        messages = [
            {"role": "system", "content": self._build_system_prompt(context, video_info)}
        ]
        
        # Add conversation history (last 5 exchanges)
        for exchange in self.conversation_history[-5:]:
            messages.append({"role": "user", "content": exchange["question"]})
            messages.append({"role": "assistant", "content": exchange["answer"]})
        
        # Add current question
        messages.append({"role": "user", "content": query})
        
        return messages
    
    def ask(self, question: str) -> Dict:
        """
        Ask a question and get an answer using RAG.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with 'answer', 'sources', and 'success'
        """
        if self.client is None:
            return {
                'success': False,
                'error': 'Groq client not initialized. Check your API key.'
            }
        
        if vector_store.vector_store is None:
            return {
                'success': False,
                'error': 'No video loaded. Please load a video first.'
            }
        
        try:
            # Retrieve relevant documents with scores
            results = vector_store.similarity_search_with_score(question)
            
            if not results:
                return {
                    'success': False,
                    'error': 'No relevant information found in the video.'
                }
            
            # Build context - use all results without strict filtering
            context_parts = []
            sources = []
            
            for doc, score in results:
                context_parts.append(doc.page_content)
                
                start_time = doc.metadata.get('start_time', 0)
                sources.append({
                    'text': doc.page_content[:200] + "...",
                    'timestamp': format_timestamp(start_time),
                    'start_seconds': start_time
                })
            
            context = "\n\n".join(context_parts)
            
            # Build messages
            messages = self._build_messages(question, context, vector_store.video_metadata)
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS
            )
            
            answer = response.choices[0].message.content
            
            # Store in conversation history
            self.conversation_history.append({
                "question": question,
                "answer": answer
            })
            
            return {
                'success': True,
                'answer': answer,
                'sources': sources,
                'model': config.LLM_MODEL
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'An error occurred: {str(e)}'
            }
    
    def clear_history(self):
        """
        Clear conversation history.
        """
        self.conversation_history = []


# Create a global instance
rag_chain = None


def get_rag_chain() -> RAGChain:
    """
    Get or create the RAG chain instance.
    """
    global rag_chain
    if rag_chain is None:
        rag_chain = RAGChain()
    return rag_chain
