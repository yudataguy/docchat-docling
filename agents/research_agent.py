from openai import OpenAI
from typing import Dict, List
from langchain.schema import Document
from config.settings import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)


class ResearchAgent:
    def __init__(self):
        """
        Initialize the research agent with OpenAI.
        """
        print("Initializing ResearchAgent with OpenAI gpt-5.2...")
        self.model_id = "gpt-5.2"
        self.max_tokens = 4000  # GPT-5 reasoning models need more tokens
        print("ResearchAgent initialized successfully.")

    def sanitize_response(self, response_text: str) -> str:
        """
        Sanitize the LLM's response by stripping unnecessary whitespace.
        """
        return response_text.strip()

    def generate_prompt(self, question: str, context: str) -> str:
        """
        Generate a structured prompt for the LLM to generate a precise and factual answer.
        """
        prompt = f"""
        You are an AI assistant designed to provide precise and factual answers based on the given context.

        **Instructions:**
        - Answer the following question using only the provided context.
        - Be clear, concise, and factual.
        - Return as much information as you can get from the context.
        - When referencing specific information, cite the source using [Source X, Page Y] format.

        **Question:** {question}
        **Context:**
        {context}

        **Provide your answer below:**
        """
        return prompt

    def _build_context_with_sources(self, documents: List[Document]) -> tuple:
        """Build context string with source annotations and extract unique sources."""
        context_parts = []
        sources = []

        for i, doc in enumerate(documents):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page")

            # Build source reference
            if page:
                source_ref = f"[Source {i+1}: {source}, Page {page}]"
                sources.append({"index": i+1, "source": source, "page": page})
            else:
                source_ref = f"[Source {i+1}: {source}]"
                sources.append({"index": i+1, "source": source, "page": None})

            context_parts.append(f"{source_ref}\n{doc.page_content}")

        return "\n\n".join(context_parts), sources

    def generate(self, question: str, documents: List[Document]) -> Dict:
        """
        Generate an initial answer using the provided documents.
        """
        print(f"ResearchAgent.generate called with question='{question}' and {len(documents)} documents.")

        # Build context with source annotations
        context, sources = self._build_context_with_sources(documents)
        print(f"Combined context length: {len(context)} characters.")

        # Create a prompt for the LLM
        prompt = self.generate_prompt(question, context)
        print("Prompt created for the LLM.")

        # Call the LLM to generate the answer
        try:
            print("Sending prompt to the model...")
            response = client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=self.max_tokens
            )
            print("LLM response received.")
            llm_response = response.choices[0].message.content.strip()
            print(f"Raw LLM response:\n{llm_response}")
        except Exception as e:
            print(f"Error during model inference: {e}")
            llm_response = "I cannot answer this question based on the provided documents."

        # Sanitize the response
        draft_answer = self.sanitize_response(llm_response) if llm_response else "I cannot answer this question based on the provided documents."

        print(f"Generated answer: {draft_answer}")

        return {
            "draft_answer": draft_answer,
            "context_used": context,
            "sources": sources
        }
