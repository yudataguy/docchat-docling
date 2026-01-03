from openai import OpenAI
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

client = OpenAI(api_key=settings.OPENAI_API_KEY)


class RelevanceChecker:
    def __init__(self):
        # Initialize with OpenAI
        self.model_id = "gpt-5-mini"
        self.max_tokens = 1000  # GPT-5 reasoning models need more tokens

    def check(self, question: str, retriever, k=3) -> str:
        """
        1. Retrieve the top-k document chunks from the global retriever.
        2. Combine them into a single text string.
        3. Pass that text + question to the LLM for classification.

        Returns: "CAN_ANSWER", "PARTIAL", or "NO_MATCH".
        """

        logger.debug(f"RelevanceChecker.check called with question='{question}' and k={k}")

        # Retrieve doc chunks from the ensemble retriever
        top_docs = retriever.invoke(question)
        if not top_docs:
            logger.debug("No documents returned from retriever.invoke(). Classifying as NO_MATCH.")
            return "NO_MATCH"

        # Combine the top k chunk texts into one string
        document_content = "\n\n".join(doc.page_content for doc in top_docs[:k])

        # Create a prompt for the LLM to classify relevance
        prompt = f"""
        You are an AI relevance checker between a user's question and provided document content.

        **Instructions:**
        - Classify how well the document content addresses the user's question.
        - Respond with only one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH.
        - Do not include any additional text or explanation.

        **Labels:**
        1) "CAN_ANSWER": The passages contain enough explicit information to fully answer the question.
        2) "PARTIAL": The passages mention or discuss the question's topic but do not provide all the details needed for a complete answer.
        3) "NO_MATCH": The passages do not discuss or mention the question's topic at all.

        **Important:** If the passages mention or reference the topic or timeframe of the question in any way, even if incomplete, respond with "PARTIAL" instead of "NO_MATCH".

        **Question:** {question}
        **Passages:** {document_content}

        **Respond ONLY with one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH**
        """

        # Call the LLM
        try:
            response = client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=self.max_tokens
            )
            llm_response = response.choices[0].message.content.strip().upper()
        except Exception as e:
            print(f"[RelevanceChecker] Error during model inference: {e}")
            return "NO_MATCH"

        print(f"[RelevanceChecker] Raw response: '{llm_response}'")

        # Parse response - check if any valid label is contained in the response
        if "CAN_ANSWER" in llm_response:
            classification = "CAN_ANSWER"
        elif "PARTIAL" in llm_response:
            classification = "PARTIAL"
        elif "NO_MATCH" in llm_response:
            classification = "NO_MATCH"
        else:
            print(f"[RelevanceChecker] Could not parse response, defaulting to PARTIAL")
            classification = "PARTIAL"  # Default to PARTIAL instead of NO_MATCH

        print(f"[RelevanceChecker] Classification: {classification}")
        return classification
