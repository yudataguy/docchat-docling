import warnings
warnings.filterwarnings("ignore", message="Could not load the custom kernel")

import gradio as gr
import hashlib
import markdown
from typing import List, Dict
import os

from document_processor.file_handler import DocumentProcessor
from retriever.builder import RetrieverBuilder
from agents.workflow import AgentWorkflow
from config import constants, settings
from utils.logging import logger

EXAMPLES = {
    "Federal Acquisition Regulation (FAR)": {
        "question": "What are the clauses related to cybersecurity in the Federal Acquisition Regulation (FAR)? Summarize them.",
        "file_paths": ["examples/FAR.pdf"]
    }
}

processor = DocumentProcessor()
retriever_builder = RetrieverBuilder()
workflow = AgentWorkflow()


def _get_file_hashes(uploaded_files: List) -> frozenset:
    """Generate SHA-256 hashes for uploaded files."""
    hashes = set()
    for file in uploaded_files:
        with open(file.name, "rb") as f:
            hashes.add(hashlib.sha256(f.read()).hexdigest())
    return frozenset(hashes)


def load_example(example_key: str):
    """Load example documents and question."""
    if not example_key or example_key not in EXAMPLES:
        return [], ""

    ex_data = EXAMPLES[example_key]
    question = ex_data["question"]
    file_paths = ex_data["file_paths"]

    loaded_files = []
    for path in file_paths:
        if os.path.exists(path):
            loaded_files.append(path)
        else:
            logger.warning(f"File not found: {path}")

    return loaded_files, question


def process_question(question_text: str, uploaded_files: List, state: Dict, progress=gr.Progress()):
    """Handle questions with document caching."""
    try:
        if not question_text.strip():
            raise ValueError("Question cannot be empty")
        if not uploaded_files:
            raise ValueError("No documents uploaded")

        current_hashes = _get_file_hashes(uploaded_files)

        if state["retriever"] is None or current_hashes != state["file_hashes"]:
            progress(0.1, desc="Processing documents...")
            chunks = processor.process(uploaded_files)

            # Pass progress callback for detailed embedding updates
            def embedding_progress(prog, desc):
                progress(0.1 + prog * 0.5, desc=desc)

            retriever = retriever_builder.build_hybrid_retriever(chunks, progress_callback=embedding_progress)

            state.update({
                "file_hashes": current_hashes,
                "retriever": retriever
            })

        progress(0.6, desc="Checking relevance...")
        progress(0.7, desc="Generating answer...")
        result = workflow.full_pipeline(
            question=question_text,
            retriever=state["retriever"]
        )
        progress(0.9, desc="Finalizing...")

        # Convert markdown to HTML with styling (dark mode compatible)
        answer_content = markdown.markdown(result["draft_answer"])
        answer_html = f"""
        <div style="background: #1e3a5f; border-left: 4px solid #3b82f6; padding: 16px; border-radius: 8px; line-height: 1.6; color: #e2e8f0;">
            {answer_content}
        </div>
        """

        verification_content = markdown.markdown(result["verification_report"].replace("\n", "<br>"))
        verification_html = f"""
        <div style="background: #14532d; border-left: 4px solid #22c55e; padding: 16px; border-radius: 8px; line-height: 1.8; color: #dcfce7;">
            {verification_content}
        </div>
        """

        # Format sources
        sources = result.get("sources", [])
        if sources:
            # Deduplicate sources by (source, page)
            seen = set()
            unique_sources = []
            for s in sources:
                key = (s["source"], s["page"])
                if key not in seen:
                    seen.add(key)
                    unique_sources.append(s)

            source_items = []
            for s in unique_sources:
                if s["page"]:
                    source_items.append(f"<li>{s['source']} — Page {s['page']}</li>")
                else:
                    source_items.append(f"<li>{s['source']}</li>")

            sources_html = f"""
            <div style="background: #1e293b; border-left: 4px solid #6366f1; padding: 16px; border-radius: 8px; line-height: 1.6; color: #c7d2fe;">
                <ul style="margin: 0; padding-left: 20px;">
                    {"".join(source_items)}
                </ul>
            </div>
            """
        else:
            sources_html = "<div style='color: #94a3b8;'>No sources available</div>"

        return answer_html, verification_html, sources_html, state

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return f"Error: {str(e)}", "", "", state


theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

with gr.Blocks(theme=theme, title="DocChat") as demo:
    session_state = gr.State({
        "file_hashes": frozenset(),
        "retriever": None
    })

    with gr.Row():
        with gr.Column():
            example_dropdown = gr.Dropdown(
                label="Shelf",
                choices=list(EXAMPLES.keys()),
                value=None,
            )
            load_example_btn = gr.Button("Load Example")
            files = gr.Files(label="Documents", file_types=constants.ALLOWED_TYPES)
            question = gr.Textbox(label="Question", lines=3)
            submit_btn = gr.Button("Submit", variant="primary")

        with gr.Column():
            gr.Markdown("**Answer**")
            answer_output = gr.HTML()
            gr.Markdown("**Sources**")
            sources_output = gr.HTML()
            gr.Markdown("**Verification Report**")
            verification_output = gr.HTML()

    load_example_btn.click(
        fn=load_example,
        inputs=[example_dropdown],
        outputs=[files, question]
    )

    submit_btn.click(
        fn=process_question,
        inputs=[question, files, session_state],
        outputs=[answer_output, verification_output, sources_output, session_state]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=5000, share=True)
