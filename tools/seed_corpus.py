"""
Seed corpus for the ChromaDB vector store: agentic AI / RAG research.

Each entry is an ORIGINAL summary written for this project (not copied from
the paper's abstract) describing a well-known paper's core idea, so the demo
has stable, citeable, licensing-safe content. Real titles and arXiv IDs are
included as metadata for source attribution in the final report.
"""

CORPUS = [
    {
        "doc_id": "react-2022",
        "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
        "arxiv_id": "2210.03629",
        "summary": (
            "Introduces a prompting approach where a language model alternates between "
            "generating reasoning traces and taking actions (like calling a search tool), "
            "using each observation to inform the next reasoning step. Interleaving thought "
            "and action reduces hallucination compared to reasoning-only chains, because the "
            "model can verify intermediate claims against real tool outputs rather than relying "
            "purely on its own generated context."
        ),
    },
    {
        "doc_id": "reflexion-2023",
        "title": "Reflexion: Language Agents with Verbal Reinforcement Learning",
        "arxiv_id": "2303.11366",
        "summary": (
            "Proposes giving an agent a memory of its own past failures, expressed in natural "
            "language rather than gradient updates. After each attempt, the agent generates a "
            "short self-critique of what went wrong, stores it, and conditions future attempts "
            "on that critique. This mirrors a critic-and-revise loop without needing to "
            "fine-tune the underlying model, which is directly relevant to reflection-style "
            "multi-agent pipelines."
        ),
    },
    {
        "doc_id": "toolformer-2023",
        "title": "Toolformer: Language Models Can Teach Themselves to Use Tools",
        "arxiv_id": "2302.04761",
        "summary": (
            "Shows that a language model can learn, largely on its own, when and how to invoke "
            "external tools (calculators, search engines, calendars) by inserting API calls into "
            "training text and keeping only the calls that improve next-token prediction. "
            "Establishes tool use as something that can be learned rather than only hand-coded "
            "into an orchestration layer."
        ),
    },
    {
        "doc_id": "rag-lewis-2020",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "arxiv_id": "2005.11401",
        "summary": (
            "The original RAG paper: couples a pretrained retriever with a pretrained "
            "sequence-to-sequence generator, so the model conditions its output on passages "
            "retrieved from an external index rather than only on parameters learned at "
            "training time. This separates 'what the model knows' from 'what it can look up,' "
            "which is the foundational argument for hybrid retrieval architectures."
        ),
    },
    {
        "doc_id": "self-rag-2023",
        "title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        "arxiv_id": "2310.11511",
        "summary": (
            "Trains a model to emit special reflection tokens that decide, at generation time, "
            "whether retrieval is even necessary and whether a generated segment is actually "
            "supported by the retrieved passages. This is an argument for building critique "
            "directly into the generation loop rather than bolting it on as a separate pass, "
            "though a separate Critic agent is more practical without custom fine-tuning."
        ),
    },
    {
        "doc_id": "cot-2022",
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "arxiv_id": "2201.11903",
        "summary": (
            "Demonstrates that simply prompting a model to produce intermediate reasoning steps "
            "before its final answer substantially improves performance on multi-step reasoning "
            "tasks, without any architecture change. This is the baseline that multi-agent "
            "systems are often compared against when justifying the added complexity of "
            "decomposing work across separate agent roles."
        ),
    },
    {
        "doc_id": "tot-2023",
        "title": "Tree of Thoughts: Deliberate Problem Solving with Large Language Models",
        "arxiv_id": "2305.10601",
        "summary": (
            "Generalizes chain-of-thought into a search over a tree of partial reasoning paths, "
            "letting the model explore multiple candidate directions, self-evaluate them, and "
            "backtrack. Relevant to planning agents that need to consider several research "
            "directions before committing to one, rather than committing to the first "
            "decomposition generated."
        ),
    },
    {
        "doc_id": "mrkl-2022",
        "title": "MRKL Systems: A Modular, Neuro-Symbolic Architecture Combining LLMs, External Knowledge, and Discrete Reasoning",
        "arxiv_id": "2205.00445",
        "summary": (
            "Argues for routing sub-tasks to whichever module handles them best -- a calculator "
            "for arithmetic, a knowledge base for facts, an LLM for language -- rather than "
            "asking one monolithic model to do everything. This is an early, explicit statement "
            "of the cost-aware model-routing argument: use the cheapest capable component for "
            "each sub-task."
        ),
    },
    {
        "doc_id": "webgpt-2021",
        "title": "WebGPT: Browser-assisted Question-Answering with Human Feedback",
        "arxiv_id": "2112.09332",
        "summary": (
            "Fine-tunes a model to use a text-based web browser -- issuing search queries, "
            "clicking links, and quoting passages -- to answer long-form questions, with human "
            "feedback used to reward answers that are well-supported by cited sources. An early "
            "demonstration that citation-grounded web retrieval measurably reduces unsupported "
            "claims in long-form answers."
        ),
    },
    {
        "doc_id": "hyde-2022",
        "title": "Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)",
        "arxiv_id": "2212.10496",
        "summary": (
            "Proposes generating a hypothetical answer to a query first, then embedding that "
            "hypothetical document to search a vector index, rather than embedding the raw query. "
            "The generated text is closer in style to what's actually stored in the index, which "
            "improves retrieval quality -- a technique applicable to the sub-question rewriting "
            "step before hitting a vector store."
        ),
    },
    {
        "doc_id": "raptor-2024",
        "title": "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval",
        "arxiv_id": "2401.18059",
        "summary": (
            "Builds a tree of recursive summaries over a document collection, so retrieval can "
            "pull either a fine-grained passage or a higher-level summary depending on what the "
            "query needs. Useful context for justifying chunking-and-summarization decisions in "
            "a vector store, especially for questions that need broad synthesis rather than a "
            "single fact."
        ),
    },
    {
        "doc_id": "crag-2024",
        "title": "Corrective Retrieval Augmented Generation (CRAG)",
        "arxiv_id": "2401.15884",
        "summary": (
            "Adds a lightweight evaluator that grades retrieved documents as correct, ambiguous, "
            "or incorrect, and triggers a corrective action -- like falling back to a web search "
            "-- when retrieval quality is poor. Directly analogous to the reranking/filtering "
            "step and the hybrid web-plus-vector-store fallback design used in this project."
        ),
    },
    {
        "doc_id": "multiagent-debate-2023",
        "title": "Improving Factuality and Reasoning in Language Models through Multiagent Debate",
        "arxiv_id": "2305.14325",
        "summary": (
            "Has multiple instances of a language model propose answers and then critique each "
            "other's reasoning over several rounds, converging on a more accurate final answer "
            "than any single instance produces alone. Supports the general claim that separating "
            "'generate' and 'critique' into distinct passes improves factual reliability."
        ),
    },
    {
        "doc_id": "agentic-rag-survey-2024",
        "title": "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG",
        "arxiv_id": "2501.09136",
        "summary": (
            "Surveys the shift from static, single-pass RAG pipelines toward agentic RAG systems "
            "that plan multi-step retrieval, use multiple tools, and reflect on intermediate "
            "results. Frames exactly the design space this project sits in: planner, hybrid "
            "retriever, writer, and critic as distinct agent roles rather than one prompt doing "
            "everything."
        ),
    },
    {
        "doc_id": "self-consistency-2022",
        "title": "Self-Consistency Improves Chain of Thought Reasoning in Language Models",
        "arxiv_id": "2203.11171",
        "summary": (
            "Samples multiple independent reasoning paths for the same question and takes the "
            "majority-vote answer, rather than trusting a single generation. A cheap alternative "
            "to a dedicated critic agent when the task has a clear final answer to vote on, "
            "though less suited to open-ended report synthesis."
        ),
    },
]
