"""Shared test fixtures."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from src.knowledge.models import (
    Evidence,
    EvidenceType,
    Gap,
    GapSeverity,
    KnowledgeNode,
    Outline,
    OutlineItem,
    PaperMeta,
    SaturationScores,
)
from src.knowledge.map import KnowledgeMap


@pytest.fixture
def sample_paper_meta() -> PaperMeta:
    return PaperMeta(
        paper_id="ss:abc123",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        year=2017,
        url="https://arxiv.org/abs/1706.03762",
        venue="NeurIPS",
        citation_count=100000,
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
        full_text_available=True,
        source="semantic_scholar",
    )


@pytest.fixture
def sample_papers() -> list[PaperMeta]:
    return [
        PaperMeta(
            paper_id="ss:001",
            title="Deep Residual Learning for Image Recognition",
            authors=["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren"],
            year=2016,
            venue="CVPR",
            citation_count=80000,
            abstract="Residual networks...",
            source="semantic_scholar",
        ),
        PaperMeta(
            paper_id="arxiv:1706.03762",
            title="Attention Is All You Need",
            authors=["Ashish Vaswani"],
            year=2017,
            venue="NeurIPS",
            citation_count=100000,
            abstract="Transformer architecture...",
            source="arxiv",
        ),
        PaperMeta(
            paper_id="ss:003",
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            authors=["Jacob Devlin"],
            year=2019,
            venue="NAACL",
            citation_count=50000,
            abstract="Bidirectional representations...",
            source="semantic_scholar",
        ),
    ]


@pytest.fixture
def sample_evidence(sample_paper_meta) -> Evidence:
    return Evidence(
        paper=sample_paper_meta,
        section="Section 3.2",
        quote="The Transformer follows this overall architecture using stacked self-attention...",
        evidence_type=EvidenceType.ORIGINAL_CLAIM,
    )


@pytest.fixture
def sample_knowledge_node(sample_paper_meta, sample_evidence) -> KnowledgeNode:
    return KnowledgeNode(
        id="node_001",
        topic="Self-Attention Mechanism",
        claim="Self-attention relates different positions of a single sequence to compute a representation",
        evidence=[sample_evidence],
        confidence=0.9,
        related_nodes=["node_002"],
        gaps=[],
    )


@pytest.fixture
def sample_knowledge_nodes(sample_papers) -> list[KnowledgeNode]:
    nodes = []
    for i, paper in enumerate(sample_papers, 1):
        evidence = Evidence(
            paper=paper,
            section=f"Section {i}",
            quote=f"Key quote from {paper.title}",
            evidence_type=EvidenceType.ORIGINAL_CLAIM,
        )
        node = KnowledgeNode(
            id=f"node_{i:03d}",
            topic=f"Topic {i}",
            claim=f"Claim from {paper.title}",
            evidence=[evidence],
            confidence=0.7 + (i * 0.1),
        )
        nodes.append(node)
    return nodes


@pytest.fixture
def sample_gaps() -> list[Gap]:
    return [
        Gap(
            description="No evidence on computational efficiency of attention variants",
            severity=GapSeverity.CRITICAL,
            saturation=0.3,
        ),
        Gap(
            description="Limited comparison with CNN-based approaches",
            severity=GapSeverity.IMPORTANT,
            saturation=0.5,
        ),
        Gap(
            description="Few studies on domain-specific applications of transformers",
            severity=GapSeverity.NICE_TO_HAVE,
            saturation=0.8,
        ),
    ]


@pytest.fixture
def sample_knowledge_map(sample_knowledge_nodes) -> KnowledgeMap:
    km = KnowledgeMap()
    for node in sample_knowledge_nodes:
        km.add_node(node)
    return km


@pytest.fixture
def sample_outline() -> Outline:
    return Outline(
        title="Literature Review: Transformer Architectures",
        chapters=[
            OutlineItem(
                title="Introduction",
                topics=["Background"],
                description="Overview of sequence transduction models",
            ),
            OutlineItem(
                title="Self-Attention Mechanisms",
                topics=["Self-Attention Mechanism"],
                description="Core attention architecture",
            ),
            OutlineItem(
                title="Transformer Variants",
                topics=["BERT", "GPT"],
                description="Key variants and their contributions",
            ),
            OutlineItem(
                title="Conclusion",
                topics=["Summary"],
                description="Synthesis and future directions",
            ),
        ],
    )


@pytest.fixture
def sample_citation_index() -> dict:
    return {
        1: {
            "paper_id": "ss:001",
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "year": 2017,
            "venue": "NeurIPS",
            "url": "https://arxiv.org/abs/1706.03762",
        },
        2: {
            "paper_id": "ss:002",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "authors": ["Jacob Devlin"],
            "year": 2019,
            "venue": "NAACL",
            "url": "",
        },
    }


@pytest.fixture
def agent_state_broad() -> dict:
    return {
        "research_question": "transformer attention mechanisms",
        "max_rounds": 5,
        "output_dir": "reports",
        "current_round": 1,
        "phase": "broad",
        "round_history": [],
        "search_queries": [],
        "paper_index": {},
        "search_results": [],
        "papers_read": {},
        "read_failures": {},
        "knowledge_nodes": [],
        "gaps": [],
        "outline": None,
        "outline_feedback": "",
        "outline_approved": False,
        "chapters": {},
        "final_report": "",
        "messages": [],
        "errors": [],
        "iteration_count": 0,
        "last_saturation_scores": {},
        "consecutive_no_improvement": 0,
        "checkpoint_metadata": {},
        "user_action": "approve",
        "output_zh": False,
        "final_report_zh": "",
    }


@pytest.fixture
def mock_settings(monkeypatch):
    """Mock settings with test values (no real API key needed)."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    from src.config.settings import Settings
    return Settings()
