"""Untrusted parser/LLM-adapter interfaces."""
from dataclasses import dataclass

@dataclass(frozen=True)
class ParserCandidate:
    air_source: str
    confidence_bp: int
    trusted: bool = False

class ParserAdapter:
    def parse_text(self, text: str) -> ParserCandidate:
        return ParserCandidate(air_source=text, confidence_bp=0, trusted=False)
