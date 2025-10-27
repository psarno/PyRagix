from collections.abc import Sequence

TokenizedCorpus = Sequence[Sequence[str]]

class BM25Okapi:
    def __init__(
        self,
        corpus: TokenizedCorpus,
        k1: float = ...,
        b: float = ...,
    ) -> None: ...
    def get_scores(self, query: Sequence[str]) -> list[float]: ...
    def get_top_n(
        self,
        query: Sequence[str],
        documents: Sequence[str],
        n: int = ...,
    ) -> list[str]: ...
