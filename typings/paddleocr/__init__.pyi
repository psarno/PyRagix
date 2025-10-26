"""Type stubs for PaddleOCR."""

from typing import Any
from numpy.typing import NDArray

# PaddleOCR returns: list[list[tuple[list[list[float]], tuple[str, float]]]]
# Each image result contains lines, each line is a 2-tuple: (bbox_coords, (text, confidence))
# bbox_coords is list of 4 corner points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
# text_conf is tuple: (text: str, confidence: float)

BBox = list[list[float]]  # 4 corner points
TextConf = tuple[str, float]  # (text, confidence)
OCRLine = tuple[BBox, TextConf]  # (bbox, (text, conf))
OCRResult = list[OCRLine]  # All lines in one image
OCRBatchResult = list[OCRResult]  # Results for multiple images

class PaddleOCR:
    def __init__(
        self,
        *,
        use_angle_cls: bool = False,
        lang: str = "en",
        use_gpu: bool = False,
        show_log: bool = True,
        **kwargs: Any,
    ) -> None: ...

    def predict(
        self,
        input: str | NDArray[Any],
        *,
        use_doc_orientation_classify: bool | None = None,
        use_doc_unwarping: bool | None = None,
        use_textline_orientation: bool | None = None,
        text_det_limit_side_len: int | None = None,
        text_det_limit_type: str | None = None,
        text_det_thresh: float | None = None,
        text_det_box_thresh: float | None = None,
        text_det_unclip_ratio: float | None = None,
        text_rec_score_thresh: float | None = None,
        return_word_box: bool | None = None,
    ) -> OCRBatchResult: ...

    def ocr(
        self,
        img: str | NDArray[Any],
        det: bool = True,
        rec: bool = True,
        cls: bool = False,
    ) -> OCRBatchResult: ...
