class BlurryResult:

    def __init__(self, is_blurry: bool, is_blank: bool, ocr_mean_score: float = 0, ocr_tokens: int = 0):
        self.is_blurry = is_blurry
        self.is_blank = is_blank
        self.ocr_mean_score = ocr_mean_score
        self.ocr_tokens = ocr_tokens

    def __repr__(self):
        return f"BlurryResult(is_blurry={self.is_blurry}, is_blank={self.is_blank}, ocr_mean_score={self.ocr_mean_score}, orc_tokens={self.ocr_tokens})"

    def to_dict(self):
        return {
            "isBlurry": self.is_blurry,
            "isBlank": self.is_blank,
            "ocrMeanScore": self.ocr_mean_score,
            "ocrTokens": self.ocr_tokens,
        }