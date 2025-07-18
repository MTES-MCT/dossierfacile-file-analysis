class BlurryResult:

    def __init__(self, laplacian_variance: float, is_blurry: bool, is_blank: bool):
        self.laplacian_variance = laplacian_variance
        self.is_blurry = is_blurry
        self.is_blank = is_blank

    def __repr__(self):
        return f"BlurryResult(laplacian_variance={self.laplacian_variance}, is_blurry={self.is_blurry}, is_blank={self.is_blank})"

    def to_dict(self):
        return {
            "laplacianVariance": self.laplacian_variance,
            "isBlurry": self.is_blurry,
            "isBlank": self.is_blank
        }