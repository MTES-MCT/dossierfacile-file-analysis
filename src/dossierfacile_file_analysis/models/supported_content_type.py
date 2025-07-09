from enum import Enum


class SupportedContentType(Enum):
    JPG = "image/jpg"
    JPEG = "image/jpeg"
    PNG = "image/png"
    PDF = "application/pdf"

    @staticmethod
    def get_supported_content_type(content_type: str):
        match content_type :
            case SupportedContentType.JPG.value:
                return SupportedContentType.JPG
            case SupportedContentType.JPEG.value:
                return SupportedContentType.JPEG
            case SupportedContentType.PNG.value:
                return SupportedContentType.PNG
            case SupportedContentType.PDF.value:
                return SupportedContentType.PDF
            case _:
                return None