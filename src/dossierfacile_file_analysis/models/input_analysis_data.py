import os

from dossierfacile_file_analysis.models.downloaded_file import DownloadedFile


class InputAnalysisData:
    def __init__(self, downloaded_file: DownloadedFile, list_of_images=None):
        if list_of_images is None:
            list_of_images = []
        self.type = downloaded_file.file_type
        self.initial_file = downloaded_file.file_path
        self.list_of_images = list_of_images if list_of_images is not None else []

    def clean_files(self):
        """
        Clean up the files created during the analysis.
        :return:
        """
        if os.path.exists(self.initial_file):
            os.remove(self.initial_file)

        for image in self.list_of_images:
            if os.path.exists(image):
                os.remove(image)
