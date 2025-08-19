import os


class ProjPaths:
    __root = None
    __model_dir = None

    @staticmethod
    def get_project_root():
        if ProjPaths.__root is None:
            current_file_path = os.path.abspath(__file__)
            common_dir = os.path.dirname(current_file_path)
            ProjPaths.__root = os.path.dirname(common_dir)
        return ProjPaths.__root

    @staticmethod
    def get_model_dir():
        if ProjPaths.__model_dir is None:
            ProjPaths.__model_dir = ProjPaths.get_project_root() + "/models"
        return ProjPaths.__model_dir

