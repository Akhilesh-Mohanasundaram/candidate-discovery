from enum import Enum

class VetoType(str, Enum):
    PURE_RESEARCH = "pure_research"
    LANGCHAIN_WRAPPER = "langchain_wrapper"
    ARCHITECT_NO_CODE = "architect_no_code"
    CONSULTING_ONLY = "consulting_only"
    TITLE_CHASER = "title_chaser"
    CV_SPEECH_ROBOTICS = "cv_speech_robotics"
