from enum import Enum

class Category(Enum):
    FACTUAL_KNOWLEDGE = 1
    MATH = 2
    SENTIMENT = 3
    SUMMARISATION = 4
    NER = 5
    CODE_DEBUG = 6
    LOGICAL = 7
    CODE_GENERATION = 8
    NONE = 9

class Request:
    def __init__(self, id, prompt):
        self.id = id;
        self.prompt = prompt
        self.category = Category.NONE
        self.difficulty = 0.0 # min: 0.0; max 10.0