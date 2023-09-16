from typing import List
from core.engine import Value


def softmax(tensor: List[Value]):
    exp_values = [value.exp() for value in tensor]
    return [exp_value / sum(exp_values) for exp_value in exp_values]