from n4.numeric import PyFloat
from n4.core import Value

def new_value(val: float) -> Value[PyFloat]:
    return Value.from_float(val, PyFloat)
