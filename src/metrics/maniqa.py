from .base import PyiqaMetric


class MANIQA(PyiqaMetric):
    name = "maniqa"
    label = "MANIQA"

    def __init__(self, device, model_name="maniqa", **kwargs):
        super().__init__(model_name, device, **kwargs)
