from .base import PyiqaMetric


class MUSIQ(PyiqaMetric):
    name = "musiq"
    label = "MUSIQ"

    def __init__(self, device, model_name="musiq", **kwargs):
        super().__init__(model_name, device, **kwargs)
