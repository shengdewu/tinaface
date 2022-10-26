from vedacore.misc import registry
from .val_engine import ValEngine


@registry.register_module('engine')
class SingleValEngine(ValEngine):

    def __init__(self, model, meshgrid, converter, num_classes, use_sigmoid,
                 test_cfg, eval_metric):
        super().__init__(model, meshgrid, converter, num_classes, use_sigmoid,
                         test_cfg, eval_metric)

    def forward(self, data):
        return self.forward_impl(**data)

    def forward_impl(self, img, img_metas):
        return self._simple_infer(img, img_metas)
