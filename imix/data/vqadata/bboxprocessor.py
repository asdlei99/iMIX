from ..builder import EMBEDDING
from .vocabprocessor import VocabProcessor


@EMBEDDING.register_module()
class BBoxProcessor(VocabProcessor):
    """Generates bboxes in proper format. Takes in a dict which contains "info"
    key which is a list of dicts containing following for each of the the
    bounding box.

    Example bbox input::

        {
            "info": [
                {
                    "bounding_box": {
                        "top_left_x": 100,
                        "top_left_y": 100,
                        "width": 200,
                        "height": 300
                    }
                },
                ...
            ]
        }


    This will further return a Sample in a dict with key "bbox" with last
    dimension of 4 corresponding to "xyxy". So sample will look like following:

    Example Sample::

        Sample({
            "coordinates": torch.Size(n, 4),
            "width": List[number], # size n
            "height": List[number], # size n
            "bbox_types": List[str] # size n, either xyxy or xywh.
            # currently only supports xyxy.
        })
    """

    def __init__(self, max_length, *args, **kwargs):
        from mmf.utils.dataset import build_bbox_tensors

        self.lambda_fn = build_bbox_tensors
        self.max_length = max_length
        # self._init_extras(config)

    def __call__(self, item):
        info = item['info']
        if self.preprocessor is not None:
            info = self.preprocessor(info)

        return {'bbox': self.lambda_fn(info, self.max_length)}
