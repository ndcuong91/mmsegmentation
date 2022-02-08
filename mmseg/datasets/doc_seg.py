from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class doc_segDataset(CustomDataset):
    """doc_segDataset
    """
    CLASSES = ('background','doc')
    PALETTE = [[120,120,120],[255, 0, 0]]
    def __init__(self, **kwargs):
        super(doc_segDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)