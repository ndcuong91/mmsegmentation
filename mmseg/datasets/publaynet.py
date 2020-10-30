from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class publaynetDataset(CustomDataset):
    """table_structure1
    """
    CLASSES = ('background','text', 'title', 'list', 'table', 'figure')
    PALETTE = [[120, 120, 120],[50, 255, 0],[255, 0, 0],[0, 255, 255],[255, 192, 203],[100, 0, 255]]
    def __init__(self, **kwargs):
        super(publaynetDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)