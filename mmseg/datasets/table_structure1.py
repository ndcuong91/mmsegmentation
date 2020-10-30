from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class table_structure1Dataset(CustomDataset):
    """table_structure1
    """
    CLASSES = ('background','cell')
    PALETTE = [[120, 0, 0],[0, 120, 0]]
    def __init__(self, **kwargs):
        super(table_structure1Dataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)