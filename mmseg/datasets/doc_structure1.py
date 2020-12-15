from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class doc_structure1Dataset(CustomDataset):
    """table_structure1
    """
    CLASSES = ('background','text', 'table', 'figure')
    PALETTE = [[120,120,120],[255, 0, 0],[0, 255, 0],[0, 0, 255]]
    def __init__(self, **kwargs):
        super(doc_structure1Dataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)