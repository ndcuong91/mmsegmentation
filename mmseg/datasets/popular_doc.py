from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class popular_docDataset(CustomDataset):
    """table_structure1
    """
    CLASSES = ('background','cccd','cccd_back','cmnd_new','cmnd_old','cmnd_old_back',
                    'driverlicense_new','driverlicense_new_back','driverlicense_old','driverlicense_old_back')
    PALETTE = [[120,120,120],[255, 0, 0],[0, 255, 0],[0, 0, 255],[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153]]
    def __init__(self, **kwargs):
        super(popular_docDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)