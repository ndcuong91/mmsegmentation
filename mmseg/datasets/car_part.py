from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class car_partDataset(CustomDataset):
    """car_partDataset
    """
    CLASSES = ('background',
               'back_bumper',
               'back_glass',
               'back_left_door',
               'back_left_light',
               'back_right_door',
               'back_right_light',
               'back_left_fender',
               'back_right_fender',
               'front_bumper',
               'front_glass',
               'front_left_door',
               'front_left_fender',
               'front_right_fender',
               'front_left_light',
               'front_right_door',
               'front_right_light',
               'left_mirror',
               'right_mirror',
               'hood',
               'tailgate',
               'trunk',
               'wheel')
    PALETTE = [[120, 120, 120],
               [255, 0, 0],
               [0, 255, 0],
               [0, 0, 255],
               [128, 64, 128],
               [244, 35, 232],
               [70, 140, 70],
               [0, 0, 180],
               [0, 0, 180],
               [190, 0, 153],
               [100, 180, 120],
               [128, 0, 0],
               [0, 128, 0],
               [0, 0, 128],
               [255, 64, 64],
               [200, 35, 100],
               [140, 0, 70],
               [200, 102, 102],
               [153, 0, 153],
               [30, 153, 20],
               [150, 50, 50],
               [150, 30, 250],
               [60, 200, 60]]

    def __init__(self, **kwargs):
        super(car_partDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
