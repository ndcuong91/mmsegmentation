_base_ = [
    '../_base_/models/pspnet_r50-d8.py',
    '../_base_/datasets/publaynet.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
model = dict(
    decode_head=dict(num_classes=6), auxiliary_head=dict(num_classes=6))
test_cfg = dict(mode='whole')