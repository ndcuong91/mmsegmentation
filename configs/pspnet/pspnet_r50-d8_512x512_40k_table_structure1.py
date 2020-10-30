_base_ = [
    '../_base_/models/pspnet_r50-d8.py',
    '../_base_/datasets/table_structure1.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
model = dict(
    decode_head=dict(num_classes=1), auxiliary_head=dict(num_classes=1))
test_cfg = dict(mode='whole')