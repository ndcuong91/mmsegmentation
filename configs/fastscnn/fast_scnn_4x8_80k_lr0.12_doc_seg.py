_base_ = [
    '../_base_/models/fast_scnn.py', '../_base_/datasets/doc_seg.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k_new.py'
]

# Re-config the data sampler.
data = dict(samples_per_gpu=8, workers_per_gpu=4)

# Re-config the optimizer.
optimizer = dict(type='SGD', lr=0.12, momentum=0.9, weight_decay=4e-5)