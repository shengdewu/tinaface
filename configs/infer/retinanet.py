# 1. data
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
size_divisor = 32

data_pipeline = [
    dict(typename='LoadImageFromFile'),
    dict(
        typename='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(typename='Resize', keep_ratio=True),
            dict(typename='RandomFlip'),
            dict(typename='Normalize', **img_norm_cfg),
            dict(typename='Pad', size_divisor=size_divisor),
            dict(typename='DefaultFormatBundle'),
            dict(typename='Collect', keys=['img']),
        ])
]

# 2. model
num_classes = 80
strides = [8, 16, 32, 64, 128]
use_sigmoid = True
scales_per_octave = 3
ratios = [0.5, 1.0, 2.0]
num_anchors = scales_per_octave * len(ratios)

detector = dict(
    typename='SingleStageDetector',
    backbone=dict(
        typename='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,  # TODO
        norm_cfg=dict(
            typename='BN',
            requires_grad=True),  # TODO
        norm_eval=True,
        style='pytorch'),  # TODO
    neck=dict(
        typename='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    head=dict(
        typename='RetinaHead',
        num_classes=num_classes,
        num_anchors=num_anchors,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        use_sigmoid=use_sigmoid))

# 3. engine
meshgrid = dict(
    typename='BBoxAnchorMeshGrid',
    strides=strides,
    base_anchor=dict(
        typename='BBoxBaseAnchor',
        octave_base_scale=4,
        scales_per_octave=scales_per_octave,
        ratios=ratios,
        base_sizes=strides))

bbox_coder = dict(
    typename='DeltaXYWHBBoxCoder',
    target_means=[.0, .0, .0, .0],
    target_stds=[1.0, 1.0, 1.0, 1.0])

## infer engine
infer_engine = dict(
    typename='InferEngine',
    detector=detector,
    meshgrid=meshgrid,
    converter=dict(
        typename='BBoxAnchorConverter',
        num_classes=num_classes,
        test_cfg=dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.6,
            nms=dict(
                typename='nms',
                iou_thr=0.5),
            max_per_img=100),
        bbox_coder=bbox_coder,
        rescale=True))

# 4. weights
weights = dict(filepath='workdir/retinanet/epoch_12_weights.pth')

# 5. show
class_names = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
