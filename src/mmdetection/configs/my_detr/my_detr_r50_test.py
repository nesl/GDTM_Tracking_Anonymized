_base_ = [
    '../_base_/datasets/coco_detection.py', '../_base_/default_runtime.py'
]

custom_imports = dict(
        imports=['mmdet.models.dense_heads.my_detr_head', 'cad'],
        allow_failed_imports=False
)

checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth'  # noqa
model = dict(type='SingleStageDetector',
    backbone=dict(type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        #init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
        init_cfg=dict(type='Pretrained', prefix='backbone.', checkpoint=checkpoint)
    ),
    neck=dict(type='ChannelMapper', #from deformable detr
        in_channels=[2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        # norm_cfg=dict(type='GN', num_groups=32),
        norm_cfg=None,
        num_outs=1
    ),
    bbox_head=dict(type='MyDETRHead',
        num_classes=80,
        in_channels=256,
        encoder_cfg=dict(type='DETREncoder',
            num_layers=1,
            self_attn_cfg=dict(type='ResSelfAttn', attn_cfg=dict(type='QKVAttention', qk_dim=256, num_heads=8)),
            ffn_cfg=dict(type='SLP', in_channels=256, expansion_ratio=4, act_cfg=dict(type='GELU')),
        ),
        decoder_cfg=dict(type='DETRDecoder',
            self_attn_cfg=dict(type='ResSelfAttn', attn_cfg=dict(type='QKVAttention', qk_dim=256, num_heads=8)),
            cross_attn_cfg=dict(type='ResCrossAttn', attn_cfg=dict(type='QKVAttention', qk_dim=256, num_heads=8)),
            ffn_cfg=dict(type='SLP', in_channels=256, expansion_ratio=4, act_cfg=dict(type='GELU')),
            return_all_layers=True
        ),
        feats_pos_cfg=dict(type='SineEncoding2d', dim=256, out_proj=True),
        loss_cls=dict(type='CrossEntropyLoss',
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='HungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=1.),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0)
        )
    ),
    test_cfg=dict(max_per_img=100)
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                           (736, 1333), (768, 1333), (800, 1333)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
                  [
                      dict(
                          type='Resize',
                          img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                          multiscale_mode='value',
                          keep_ratio=True),
                      dict(
                          type='RandomCrop',
                          crop_type='absolute_range',
                          crop_size=(384, 600),
                          allow_negative_crop=True),
                      dict(
                          type='Resize',
                          img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                     (576, 1333), (608, 1333), (640, 1333),
                                     (672, 1333), (704, 1333), (736, 1333),
                                     (768, 1333), (800, 1333)],
                          multiscale_mode='value',
                          override=True,
                          keep_ratio=True)
                  ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001*2,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
# optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[40])
runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(interval=10)
