_base_ = [
    '../_base_/datasets/mmm/2022-09-01/trucks1_lightsT_obstaclesF.py'
]


valid_mods=['mocap', 'zed_camera_left', 'zed_camera_depth', 'zed_camera_left_r50',
            'range_doppler', 'azimuth_static', 'mic_waveform',
            'realsense_camera_depth', 'realsense_camera_img']

valid_nodes=[1,2,3,4]

# data_root = 'data/mmm/2022-09-01/trucks0_lightsT_obstaclesF/train'

traincacher=dict(type='DataCacher',
    cache_dir='/dev/shm/cache_train/',
    num_future_frames=0,
    num_past_frames=9,
    valid_nodes=[1,2,3,4],
    valid_mods=['mocap', 'realsense_camera_r50'],
    include_z=False,
)

trainset=dict(type='HDF5Dataset',
    cacher_cfg=traincacher,
    # name='train',
    # uid=9234,
    num_future_frames=0,
    num_past_frames=9,
    # valid_nodes=[1,2,3,4],
    # valid_mods=['mocap', 'zed_camera_left_r50'],
    #include_z=False,
)


# data_root = 'data/mmm/2022-09-01/trucks1_lightsT_obstaclesF/test'


valset=dict(type='HDF5Dataset',
    cacher_cfg=dict(type='DataCacher',
        cache_dir='/dev/shm/cache_val/',
        valid_nodes=[1,2,3,4],
        valid_mods=['mocap', 'realsense_camera_r50', 'realsense_camera_img'],
        include_z=False,
        max_len=500,
    ),
    num_future_frames=0,
    num_past_frames=0,
    limit_axis=True,
    draw_cov=True,
)


checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth'  # noqa

r50_backbone_cfg=[
    dict(type='ChannelMapper',
        in_channels=[2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='BN'),
        num_outs=1
    ),
]

r50_model_cfg=dict(type='SingleModalityModel', ffn_cfg=None)

model_cfgs = {('realsense_camera_r50', 'node_1'): r50_model_cfg,
              ('realsense_camera_r50', 'node_2'): r50_model_cfg,
              ('realsense_camera_r50', 'node_3'): r50_model_cfg,
              ('realsense_camera_r50', 'node_4'): r50_model_cfg}
backbone_cfgs = {'realsense_camera_r50': r50_backbone_cfg}

model = dict(type='DecoderMocapModel',
        output_head_cfg=dict(type='OutputHead',
         include_z=False,
         predict_full_cov=True,
         cov_add=30.0,
         predict_rotation=True,
         predict_velocity=False,
         num_sa_layers=0,
         to_cm=True,
         mlp_dropout_rate=0.0
    ),
    model_cfgs=model_cfgs,
    backbone_cfgs=backbone_cfgs,
    track_eval=True,
    mse_loss_weight=0.0,
    max_age=5,
    grid_loss=True,
    # include_z=False,
    #mean_scale=[7,5],
    pos_loss_weight=1,
    # predict_full_cov=True,
    num_queries=1,
    # add_grid_to_mean=False,
    autoregressive=True,
    global_ca_layers=0,
    mod_dropout_rate=0.0,
    #match_by_id=True
)


orig_bs = 2
orig_lr = 1e-4 
factor = 4
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    shuffle=True, #trainset shuffle only
    train=trainset,
    val=valset,
    test=valset
)

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.0001,
    # paramwise_cfg=dict(
        # custom_keys={
            # 'backbone': dict(lr_mult=0.1),
            # 'sampling_offsets': dict(lr_mult=0.1),
            # 'reference_points': dict(lr_mult=0.1)
        # }
    # )
)

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
total_epochs = 50
lr_config = dict(policy='step', step=[40])
evaluation = dict(metric=['bbox', 'track'], interval=1e8)

find_unused_parameters = True

checkpoint_config = dict(interval=total_epochs)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
