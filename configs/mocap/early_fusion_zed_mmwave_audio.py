_base_ = [
    '../_base_/datasets/one_car_early_fusion.py'
]

valid_mods=['mocap', 'zed_camera_left','range_doppler', 'mic_waveform', "realsense_camera_depth"]

trainset=dict(type='HDF5Dataset',
    cacher_cfg=dict(type='DataCacher',
        cache_dir='/dev/shm/cache_train/',
        num_future_frames=0,
        num_past_frames=9,
        valid_nodes=[1,2,3],
        valid_mods=valid_mods,
        include_z=False,
    ),
    num_future_frames=0,
    num_past_frames=1,
)

valset=dict(type='HDF5Dataset',
    cacher_cfg=dict(type='DataCacher',
        cache_dir='/dev/shm/cache_val/',
        valid_nodes=[1,2,3],
        valid_mods=valid_mods,
        include_z=False,
    ),
    num_future_frames=0,
    num_past_frames=0,
    limit_axis=True,
    draw_cov=True,
)

testset=dict(type='HDF5Dataset',
    cacher_cfg=dict(type='DataCacher',
        cache_dir='/dev/shm/cache_test/',
        valid_nodes=[1,2,3],
        valid_mods=valid_mods,
        include_z=False,
    ),
    num_future_frames=0,
    num_past_frames=0,
    limit_axis=True,
    draw_cov=True,
)

r50_cfg=[
    dict(type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    dict(type='ChannelMapper',
        in_channels=[2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=1
    )
]

depth_r50_cfg=[
    dict(type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    dict(type='ChannelMapper',
        in_channels=[2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=1
    )
]

backbone_cfgs = {
    'zed_camera_left': r50_cfg, 
    "realsense_camera_depth": depth_r50_cfg,
    'range_doppler': dict(type='mmWaveBackbone'),
    'mic_waveform': dict(type='AudioBackbone'),
}

depth_cfg=dict(type='LinearEncoder', in_len=108, out_len=108, use_pos_encodings=True, ffn_cfg=dict(type='SLP', in_channels=512))
zed_cfg=dict(type='LinearEncoder', in_len=135, out_len=135, use_pos_encodings=True, ffn_cfg=dict(type='SLP', in_channels=512))
rdoppler_cfg=dict(type='LinearEncoder', in_len=16, out_len=16, use_pos_encodings=True, ffn_cfg=dict(type='SLP', in_channels=512))
audio_cfg=dict(type='LinearEncoder', in_len=16, out_len=16, use_pos_encodings=True, ffn_cfg=dict(type='SLP', in_channels=512))

adapter_cfgs = {
    ('realsense_camera_depth', 'node_1'): depth_cfg,
    ('realsense_camera_depth', 'node_2'): depth_cfg,
    ('realsense_camera_depth', 'node_3'): depth_cfg,
    ('zed_camera_left', 'node_1'): zed_cfg,
    ('zed_camera_left', 'node_2'): zed_cfg,
    ('zed_camera_left', 'node_3'): zed_cfg,
 #   ('zed_camera_left', 'node_4'): zed_cfg,
    ('range_doppler', 'node_1'): rdoppler_cfg,
    ('range_doppler', 'node_2'): rdoppler_cfg,
    ('range_doppler', 'node_3'): rdoppler_cfg,
 #   ('range_doppler', 'node_4'): rdoppler_cfg,
    ('mic_waveform', 'node_1'): audio_cfg,
    ('mic_waveform', 'node_2'): audio_cfg,
    ('mic_waveform', 'node_3'): audio_cfg,
 #   ('mic_waveform', 'node_4'): audio_cfg

}


model = dict(type='EarlyFusion',
        output_head_cfg=dict(type='OutputHead',
         include_z=False,
         predict_full_cov=True,
         cov_add=1.0,
         input_dim=512,
         predict_rotation=True,
         predict_velocity=False,
         num_sa_layers=0,
         to_cm=True,
         mlp_dropout_rate=0.0
    ),
    model_cfgs=adapter_cfgs,
    backbone_cfgs=backbone_cfgs,
    track_eval=True,
    pos_loss_weight=1,
    num_queries=1,
    mod_dropout_rate=0.0,
    loss_type='nll',
    global_ca_layers=6

)


# orig_bs = 2
# orig_lr = 1e-4 
# factor = 4
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    shuffle=True, #trainset shuffle only
    train=trainset,
    val=valset,
    test=testset
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
total_epochs = 40
lr_config = dict(policy='step', step=[40])
evaluation = dict(metric=['bbox', 'track'], interval=1e8)

find_unused_parameters = True

checkpoint_config = dict(interval=10)
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
