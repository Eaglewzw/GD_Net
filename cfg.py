# yolov3_mcu_config.py
# YOLOv3_McuNet 模型配置文件


cfg = {
    ## Loss weights
    'loss_obj_weight': 1.0,
    'loss_cls_weight': 1.0,
    'loss_box_weight': 5.0,
    'obj_pos_weight': 5.0,

    ## Thresholds
    # iou_thresh: anchor-to-GT 匹配时使用，训练阶段生效
    'iou_thresh': 0.25,
    # conf_thresh / nms_thresh: 推理后处理阈值，评估/推理时生效
    'conf_thresh': 0.25,
    'nms_thresh': 0.25,

    ## Backbone
    'backbone_type': 'mcunet',   # 'mcunet' | 'lsnet'
    'lsnet_type': 'lsnet_t',    # 'lsnet_t' | 'lsnet_s' | 'lsnet_b'
    'pretrained': True,
    'stride': [8, 16, 32],      # P3, P4, P5
    'width': 1.0,
    'depth': 1.0,
    'max_stride': 32,

    ## Neck
    'neck': 'sppf',
    'neck_act': 'silu',
    'neck_norm': 'BN',
    'neck_depthwise': False,
    'expand_ratio': 0.5,
    'pooling_size': 5,

    ## FPN
    'fpn': 'yolov3_panet',
    'fpn_act': 'silu',
    'fpn_norm': 'BN',
    'fpn_depthwise': False,

    ## Head
    'head': 'decoupled_head',
    'head_act': 'silu',
    'head_norm': 'BN',
    'num_cls_head': 2,
    'num_reg_head': 2,
    'head_depthwise': False,

    ## Anchors (9 total, 3 per level: P3/P4/P5)
    # 当前（适配 160px）
    'anchor_size': [[16, 21], [18, 24], [21, 21],
                    [22, 26], [24, 31], [26, 22],
                    [30, 28], [36, 33], [40, 43]],

    # 若改为 320px（大约 ×2）
    # 'anchor_size': [[32, 42], [36, 48], [42, 42],
    #                 [44, 52], [48, 62], [52, 44],
    #                 [60, 56], [72, 66], [80, 86]],
}
