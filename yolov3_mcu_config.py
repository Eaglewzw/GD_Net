# yolov3_mcu_config.py
# YOLOv3_McuNet 模型配置文件


cfg = {
    'loss_obj_weight': 1.0,
    'loss_cls_weight': 1.0,
    'loss_box_weight': 5.0,
    'iou_thresh':  0.5,
    'conf_thresh': 0.0001,
    'nms_thresh': 0.0001,
    ## Backbone
    # 'backbone': 'darknet53',
    'pretrained': True,
    'stride': [8, 16, 32],  # P3, P4, P5
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
    'fpn': 'yolov3_fpn',
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
    # 'anchor_size': [[8, 10], [24, 26], [41, 48],            # P3
    #                 [70, 33], [55, 74], [67, 100],          # P4
    #                 [97, 69], [84, 122], [153, 153]],       # P5
    'anchor_size': [[16, 21], [18, 24], [21, 21],     # P3 (小目标)
                    [22, 26], [24, 31], [26, 22],     # P4 (中目标)
                    [30, 28], [36, 33], [40, 43]],    # P5 (大目标)
}
