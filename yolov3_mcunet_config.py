# YOLOv3 Config

yolov3_mcunet_cfg = {
    'yolov3_mcunet_in1':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'darknet53',
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
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],     # P3
                        [30, 61],   [62, 45],   [59, 119],    # P4
                        [116, 90],  [156, 198], [373, 326]],  # P5
        # ---------------- Data process config ----------------
        'trans_type': 'yolo_l',
        'multi_scale': [0.5, 1.25],  # 320 -> 800
        # ---------------- Assignment config ----------------
        ## matcher
        'iou_thresh': 0.5,
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        # ---------------- Train config ----------------
        'trainer_type': 'yolo',
    },

    'yolov3_mcunet_in2':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'darknet53',
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
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],     # P3
                        [30, 61],   [62, 45],   [59, 119],    # P4
                        [116, 90],  [156, 198], [373, 326]],  # P5
        # ---------------- Data process config ----------------
        'trans_type': 'yolo_l',
        'multi_scale': [0.5, 1.25],  # 320 -> 800
        # ---------------- Assignment config ----------------
        ## matcher
        'iou_thresh': 0.5,
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        # ---------------- Train config ----------------
        'trainer_type': 'yolo',
    },

    'yolov3_mcunet_in3':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'darknet53',
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
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],     # P3
                        [30, 61],   [62, 45],   [59, 119],    # P4
                        [116, 90],  [156, 198], [373, 326]],  # P5
        # ---------------- Data process config ----------------
        'trans_type': 'yolo_l',
        'multi_scale': [0.5, 1.25],  # 320 -> 800
        # ---------------- Assignment config ----------------
        ## matcher
        'iou_thresh': 0.5,
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        # ---------------- Train config ----------------
        'trainer_type': 'yolo',
    },

    'yolov3_mcunet_in4':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'darknet53',
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
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],     # P3
                        [30, 61],   [62, 45],   [59, 119],    # P4
                        [116, 90],  [156, 198], [373, 326]],  # P5
        # ---------------- Data process config ----------------
        'trans_type': 'yolo_l',
        'multi_scale': [0.5, 1.25],  # 320 -> 800
        # ---------------- Assignment config ----------------
        ## matcher
        'iou_thresh': 0.5,
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        # ---------------- Train config ----------------
        'trainer_type': 'yolo',
    },

    'yolov3_mcunet_vww0':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'darknet53',
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
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],     # P3
                        [30, 61],   [62, 45],   [59, 119],    # P4
                        [116, 90],  [156, 198], [373, 326]],  # P5
        # ---------------- Data process config ----------------
        'trans_type': 'yolo_l',
        'multi_scale': [0.5, 1.25],  # 320 -> 800
        # ---------------- Assignment config ----------------
        ## matcher
        'iou_thresh': 0.5,
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        # ---------------- Train config ----------------
        'trainer_type': 'yolo',
    },

    'yolov3_mcunet_vww1':{
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'darknet53',
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
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],     # P3
                        [30, 61],   [62, 45],   [59, 119],    # P4
                        [116, 90],  [156, 198], [373, 326]],  # P5
        # ---------------- Data process config ----------------
        'trans_type': 'yolo_l',
        'multi_scale': [0.5, 1.25],  # 320 -> 800
        # ---------------- Assignment config ----------------
        ## matcher
        'iou_thresh': 0.5,
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        # ---------------- Train config ----------------
        'trainer_type': 'yolo',
    },

}