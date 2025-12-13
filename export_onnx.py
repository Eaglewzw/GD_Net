import torch
import torch.onnx
import os
from gd_net.yolov3_mcu_net import YOLOv3_McuNet
import warnings

warnings.filterwarnings("ignore")  # 全局关闭警告

# 关闭 torch.meshgrid 未来警告
os.environ["TORCH_DEPRECATED_MESHGRID_INDEXING"] = "1"

class YOLOv3_McuNet_For_ONNX(YOLOv3_McuNet):
    """
    专为ONNX导出设计的模型类，修改post_process函数以避免使用numpy
    """
    def post_process(self, obj_preds, cls_preds, box_preds):
        """
        与原post_process相同，但返回torch张量而不是numpy数组
        """
        assert len(cls_preds) == self.num_levels
        all_scores = []
        all_labels = []
        all_bboxes = []

        for obj_pred_i, cls_pred_i, box_pred_i in zip(obj_preds, cls_preds, box_preds):
            if self.no_multi_labels:
                # [M,]
                scores, labels = torch.max(torch.sqrt(obj_pred_i.sigmoid() * cls_pred_i.sigmoid()), dim=1)

                # Keep top k top scoring indices only.
                num_topk = min(self.topk_candidates, box_pred_i.size(0))

                # topk candidates
                predicted_prob, topk_idxs = scores.sort(descending=True)
                topk_scores = predicted_prob[:num_topk]
                topk_idxs = topk_idxs[:num_topk]

                # filter out the proposals with low confidence score
                keep_idxs = topk_scores > self.conf_thresh
                scores = topk_scores[keep_idxs]
                topk_idxs = topk_idxs[keep_idxs]

                labels = labels[topk_idxs]
                bboxes = box_pred_i[topk_idxs]
            else:
                # [M, C] -> [MC,]
                scores_i = (torch.sqrt(obj_pred_i.sigmoid() * cls_pred_i.sigmoid())).flatten()

                # Keep top k top scoring indices only.
                num_topk = min(self.topk_candidates, box_pred_i.size(0))

                # torch.sort is actually faster than .topk (at least on GPUs)
                predicted_prob, topk_idxs = scores_i.sort(descending=True)
                topk_scores = predicted_prob[:num_topk]
                topk_idxs = topk_idxs[:num_topk]

                # filter out the proposals with low confidence score
                keep_idxs = topk_scores > self.conf_thresh
                scores = topk_scores[keep_idxs]
                topk_idxs = topk_idxs[keep_idxs]

                anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
                labels = topk_idxs % self.num_classes

                bboxes = box_pred_i[anchor_idxs]

            all_scores.append(scores)
            all_labels.append(labels)
            all_bboxes.append(bboxes)

        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        bboxes = torch.cat(all_bboxes)
        
        # 这里返回的是torch张量，而不是numpy数组
        return bboxes, scores, labels

    @torch.no_grad()
    def inference(self, x):
        # 主干网络
        pyramid_feats = self.backbone(x)

        # 颈部网络
        pyramid_feats[-1] = self.neck(pyramid_feats[-1])

        # 特征金字塔
        pyramid_feats = self.fpn(pyramid_feats)

        # 检测头
        all_anchors = []
        all_obj_preds = []
        all_cls_preds = []
        all_box_preds = []
        for level, (feat, head) in enumerate(zip(pyramid_feats, self.non_shared_heads)):
            cls_feat, reg_feat = head(feat)

            # [1, C, H, W]
            obj_pred = self.obj_preds[level](reg_feat)
            cls_pred = self.cls_preds[level](cls_feat)
            reg_pred = self.reg_preds[level](reg_feat)

            # anchors: [M, 2]
            fmp_size = cls_pred.shape[-2:]
            anchors = self.generate_anchors(level, fmp_size)

            # [1, AC, H, W] -> [H, W, AC] -> [M, C]
            obj_pred = obj_pred[0].permute(1, 2, 0).contiguous().view(-1, 1)
            cls_pred = cls_pred[0].permute(1, 2, 0).contiguous().view(-1, self.num_classes)
            reg_pred = reg_pred[0].permute(1, 2, 0).contiguous().view(-1, 4)

            # decode bbox
            ctr_pred = (torch.sigmoid(reg_pred[..., :2]) + anchors[..., :2]) * self.stride[level]
            wh_pred = torch.exp(reg_pred[..., 2:]) * anchors[..., 2:]
            pred_x1y1 = ctr_pred - wh_pred * 0.5
            pred_x2y2 = ctr_pred + wh_pred * 0.5
            box_pred = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

            all_obj_preds.append(obj_pred)
            all_cls_preds.append(cls_pred)
            all_box_preds.append(box_pred)
            all_anchors.append(anchors)

        if self.deploy:
            obj_preds = torch.cat(all_obj_preds, dim=0)
            cls_preds = torch.cat(all_cls_preds, dim=0)
            box_preds = torch.cat(all_box_preds, dim=0)
            scores = torch.sqrt(obj_preds.sigmoid() * cls_preds.sigmoid())
            bboxes = box_preds
            # [n_anchors_all, 4 + C]
            outputs = torch.cat([bboxes, scores], dim=-1)

        else:
            # post process - 返回torch张量而非numpy
            bboxes, scores, labels = self.post_process(
                all_obj_preds, all_cls_preds, all_box_preds)
            # 保持为张量格式
            outputs = {
                "scores": scores,
                "labels": labels,
                "bboxes": bboxes
            }

        return outputs

def convert_pth_to_onnx(pth_path, onnx_path, img_size=640, num_classes=1, device='cpu', deploy_mode=True):
    """
    将训练好的 YOLOv3_McuNet .pth 模型转换为 ONNX 格式

    Args:
        pth_path: 训练好的 .pth 模型路径
        onnx_path: 输出的 .onnx 文件路径
        img_size: 输入图像尺寸
        num_classes: 类别数量
        device: 设备 ('cpu' 或 'cuda')
        deploy_mode: 是否为部署模式
    """

    # 1. 加载模型配置（与训练时相同）
    cfg = {
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        'iou_thresh': 0.5,
        'pretrained': False,  # 转换时不需要预训练权重
        'stride': [8, 16, 32],
        'width': 1.0,
        'depth': 1.0,
        'max_stride': 32,
        'neck': 'sppf',
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'fpn': 'yolov3_fpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        'head': 'decoupled_head',
        'head_act': 'silu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        'anchor_size': [[16, 21], [18, 24], [21, 21],     # P3 (小目标)
                        [22, 26], [24, 31], [26, 22],     # P4 (中目标)
                        [30, 28], [36, 33], [40, 43]],    # P5 (大目标)
    }

    # 2. 创建模型实例 - 使用修改后的类
    model = YOLOv3_McuNet_For_ONNX(cfg, device, num_classes=num_classes, trainable=False, deploy=deploy_mode).to(device)

    # 3. 加载训练好的权重
    if os.path.exists(pth_path):
        state_dict = torch.load(pth_path, map_location=device)

        # 处理可能的权重键名不匹配
        if 'model' in state_dict:
            state_dict = state_dict['model']

        # 移除可能存在的 'module.' 前缀（如果是多GPU训练保存的）
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        print(f"✅ 成功加载模型权重: {pth_path}")
    else:
        raise FileNotFoundError(f"❌ 找不到模型文件: {pth_path}")

    # 4. 设置为评估模式
    model.eval()
    print("✅ 模型设置为评估模式")

    # 5. 创建虚拟输入
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=device)
    print(f"✅ 创建虚拟输入: {dummy_input.shape}")

    # 6. 前向传播一次以确认模型正常工作
    with torch.no_grad():
        outputs = model(dummy_input)
        print("✅ 模型前向传播测试成功")
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                print(f"   {key}: {value.shape}")
        else:
            print(f"   输出类型: {type(outputs)}, 形状: {outputs.shape if hasattr(outputs, 'shape') else 'N/A'}")

    # 7. 导出为ONNX格式
    print("🚀 开始导出ONNX模型...")

    # 根据模型在部署模式下的输出调整导出参数
    if deploy_mode:
        # 在部署模式下，模型输出为 [n_anchors_all, 4 + C] 的张量
        torch.onnx.export(
            model,                    # 要转换的模型
            dummy_input,              # 模型输入
            onnx_path,                # 输出ONNX文件路径
            export_params=True,       # 将模型参数一起导出
            opset_version=18,         # ONNX算子集版本（推荐11或更高）
            do_constant_folding=True, # 是否执行常量折叠优化
            input_names=['input'],    # 输入节点名称
            output_names=['output'],  # 输出节点名称（部署模式下是单个张量）
            dynamic_axes={
                'input': {0: 'batch_size'},  # 批处理维度
                'output': {0: 'num_anchors'}, # 锚框数量维度
            },
            verbose=False             # 是否打印详细信息
        )
    else:
        # 在非部署模式下，模型输出为字典，需要进一步处理
        torch.onnx.export(
            model,                    # 要转换的模型
            dummy_input,              # 模型输入
            onnx_path,                # 输出ONNX文件路径
            export_params=True,       # 将模型参数一起导出
            opset_version=18,         # ONNX算子集版本（推荐11或更高）
            do_constant_folding=True, # 是否执行常量折叠优化
            input_names=['input'],    # 输入节点名称
            output_names=['output'],  # 为简化，暂时使用单个输出
            dynamic_axes={
                'input': {0: 'batch_size'},  # 批处理维度
            },
            external_data_format=False,
            verbose=False             # 是否打印详细信息
        )

    print(f"✅ ONNX模型导出成功: {onnx_path}")

    return onnx_path

def verify_onnx_model(onnx_path, img_size=640, device='cpu'):
    """
    验证导出的ONNX模型
    """
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np

        # 1. 检查ONNX模型格式
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX模型格式检查通过")

        # 2. 打印模型信息
        print("\n📊 ONNX模型信息:")
        print(f"   输入: {[input.name for input in onnx_model.graph.input]}")
        print(f"   输出: {[output.name for output in onnx_model.graph.output]}")

        # 3. 使用ONNX Runtime验证推理
        ort_session = ort.InferenceSession(onnx_path)

        # 准备输入数据
        dummy_input_np = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

        # 运行推理
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input_np}
        ort_outputs = ort_session.run(None, ort_inputs)

        print("✅ ONNX Runtime推理成功")
        for i, output in enumerate(ort_outputs):
            print(f"   输出 {i}: 形状 {output.shape}, 数据类型 {output.dtype}")

        return True

    except Exception as e:
        print(f"❌ ONNX模型验证失败: {e}")
        return False

# 使用示例
if __name__ == "__main__":
    # 配置参数
    pth_model_path = "./checkpoints/best_yolov3_mcu.pth"  # 训练好的模型路径
    onnx_output_path = "./checkpoints/yolov3_mcu.onnx"    # 输出的ONNX文件路径
    img_size = 256
    num_classes = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("🔧 开始转换PTH模型到ONNX格式...")
    print(f"   设备: {device}")
    print(f"   输入尺寸: {img_size}x{img_size}")
    print(f"   类别数: {num_classes}")

    try:
        # 执行转换
        onnx_path = convert_pth_to_onnx(
            pth_path=pth_model_path,
            onnx_path=onnx_output_path,
            img_size=img_size,
            num_classes=num_classes,
            device=device
        )

        # 验证转换结果
        print("\n🔍 开始验证ONNX模型...")
        verify_onnx_model(onnx_path, img_size, device)

        print(f"\n🎉 转换完成！ONNX模型已保存至: {onnx_path}")

    except Exception as e:
        print(f"❌ 转换过程中出现错误: {e}")
        import traceback
        traceback.print_exc()