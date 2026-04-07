import torch
import torch.nn as nn

from utils.torch_utils import multiclass_nms

from .backbone_mcunet import build_mcunet_backbone
from .neck import build_neck
from .fpn import build_fpn
from .head import build_head
from .backbone_lsnet import build_lsnet_backbone


class YOLOv3_McuNet(nn.Module):
    def __init__(self, cfg, device, num_classes=1, conf_thresh=0.5, topk=100,
                 nms_thresh=0.5, trainable=False, deploy=False,
                 no_multi_labels=False, nms_class_agnostic=False):
        super().__init__()
        self.cfg               = cfg
        self.device            = device
        self.num_classes       = num_classes
        self.trainable         = trainable
        self.conf_thresh       = conf_thresh
        self.nms_thresh        = nms_thresh
        self.topk_candidates   = topk
        self.stride            = [8, 16, 32]
        self.deploy            = deploy
        self.no_multi_labels   = no_multi_labels
        self.nms_class_agnostic = nms_class_agnostic

        # ── Anchor boxes ──
        self.num_levels  = 3
        self.num_anchors = len(cfg['anchor_size']) // self.num_levels
        self.anchor_size = torch.as_tensor(cfg['anchor_size']).float().view(
            self.num_levels, self.num_anchors, 2)          # [S, A, 2]
        self._anchor_cache: dict = {}

        # ── Backbone ──
        backbone = cfg.get('backbone', 'mcunet-imagenet')
        if backbone.startswith('mcunet-'):
            model_type = backbone.split('-', 1)[1]   # 'vww' 或 'imagenet'
            self.backbone, feats_dim = build_mcunet_backbone(model_type)
        elif backbone.startswith('lsnet-'):
            lsnet_type = backbone.replace('-', '_')   # 'lsnet-t' -> 'lsnet_t'
            self.backbone, feats_dim = build_lsnet_backbone(lsnet_type, pretrained=True)
        else:
            raise ValueError(f"Unknown backbone: '{backbone}'. "
                             f"Use 'mcunet-vww', 'mcunet-imagenet', 'lsnet-t', 'lsnet-s', 'lsnet-b'")

        # ── Neck (SPPF) ──
        self.neck = build_neck(cfg, in_dim=feats_dim[-1], out_dim=feats_dim[-1])
        feats_dim[-1] = self.neck.out_dim

        # ── FPN ──
        self.fpn = build_fpn(cfg=cfg, in_dims=feats_dim, out_dim=None)
        self.head_dim = self.fpn.out_dim

        # ── Detection Heads ──
        self.non_shared_heads = nn.ModuleList(
            [build_head(cfg, d, d, num_classes) for d in self.head_dim])
        self.obj_preds = nn.ModuleList(
            [nn.Conv2d(h.reg_out_dim, 1 * self.num_anchors, 1) for h in self.non_shared_heads])
        self.cls_preds = nn.ModuleList(
            [nn.Conv2d(h.cls_out_dim, num_classes * self.num_anchors, 1) for h in self.non_shared_heads])
        self.reg_preds = nn.ModuleList(
            [nn.Conv2d(h.reg_out_dim, 4 * self.num_anchors, 1) for h in self.non_shared_heads])

    # ── Anchor cache ──
    def generate_anchors(self, level, fmp_size):
        fmp_h, fmp_w = fmp_size
        key = (level, fmp_h, fmp_w)
        if key in self._anchor_cache:
            return self._anchor_cache[key]

        anchor_size = self.anchor_size[level]
        anchor_y, anchor_x = torch.meshgrid(
            [torch.arange(fmp_h), torch.arange(fmp_w)], indexing='ij')
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2)
        anchor_xy = anchor_xy.unsqueeze(1).repeat(1, self.num_anchors, 1).view(-1, 2).to(self.device)
        anchor_wh = anchor_size.unsqueeze(0).repeat(fmp_h * fmp_w, 1, 1).view(-1, 2).to(self.device)

        anchors = torch.cat([anchor_xy, anchor_wh], dim=-1)
        self._anchor_cache[key] = anchors
        return anchors

    # ── Box decode ──
    def _decode_boxes(self, reg_pred, anchors, stride):
        ctr  = (torch.sigmoid(reg_pred[..., :2]) + anchors[..., :2]) * stride
        wh   = torch.exp(reg_pred[..., 2:]) * anchors[..., 2:]
        return torch.cat([ctr - wh * 0.5, ctr + wh * 0.5], dim=-1)

    # ── 公共检测头前向（train/inference 共用）──
    def _run_heads(self, pyramid_feats, batch_dim):
        """
        batch_dim=0: inference 模式（单张，squeeze batch 维）
        batch_dim=bs: training 模式（保留 batch 维）
        返回 (all_obj, all_cls, all_box, all_fmp_sizes)
        """
        all_obj, all_cls, all_box, all_fmp = [], [], [], []
        for level, (feat, head) in enumerate(zip(pyramid_feats, self.non_shared_heads)):
            cls_feat, reg_feat = head(feat)
            obj_pred = self.obj_preds[level](reg_feat)
            cls_pred = self.cls_preds[level](cls_feat)
            reg_pred = self.reg_preds[level](reg_feat)

            fmp_size = cls_pred.shape[-2:]
            anchors  = self.generate_anchors(level, fmp_size)

            if batch_dim == 0:
                # inference: [1, C, H, W] -> [M, C]
                obj_pred = obj_pred[0].permute(1, 2, 0).contiguous().view(-1, 1)
                cls_pred = cls_pred[0].permute(1, 2, 0).contiguous().view(-1, self.num_classes)
                reg_pred = reg_pred[0].permute(1, 2, 0).contiguous().view(-1, 4)
            else:
                # training: [B, C, H, W] -> [B, M, C]
                obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().view(batch_dim, -1, 1)
                cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(batch_dim, -1, self.num_classes)
                reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(batch_dim, -1, 4)

            box_pred = self._decode_boxes(reg_pred, anchors, self.stride[level])
            all_obj.append(obj_pred)
            all_cls.append(cls_pred)
            all_box.append(box_pred)
            all_fmp.append(fmp_size)
        return all_obj, all_cls, all_box, all_fmp

    # ── Post-process ──
    def post_process(self, obj_preds, cls_preds, box_preds):
        all_scores, all_labels, all_bboxes = [], [], []
        for obj_pred_i, cls_pred_i, box_pred_i in zip(obj_preds, cls_preds, box_preds):
            if self.no_multi_labels:
                scores, labels = torch.max(
                    torch.sqrt(obj_pred_i.sigmoid() * cls_pred_i.sigmoid()), dim=1)
                num_topk = min(self.topk_candidates, box_pred_i.size(0))
                predicted_prob, topk_idxs = scores.sort(descending=True)
                topk_scores = predicted_prob[:num_topk]
                topk_idxs   = topk_idxs[:num_topk]
                keep_idxs   = topk_scores > self.conf_thresh
                scores      = topk_scores[keep_idxs]
                topk_idxs   = topk_idxs[keep_idxs]
                labels       = labels[topk_idxs]
                bboxes       = box_pred_i[topk_idxs]
            else:
                scores_i  = torch.sqrt(obj_pred_i.sigmoid() * cls_pred_i.sigmoid()).flatten()
                num_topk  = min(self.topk_candidates, box_pred_i.size(0))
                predicted_prob, topk_idxs = scores_i.sort(descending=True)
                topk_scores = predicted_prob[:num_topk]
                topk_idxs   = topk_idxs[:num_topk]
                keep_idxs   = topk_scores > self.conf_thresh
                scores      = topk_scores[keep_idxs]
                topk_idxs   = topk_idxs[keep_idxs]
                anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
                labels       = topk_idxs % self.num_classes
                bboxes       = box_pred_i[anchor_idxs]

            all_scores.append(scores)
            all_labels.append(labels)
            all_bboxes.append(bboxes)

        scores = torch.cat(all_scores).cpu().numpy()
        labels = torch.cat(all_labels).cpu().numpy()
        bboxes = torch.cat(all_bboxes).cpu().numpy()
        return multiclass_nms(scores, labels, bboxes, self.nms_thresh,
                              self.num_classes, self.nms_class_agnostic)

    # ── Inference ──
    @torch.no_grad()
    def inference(self, x):
        pyramid_feats = self.backbone(x)
        pyramid_feats[-1] = self.neck(pyramid_feats[-1])
        pyramid_feats = self.fpn(pyramid_feats)

        all_obj, all_cls, all_box, _ = self._run_heads(pyramid_feats, batch_dim=0)

        if self.deploy:
            obj = torch.cat(all_obj, dim=0)
            cls = torch.cat(all_cls, dim=0)
            box = torch.cat(all_box, dim=0)
            scores = torch.sqrt(obj.sigmoid() * cls.sigmoid())
            return torch.cat([box, scores], dim=-1)

        scores, labels, bboxes = self.post_process(all_obj, all_cls, all_box)
        return {"scores": scores, "labels": labels, "bboxes": bboxes}

    # ── Training forward ──
    def forward(self, x):
        if not self.trainable:
            return self.inference(x)

        bs = x.shape[0]
        pyramid_feats = self.backbone(x)
        pyramid_feats[-1] = self.neck(pyramid_feats[-1])
        pyramid_feats = self.fpn(pyramid_feats)

        all_obj, all_cls, all_box, all_fmp = self._run_heads(pyramid_feats, batch_dim=bs)

        return {
            "pred_obj":  all_obj,
            "pred_cls":  all_cls,
            "pred_box":  all_box,
            "fmp_sizes": all_fmp,
            "strides":   self.stride,
        }
