import numpy as np
import torch
from . import config

from .nms.nms_wrapper import batched_nms


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):

    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.get('typename', 'nms')
    if nms_type.startswith('lb_'):
        nms_cfg_['typename'] = nms_type[3:]
        return _lb_multiclass_nms(multi_bboxes, multi_scores, score_thr,
                                  nms_cfg_, max_num, score_factors)
    else:
        return _multiclass_nms(multi_bboxes, multi_scores, score_thr, nms_cfg_,
                               max_num, score_factors)


def _multiclass_nms(multi_bboxes,
                    multi_scores,
                    score_thr,
                    nms_cfg,
                    max_num=-1,
                    score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 4)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    scores = scores[valid_mask]
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        return bboxes, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep]


def _lb_multiclass_nms(multi_bboxes,
                       multi_scores,
                       score_thr,
                       nms_cfg,
                       max_num=-1,
                       score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # (TODO): as ONNX does not support repeat now,
    # we have to use this ugly code
    bboxes = torch.masked_select(
        bboxes,
        torch.stack((valid_mask, valid_mask, valid_mask, valid_mask),
                    -1)).view(-1, 4)
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        return bboxes, labels

    inds = scores.argsort(descending=True)
    bboxes = bboxes[inds]
    scores = scores[inds]
    labels = labels[inds]

    batch_bboxes = torch.empty((0, 4),
                               dtype=bboxes.dtype,
                               device=bboxes.device)
    batch_scores = torch.empty((0, ), dtype=scores.dtype, device=scores.device)
    batch_labels = torch.empty((0, ), dtype=labels.dtype, device=labels.device)
    while bboxes.shape[0] > 0:
        num = min(100000, bboxes.shape[0])
        batch_bboxes = torch.cat([batch_bboxes, bboxes[:num]])
        batch_scores = torch.cat([batch_scores, scores[:num]])
        batch_labels = torch.cat([batch_labels, labels[:num]])
        bboxes = bboxes[num:]
        scores = scores[num:]
        labels = labels[num:]

        _, keep = batched_nms(batch_bboxes, batch_scores, batch_labels,
                              nms_cfg)
        batch_bboxes = batch_bboxes[keep]
        batch_scores = batch_scores[keep]
        batch_labels = batch_labels[keep]

    dets = torch.cat([batch_bboxes, batch_scores[:, None]], dim=-1)
    labels = batch_labels

    if max_num > 0:
        dets = dets[:max_num]
        labels = labels[:max_num]

    return dets, labels


def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]


def bbox_flip(bboxes, img_shape, direction='horizontal'):
    """Flip bboxes horizontally or vertically.
    Args:
        bboxes (Tensor): Shape (..., 4*k)
        img_shape (tuple): Image shape.
        direction (str): Flip direction, options are "horizontal", "vertical",
            "diagonal". Default: "horizontal"
    Returns:
        Tensor: Flipped bboxes.
    """
    assert bboxes.shape[-1] % 4 == 0
    assert direction in ['horizontal', 'vertical', 'diagonal']
    flipped = bboxes.clone()
    if direction == 'horizontal':
        flipped[..., 0::4] = img_shape[1] - bboxes[..., 2::4]
        flipped[..., 2::4] = img_shape[1] - bboxes[..., 0::4]
    elif direction == 'vertical':
        flipped[..., 1::4] = img_shape[0] - bboxes[..., 3::4]
        flipped[..., 3::4] = img_shape[0] - bboxes[..., 1::4]
    else:
        flipped[..., 0::4] = img_shape[1] - bboxes[..., 2::4]
        flipped[..., 1::4] = img_shape[0] - bboxes[..., 3::4]
        flipped[..., 2::4] = img_shape[1] - bboxes[..., 0::4]
        flipped[..., 3::4] = img_shape[0] - bboxes[..., 1::4]
    return flipped


def bbox_revert(bboxes,
                img_shape,
                scale_factor,
                flip,
                flip_direction='horizontal'):
    """Map bboxes from testing scale to original image scale."""
    new_bboxes = bbox_flip(bboxes, img_shape,
                           flip_direction) if flip else bboxes
    new_bboxes = new_bboxes.view(-1, 4) / new_bboxes.new_tensor(scale_factor)
    return new_bboxes.view(bboxes.shape)


class BBoxBaseAnchor:

    def __init__(self,
                 ratios,
                 scales=None,
                 base_sizes=None,
                 scale_major=True,
                 octave_base_scale=None,
                 scales_per_octave=None,
                 centers=None,
                 center_offset=0.):
        """Generate base anchors.

            list(torch.Tensor): Base anchors of a feature grid in multiple
                feature levels.
        """
        self.ratios = torch.Tensor(ratios)
        if center_offset != 0:
            assert centers is None, 'center cannot be set when center_offset' \
                f'!=0, {centers} is given.'
        if not (0 <= center_offset <= 1):
            raise ValueError('center_offset should be in range [0, 1], '
                             f'{center_offset} is given.')
        if centers is not None:
            assert len(centers) == len(base_sizes), \
                'The number of base_sizes should be the same as centers, ' \
                f'got {base_sizes} and {centers}'

        # calculate scales of anchors
        assert ((octave_base_scale is not None
                and scales_per_octave is not None) ^ (scales is not None)), \
            'scales and octave_base_scale with scales_per_octave cannot' \
            ' be set at the same time'
        if scales is not None:
            self.scales = torch.Tensor(scales)
        elif octave_base_scale is not None and scales_per_octave is not None:
            octave_scales = np.array(
                [2**(i / scales_per_octave) for i in range(scales_per_octave)])
            scales = octave_scales * octave_base_scale
            self.scales = torch.Tensor(scales)
        else:
            raise ValueError('Either scales or octave_base_scale with '
                             'scales_per_octave should be set')

        self.base_sizes = base_sizes
        self.scale_major = scale_major
        self.centers = centers
        self.center_offset = center_offset

    def generate(self):
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self._single_level_base_anchors(base_size, self.scales,
                                                self.ratios,
                                                self.center_offset,
                                                self.scale_major, center))
        return multi_level_base_anchors

    @staticmethod
    def _single_level_base_anchors(base_size,
                                   scales,
                                   ratios,
                                   center_offset,
                                   scale_major,
                                   center=None):
        """Generate base anchors of a single level.

        Args:
            base_size (int | float): Basic size of an anchor.
            scales (torch.Tensor): Scales of the anchor.
            ratios (torch.Tensor): The ratio between between the height
                and width of anchors in a single level.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.

        Returns:
            torch.Tensor: Anchors in a single-level feature maps
        """
        w = base_size
        h = base_size
        if center is None:
            x_center = center_offset * w
            y_center = center_offset * h
        else:
            x_center, y_center = center

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if scale_major:
            ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
            y_center + 0.5 * hs
        ]
        base_anchors = torch.stack(base_anchors, dim=-1)

        return base_anchors


class BBoxAnchorMeshGrid:

    def __init__(self, strides, base_anchor):
        assert len(strides) > 0
        self.strides = strides

        kwargs = config.get_kwargs(base_anchor, 'BBoxBaseAnchor')
        self.base_anchors = BBoxBaseAnchor(**kwargs).generate()

    def gen_anchor_mesh(self,
                        featmap_sizes,
                        img_metas,
                        dtype=torch.float,
                        device='cuda'):
        """Get anchors according to feature map sizes.
        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors
        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self._gen_anchor_mesh(featmap_sizes, dtype,
                                                    device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.valid_flags(featmap_sizes,
                                                 img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def _gen_anchor_mesh(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.
        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.
        Returns:
            tuple: points of each image.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self._single_level_anchor_mesh(
                self.base_anchors[i].to(device).to(dtype),
                featmap_sizes[i],
                self.strides[i],
                device=device)
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def _single_level_anchor_mesh(self, base_anchors, featmap_size, stride,
                                  device):
        """Generate grid anchors of a single level.

        Note:
            This function is usually called by method ``self.grid_anchors``.

        Args:
            base_anchors (torch.Tensor): The base anchors of a feature grid.
            featmap_size (tuple[int]): Size of the feature maps.
            stride (tuple[int], optional): Stride of the feature map.
                Defaults to (16, 16).
            device (str, optional): Device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors in the overall feature maps.
        """
        feat_h, feat_w = featmap_size
        # shift_x = torch.arange(0, feat_w, device=device) * stride[0]
        # shift_y = torch.arange(0, feat_h, device=device) * stride[1]
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def _meshgrid(self, x, y, row_major=True):
        """Generate mesh grid of x and y.

        Args:
            x (torch.Tensor): Grids of x dimension.
            y (torch.Tensor): Grids of y dimension.
            row_major (bool, optional): Whether to return y grids first.
                Defaults to True.

        Returns:
            tuple[torch.Tensor]: The mesh grids of x and y.
        """
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def valid_flags(self, featmap_sizes, pad_shape, device='cuda'):
        """Generate valid flags of anchors in multiple feature levels.

        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in
                multiple feature levels.
            pad_shape (tuple): The padded shape of the image.
            device (str): Device where the anchors will be put on.

        Return:
            list(torch.Tensor): Valid flags of anchors in multiple levels.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            anchor_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            # valid_feat_h = min(int(np.ceil(h / anchor_stride[0])), feat_h)
            # valid_feat_w = min(int(np.ceil(w / anchor_stride[1])), feat_w)
            valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
            valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
            flags = self._single_level_valid_flags(
                (feat_h, feat_w), (valid_feat_h, valid_feat_w),
                self.num_base_anchors[i],
                device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    def _single_level_valid_flags(self,
                                  featmap_size,
                                  valid_size,
                                  num_base_anchors,
                                  device='cuda'):
        """Generate the valid flags of anchor in a single feature map.

        Args:
            featmap_size (tuple[int]): The size of feature maps.
            valid_size (tuple[int]): The valid size of the feature maps.
            num_base_anchors (int): The number of base anchors.
            device (str, optional): Device where the flags will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: The valid flags of each anchor in a single level
                feature map.
        """
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(valid.size(0),
                                      num_base_anchors).contiguous().view(-1)
        return valid

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    @property
    def num_base_anchors(self):
        """list[int]: total number of base anchors in a feature grid"""
        return [base_anchors.size(0) for base_anchors in self.base_anchors]


class DeltaXYWHBBoxCoder:
    """Delta XYWH BBox coder.

    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`_,
    this coder encodes bbox (x1, y1, x2, y2) into delta (dx, dy, dw, dh) and
    decodes delta (dx, dy, dw, dh) back to original bbox (x1, y1, x2, y2).

    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
    """

    def __init__(self,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1.)):
        self.means = target_means
        self.stds = target_stds

    def encode(self, bboxes, gt_bboxes):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor): Source boxes, e.g., object proposals.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """

        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 4
        encoded_bboxes = bbox2delta(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            boxes (torch.Tensor): Basic boxes.
            pred_bboxes (torch.Tensor): Encoded boxes with shape
            max_shape (tuple[int], optional): Maximum shape of boxes.
                Defaults to None.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.

        Returns:
            torch.Tensor: Decoded boxes.
        """

        assert pred_bboxes.size(0) == bboxes.size(0)
        decoded_bboxes = delta2bbox(bboxes, pred_bboxes, self.means, self.stds,
                                    max_shape, wh_ratio_clip)

        return decoded_bboxes


def bbox2delta(proposals, gt, means=(0., 0., 0., 0.), stds=(1., 1., 1., 1.)):
    """Compute deltas of proposals w.r.t. gt.

    We usually compute the deltas of x, y, w, h of proposals w.r.t ground
    truth bboxes to get regression target.
    This is the inverse function of :func:`delta2bbox`.

    Args:
        proposals (Tensor): Boxes to be transformed, shape (N, ..., 4)
        gt (Tensor): Gt bboxes to be used as base, shape (N, ..., 4)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates

    Returns:
        Tensor: deltas with shape (N, 4), where columns represent dx, dy,
            dw, dh.
    """
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0]
    gh = gt[..., 3] - gt[..., 1]

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2bbox(rois,
               deltas,
               means=(0., 0., 0., 0.),
               stds=(1., 1., 1., 1.),
               max_shape=None,
               wh_ratio_clip=16 / 1000):
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of :func:`bbox2delta`.

    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 4)
        deltas (Tensor): Encoded offsets with respect to each roi.
            Has shape (N, 4 * num_classes). Note N = num_anchors * W * H when
            rois is a grid of anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates
        max_shape (tuple[int, int]): Maximum bounds for boxes. specifies (H, W)
        wh_ratio_clip (float): Maximum aspect ratio for boxes.

    Returns:
        Tensor: Boxes with shape (N, 4), where columns represent
            tl_x, tl_y, br_x, br_y.

    References:
        .. [1] https://arxiv.org/abs/1311.2524

    Example:
        >>> rois = torch.Tensor([[ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 5.,  5.,  5.,  5.]])
        >>> deltas = torch.Tensor([[  0.,   0.,   0.,   0.],
        >>>                        [  1.,   1.,   1.,   1.],
        >>>                        [  0.,   0.,   2.,  -1.],
        >>>                        [ 0.7, -1.9, -0.5,  0.3]])
        >>> delta2bbox(rois, deltas, max_shape=(32, 32))
        tensor([[0.0000, 0.0000, 1.0000, 1.0000],
                [0.1409, 0.1409, 2.8591, 2.8591],
                [0.0000, 0.3161, 4.1945, 0.6839],
                [5.0000, 5.0000, 5.0000, 5.0000]])
    """
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Compute center of each roi
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    # Compute width/height of each roi
    pw = (rois[:, 2] - rois[:, 0]).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1]).unsqueeze(1).expand_as(dh)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    gx = px + pw * dx
    gy = py + ph * dy
    # Convert center-xy/width/height to top-left, bottom-right
    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)
    return bboxes


class IoUBBoxAnchorConverter:

    def __init__(self,
                 num_classes,
                 nms_pre,
                 bbox_coder,
                 alpha=0.4,
                 height_th=9,
                 revert=True,
                 use_sigmoid=True):
        super().__init__()
        kwargs = config.get_kwargs(bbox_coder, 'DeltaXYWHBBoxCoder')
        self.bbox_coder = DeltaXYWHBBoxCoder(**kwargs)
        self.alpha = alpha
        self.height_th = height_th
        self.use_sigmoid_cls = use_sigmoid
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.nms_pre = nms_pre
        self.revert = revert

    def get_bboxes(self, mlvl_anchors, img_metas, cls_scores, bbox_preds,
                   iou_preds):
        """Transform network output for a batch into bbox predictions.

        Aapted from https://github.com/open-mmlab/mmdetection

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class labelof the
                corresponding box.

        Example:
            >>> self = AnchorHead(
            >>>     num_classes=9,
            >>>     in_channels=1,
            >>>     anchor_generator=dict(
            >>>         type='AnchorGenerator',
            >>>         scales=[8],
            >>>         ratios=[0.5, 1.0, 2.0],
            >>>         strides=[4,]))
            >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
            >>> cfg = Config(dict(
            >>>     score_thr=0.00,
            >>>     nms=dict(type='nms', iou_thr=1.0),
            >>>     max_per_img=10))
            >>> feat = torch.rand(1, 1, 3, 3)
            >>> cls_score, bbox_pred = self.forward_single(feat)
            >>> # note the input lists are over different levels, not images
            >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
            >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
            >>>                               img_metas, cfg)
            >>> det_bboxes, det_labels = result_list[0]
            >>> assert len(result_list) == 1
            >>> assert det_bboxes.shape[1] == 5
            >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
        """
        assert len(cls_scores) == len(bbox_preds) == len(iou_preds)
        num_levels = len(cls_scores)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            iou_pred_list = [
                iou_preds[i][img_id].detach() for i in range(num_levels)
            ]

            # TODO: hard code. 0 for anchor_list, 1 for valid_flag_list
            anchors = mlvl_anchors[0][img_id]
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                iou_pred_list, anchors,
                                                img_metas[img_id],
                                                self.nms_pre, self.revert)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           iou_pred_list,
                           mlvl_anchors,
                           img_metas,
                           nms_pre,
                           revert=True):
        """Transform outputs for a single batch item into bbox predictions.

        Aapted from https://github.com/open-mmlab/mmdetection

        Args:
            cls_score_list (list[Tensor]): Box scores for a single scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas for a single
                scale level with shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """

        # if len(mlvl_anchors) > 1:
        #     mlvl_anchors = mlvl_anchors[0]

        def filter_boxes(boxes, min_scale, max_scale):
            ws = boxes[:, 2] - boxes[:, 0]
            hs = boxes[:, 3] - boxes[:, 1]
            scales = torch.sqrt(ws * hs)

            return (scales >= max(1, min_scale)) & (scales <= max_scale)

        img_shape = img_metas['img_shape']
        scale_factor = img_metas['scale_factor']
        assert len(cls_score_list) == len(bbox_pred_list) == len(
            iou_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, iou_pred, anchors in zip(
                cls_score_list, bbox_pred_list, iou_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            iou_pred = iou_pred.permute(1, 2, 0).reshape(-1, 1).sigmoid()
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)

            if self.alpha is None:
                scores *= iou_pred
            elif isinstance(self.alpha, float):
                scores = torch.pow(scores, 2 * self.alpha) * torch.pow(
                    iou_pred, 2 * (1 - self.alpha))
            else:
                raise ValueError("alpha must be float or None")

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since  v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)

        keeps = filter_boxes(mlvl_bboxes, 1, 10000)
        mlvl_bboxes = mlvl_bboxes[keeps]
        mlvl_scores = mlvl_scores[keeps]

        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since  v2.0
            # BG cat_id: num_class
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = mlvl_scores.new_ones(mlvl_scores.shape[0]).detach()
        if revert:
            flip = img_metas.get('flip', False)
            flip_direction = img_metas.get('flip_direction', 'horizontal')
            mlvl_bboxes = bbox_revert(mlvl_bboxes, img_shape, scale_factor,
                                      flip, flip_direction)

        if self.height_th is not None:
            hs = mlvl_bboxes[:, 3] - mlvl_bboxes[:, 1]
            valid = (hs >= self.height_th)
            mlvl_bboxes, mlvl_scores, mlvl_centerness = (
                mlvl_bboxes[valid], mlvl_scores[valid], mlvl_centerness[valid])

        return mlvl_bboxes, mlvl_scores, mlvl_centerness


class BoxPost:
    def __init__(self):
        mesh_grid = config.get_kwargs(config.meshgrid, 'BBoxAnchorMeshGrid')
        self.mesh_grid = BBoxAnchorMeshGrid(**mesh_grid)
        convert = config.get_kwargs(config.converter, 'IoUBBoxAnchorConverter')
        self.converter = IoUBBoxAnchorConverter(**convert)
        if config.use_sigmoid:
            self.cls_out_channels = config.num_classes
        else:
            self.cls_out_channels = config.num_classes + 1

        self.test_cfg = config.test_cfg
        return

    def _get_raw_dets(self, feats, img_metas):
        """
        Args:
            feats( [torch.Tensor] ): shape N*3*H*W, N is batch size
            img_metas(list): len(img_metas) = N
        Returns:
            dets(list): len(dets) is the batch size, len(dets[ii]) = #classes,
                dets[ii][jj] is an np.array whose shape is N*5
        """
        featmap_sizes = [feat.shape[-2:] for feat in feats[0]]
        dtype = feats[0][0].dtype
        device = feats[0][0].device
        anchor_mesh = self.mesh_grid.gen_anchor_mesh(featmap_sizes,
                                                     img_metas,
                                                     dtype,
                                                     device)
        # bboxes, scores, score_factor
        dets = self.converter.get_bboxes(anchor_mesh, img_metas, *feats)

        return dets

    def simple_infer(self, feats, img_metas):
        """
        Args:
            feats([torch.Tensor]): shape N*3*H*W, N is batch size
            img_metas(list): len(img_metas) = N
        Returns:
            dets(list): len(dets) is the batch size, len(dets[ii]) = #classes,
                dets[ii][jj] is an np.array whose shape is N*5
        """
        dets = self._get_raw_dets(feats, img_metas)
        batch_size = len(dets)

        result_list = []
        for ii in range(batch_size):
            bboxes, scores, centerness = dets[ii]
            # det_bboxes, det_labels = self._torch_nms(bboxes,
            #                                          scores,
            #                                          self.test_cfg.score_thr,
            #                                          self.test_cfg.max_per_img,
            #                                          score_factors=centerness)

            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                self.test_cfg['score_thr'],
                self.test_cfg['nms'],
                self.test_cfg['max_per_img'],
                score_factors=centerness)

            bbox_result = bbox2result(det_bboxes, det_labels,
                                      self.cls_out_channels)
            result_list.append(bbox_result)

        return result_list