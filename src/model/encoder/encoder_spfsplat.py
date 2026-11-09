from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
import math

from .backbone.croco.misc import transpose_to_landscape
from .heads import head_factory, camera_head_factory
from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.normalize_shim import apply_normalize_shim, normalize_image
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import Backbone, BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter
from .encoder import Encoder
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg
from ...misc.cam_utils import camera_normalization, convert_pose_to_4x4, depth_projector
from ...geometry.camera_emb import get_plucker_embedding
from .heads.pose_head import PoseHeadCfg
from ...misc.intrinsics_utils import estimate_intrinsics

inf = float('inf')


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderSPFSplatCfg:
    name: Literal["spfsplat"]
    d_feature: int
    num_monocular_samples: int
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    gaussian_adapter: GaussianAdapterCfg
    apply_bounds_shim: bool
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    num_surfaces: int
    gs_params_head_type: str
    pose_head: PoseHeadCfg

    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pretrained_weights: str = ""
    pose_free: bool = True
    pose_make_baseline_1: bool = True
    pose_make_relative: bool = True
    pose_head_type: str = 'mlp'
    estimating_focal: bool = False
    estimating_pose: bool = True
    igf: Optional[dict] = None  # ecosplat IGF stage-2 finetune cfg; None disables.
    sg: Optional[dict] = None  # sg_wrapper spherical-Gaussian color cfg; None = stock SH.
    min_protect_rate: float = 1.0 / 16.0  # PLGC top-K floor; set 0.05 to match paper 5% level.




def rearrange_head(feat, patch_size, H, W):
    B = feat.shape[0]
    feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    feat = F.pixel_shuffle(feat, patch_size)  # B,D,H,W
    feat = rearrange(feat, "b d h w -> b (h w) d")
    return feat


class EncoderSPFSplat(Encoder[EncoderSPFSplatCfg]):
    backbone: nn.Module
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderSPFSplatCfg) -> None:
        super().__init__(cfg)

        self.backbone = get_backbone(cfg.backbone, 3)

        self.pose_free = cfg.pose_free
        # sg_wrapper: spherical-Gaussian color parametrization (MEGS2-style).
        # None keeps the stock SH adapter; the SG adapter shares the
        # UnifiedGaussianAdapter interface, so call sites do not change.
        self.sg = None
        if cfg.sg is not None:
            # sg + igf is the stage-2 combination: IGF is color-agnostic (it
            # deep-copies whatever param head is present and its losses touch
            # only opacity/geometry), so the SG head merges exactly like SH.
            assert cfg.pose_free, "the SG adapter currently requires the pose-free path"
            from sg_wrapper import SGAdapterCfg, UnifiedSGGaussianAdapter
            self.sg = SGAdapterCfg(**dict(cfg.sg))
            self.gaussian_adapter = UnifiedSGGaussianAdapter(cfg.gaussian_adapter, self.sg)
        elif self.pose_free:
            self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)
        else:
            self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in  # 1 for opacity

        self.gs_params_head_type = cfg.gs_params_head_type
        # 显存取舍：一次前向要为每个视角各跑一个 pts3d 头和一个颜色头，DPT 的
        # 中间激活是显存主项。开关由 ModelWrapper 按 train.grad_checkpoint 注入。
        self.grad_checkpoint = False
       
        self.set_center_head(output_mode='pts3d', head_type='dpt', landscape_only=True,
                            depth_mode=('exp', -inf, inf), conf_mode=None,)
 
            
        self.set_gs_params_head(cfg, cfg.gs_params_head_type)

        if self.cfg.estimating_pose:
            self.set_pose_head(cfg, cfg.pose_head_type)

        for name, param in self.named_parameters():
            param.requires_grad = False


        # IGF stage-2 module (clones gaussian param heads + image embed).
        self.igf = None
        if cfg.igf is not None:
            if not cfg.pretrained_weights:
                raise ValueError(
                    "IGF strict-loads the base checkpoint inside the encoder, so"
                    " model.encoder.pretrained_weights must point at a checkpoint"
                    " with the SAME head width as this config (the"
                    " representation-adapted base, e.g. sg3_step30000.ckpt for"
                    " the SG variant)."
                )
            from collections import OrderedDict
            stage1_statedict = torch.load(cfg.pretrained_weights, map_location='cpu')['state_dict']
            weights_to_load = OrderedDict({}) # 변수 이름 변경 (더 명확하게)
            # 모든 키가 'encoder.'로 시작한다고 가정
            prefix = 'encoder.'
            prefix_len = len(prefix)

            for k, v in stage1_statedict.items():
                if k.startswith(prefix): # 안전하게 startswith 확인 (필수는 아님)
                    trimmed_k = k[prefix_len:] # 'encoder.' 제거
                    weights_to_load[trimmed_k] = v

            self.load_state_dict(weights_to_load)


            from ecosplat_wrapper import IGFConfig, IGFModule
            self.igf = IGFModule(
                heads_to_clone=[self.gaussian_param_head, self.gaussian_param_head2],
                igf_cfg=IGFConfig(**dict(cfg.igf)),
            )
            for p in self.igf.parameters():
                p.requires_grad = True

        # IGF importance-mask cache: the mask is detached pseudo-GT and its
        # CPU/GPU pipeline (variation map, quantile, bipartite matching) is
        # the per-micro-batch bottleneck; consecutive micro-batches of the
        # same scene at the same drawn rate reproduce it bit-exactly, so we
        # reuse it within a small staleness span instead.
        self._mask_cache = None  # (key, rate, infos, age)

        # sg_wrapper: partial stage-1 init for representation-swap runs (SG
        # head or reduced-SH baselines). Loads backbone / center / pose
        # heads and the param-head trunk; the parametrization-specific final
        # conv keeps a fresh init except its aligned leading channels
        # (density, scale, rotation, c0/SH-DC). Which modules actually train
        # is controlled by train.freeze_pretrained / freeze_pose_head.
        if cfg.pretrained_weights and cfg.igf is None:
            from sg_wrapper import init_encoder_from_stage1, is_stage1_checkpoint

            if is_stage1_checkpoint(cfg.pretrained_weights):
                init_encoder_from_stage1(self, cfg.pretrained_weights)
                # Signals src/main.py to skip its own (shape-strict) reload.
                self._stage1_partial_loaded = True
            # Otherwise the path holds upstream CroCo/MASt3R weights (top level
            # `model`), which is the from-scratch stage-1 (PGT) initialisation:
            # src/main.py loads it, so nothing to do here.

    def set_grad_checkpoint(self, flag: bool) -> None:
        """打开或关闭激活重算，覆盖头部与骨干的每个 transformer block。

        Args:
            flag: 为真时前向不留中间激活，反向前向重算。
        """
        # 骨干（ViT 编码器 + 解码器）的激活量随视角数线性增长，是十视角下的大头；
        # 头部每个视角各跑一次 DPT，量也不小。两者一起开关，口径才一致。
        self.grad_checkpoint = bool(flag)
        backbone = getattr(self, "backbone", None)
        if backbone is not None and hasattr(backbone, "grad_checkpoint"):
            backbone.grad_checkpoint = bool(flag)

    def set_center_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode):
        self.backbone.depth_mode = depth_mode
        self.backbone.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode))

        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)


    def set_gs_params_head(self, cfg, head_type):
        if head_type == 'linear':
            self.gaussian_param_head = nn.Sequential(
                nn.ReLU(),
                nn.Linear(
                    self.backbone.dec_embed_dim,
                    cfg.num_surfaces * self.patch_size ** 2 * self.raw_gs_dim,
                ),
            )

            self.gaussian_param_head2 = deepcopy(self.gaussian_param_head)

        elif 'dpt' in head_type:
            self.gaussian_param_head = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)
            self.gaussian_param_head2 = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)
        else:
            raise NotImplementedError(f"unexpected {head_type=}")
        
   
    def set_pose_head(self, cfg, head_type='mlp'):
        self.pose_head = camera_head_factory(head_type, 'pose', self.backbone, cfg.pose_head)
        self.pose_head2 = camera_head_factory(head_type, 'pose', self.backbone, cfg.pose_head)



    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def _head_forward(self, head, tokens, img_info, ray_embedding=None):
        """Run one DPT head, optionally recomputing it during backward.

        Args:
            head: DPT head module, or the plain function returned by
                transpose_to_landscape.
            tokens: four token tensors of shape (B, N, C).
            img_info: second positional argument of the head, either (H, W) or
                the true_shape tensor of shape (B, 2).
            ray_embedding: third positional argument, kept for signature parity
                with the existing call sites.

        Returns:
            Tensor of shape (B, C_out, H, W).
        """
        # 步骤 1：训练期且开关打开时，把 DPT 的中间激活换成反向重算，省显存；
        # 验证/推理没有反向图，直接前向即可，没必要付重算的时间。
        if self.grad_checkpoint and torch.is_grad_enabled():
            return checkpoint(head, tokens, img_info, ray_embedding, use_reentrant=False)
        return head(tokens, img_info, ray_embedding)

    def _downstream_head(self, head_num, decout, img_shape, ray_embedding=None):
        head = getattr(self, f'head{head_num}')
        return self._head_forward(head, decout, img_shape, ray_embedding)
    
    

    _MASK_REUSE_SPAN = 3  # micro-batches a cached mask may serve

    def _cached_distill(self, context, extrinsics, protect_rate, means_pix, cov_pix, depths_per_view):
        """Return distill_infos for this micro-batch, reusing the cached mask.

        The cache key is the scene's first context frame (stable across the
        accumulate window) plus image resolution and the pivotal camera; the
        drawn rate is matched on 0.05 buckets (the PLGC grid step), so a
        redraw within the same bucket reuses the mask -- its threshold moves
        by at most one order statistic, within the pseudo-target tolerance.
        Cache age beyond _MASK_REUSE_SPAN forces a recompute.
        """
        b, v_cxt, _, h, w = context["image"].shape
        key = (
            int(context["index"][0, 0].item()),
            h,
            w,
            float(extrinsics[0, 0, 0, 3].item()),
        )
        rate_bucket = round(protect_rate / 0.05)  # PLGC 19-grid -> 0.05 buckets
        cached = self._mask_cache
        if (
            cached is not None
            and cached[0] == key
            and cached[1] == rate_bucket
            and cached[3] < self._MASK_REUSE_SPAN
        ):
            self._mask_cache = (key, rate_bucket, cached[2], cached[3] + 1)
            return cached[2]

        infos = self.igf.compute_distill(
            ori_gaussians={"means_pix": means_pix, "cov_pix": cov_pix},
            image=context["image"],
            intrinsics=context["intrinsics"],
            extrinsics=extrinsics,
            protect_rate=protect_rate,
            depth=depths_per_view,
        )
        self._mask_cache = (key, rate_bucket, infos, 0)
        return infos

    def forward(
        self,
        context: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
        target: Optional[dict] = None,
    ) :
        device = context["image"].device
        b, v_cxt, _, h, w = context["image"].shape

        if self.training and self.igf is None and not self.cfg.pretrained_weights:
            raise RuntimeError(
                "stage-2 training needs either cfg.model.encoder.igf "
                "(`model.encoder.igf: {}`) or a stage-1 checkpoint via "
                "cfg.model.encoder.pretrained_weights."
            )

        if target is not None:
            v_tgt = target["image"].shape[1]
            context_target = {
                "image": normalize_image(torch.cat([context["image"], target["image"]], dim=1)),
                "intrinsics": torch.cat([context["intrinsics"], target["intrinsics"]], dim=1),
            }
            # Encode the context and target images.
            out = self.backbone(context_target, target_num_views=v_tgt)
            dec_feat, dec_feat_w_tgt = out['dec_feat'], out['dec_feat_w_tgt']
        else:
            v_tgt = 0
            context_input = {
                "image": normalize_image(context["image"]),
                "intrinsics": context["intrinsics"],
            }
            # Encode the context images.
            out = self.backbone(context_input)
            dec_feat = out['dec_feat']

        shape, images = out["shape"], out["images"]
        
        # IGF stage-2: sample protect_rate + compute rate_feat once, used in merge heads below.
        # Train: PLGC-sampled rho. Eval: cfg.inference_rho (default 0.4) or context["protect_rate"] override.
        igf_active = self.igf is not None
        budget_mode = bool(igf_active and self.cfg.igf and self.cfg.igf.get("budget_mode", False))
        budget_stats = None
        budget_m_hist = None
        rate_feat = None
        protect_rate = None
        if igf_active:
            override = context.get("protect_rate") if isinstance(context, dict) else None
            protect_rate = self.igf.get_rho(global_step, self.training, override=override)
            rho_3ch = torch.ones(b * v_cxt, 3, h, w, device=device, dtype=context["image"].dtype) * protect_rate
            rate_feat = self.igf.rate_embed(rho_3ch).view(b, v_cxt, -1, h, w)
            assert rate_feat.shape[2] == 256, (
                f"rate_feat must have 256 channels to match stage-2; got {rate_feat.shape[2]}."
            )

        with torch.amp.autocast('cuda', enabled=False):
            all_mean_res, all_other_params = [], []
            merged_other_params = [] if igf_active else None
            if self.cfg.estimating_pose:
                all_pose_params, all_pose_params_cwt = [], []


            # Pts3d head (context only)
            res1 = self._downstream_head(1, [tok[:, 0].float() for tok in dec_feat], shape[:, 0])
            all_mean_res.append(res1)
            for i in range(1, v_cxt):
                res2 = self._downstream_head(2, [tok[:, i].float() for tok in dec_feat], shape[:, i])
                all_mean_res.append(res2)

            # Gaussian parameter head (context only)
            if 'dpt' in self.gs_params_head_type:
                GS_res1 = self._head_forward(self.gaussian_param_head, [tok[:, 0].float() for tok in dec_feat], images[:, 0, :3], shape[0, 0].cpu().tolist())
                GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
                all_other_params.append(GS_res1)
                if igf_active:
                    GS_res1_m = self.igf.merge_heads[0](
                        [tok[:, 0].float() for tok in dec_feat], images[:, 0, :3], shape[0, 0].cpu().tolist(),
                        rate_feat=rate_feat[:, 0].float(),
                    )
                    merged_other_params.append(rearrange(GS_res1_m, "b d h w -> b (h w) d"))
                for i in range(1, v_cxt):
                    GS_res2 = self._head_forward(self.gaussian_param_head2, [tok[:, i].float() for tok in dec_feat], images[:, i, :3], shape[0, i].cpu().tolist())
                    GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")
                    all_other_params.append(GS_res2)
                    if igf_active:
                        GS_res2_m = self.igf.merge_heads[1](
                            [tok[:, i].float() for tok in dec_feat], images[:, i, :3], shape[0, i].cpu().tolist(),
                            rate_feat=rate_feat[:, i].float(),
                        )
                        merged_other_params.append(rearrange(GS_res2_m, "b d h w -> b (h w) d"))
            else:
                raise NotImplementedError(f"unexpected {self.gs_params_head_type=}")
            

            # Pose head
            if self.cfg.estimating_pose:
                # Context views
                pose_res1 = self.pose_head([tok[:, 0].float() for tok in dec_feat], shape[0, 0].cpu().tolist()) # (16, 9)
                all_pose_params.append(pose_res1)
                for i in range(1, v_cxt):
                    pose_res2 = self.pose_head2([tok[:, i].float() for tok in dec_feat], shape[0, i].cpu().tolist()) # (16, 9)
                    all_pose_params.append(pose_res2)

                # Context + target views
                if target is not None:
                    pose_res1 = self.pose_head([tok[:, 0].float() for tok in dec_feat_w_tgt], shape[0, 0].cpu().tolist()) # (16, 9)
                    all_pose_params_cwt.append(pose_res1)
                    for i in range(1, v_cxt + v_tgt):
                        pose_res2 = self.pose_head2([tok[:, i].float() for tok in dec_feat_w_tgt], shape[0, i].cpu().tolist()) # (16, 9)
                        all_pose_params_cwt.append(pose_res2)
            

            
            
        gaussians = torch.stack(all_other_params, dim=1) # [b, v, 65536, 83]
        
        if self.cfg.estimating_pose:
            poses_enc = torch.stack(all_pose_params, dim=1) # (b, v 9)
            pred_extrinsics = self.process_pose(poses_enc, context['index']) # (b, v, 4, 4)

            if target is not None:
                poses_enc_cwt = torch.stack(all_pose_params_cwt, dim=1) # (b, v + v2, 9)
                pred_extrinsics_cwt = self.process_pose(poses_enc_cwt, context['index']) # (b, v + v2, 4, 4)



        pts_all = [all_mean_res_i['pts3d'] for all_mean_res_i in all_mean_res]
        pts_all = torch.stack(pts_all, dim=1) # [b, v, h, w, 3]
        pts_all = rearrange(pts_all, "b v h w xyz -> b v (h w) xyz")
        context_extrinsics = pred_extrinsics[:, :v_cxt] if self.cfg.estimating_pose else context["extrinsics"]
        depths_per_view = self.process_depth(context_extrinsics, rearrange(pts_all, "b v (h w) xyz -> b v h w xyz", h=h, w=w)) # depth for each cam, (b, v, h, w)
            

        gaussians = rearrange(gaussians, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces) # for cfg.num_surfaces
        densities = gaussians[..., 0].sigmoid().unsqueeze(-1)
        gaussian_parameters = gaussians[..., 1:]
        pts_all = pts_all.unsqueeze(-2)  # for cfg.num_surfaces


        # Convert the features and depths into Gaussians.
        gaussians = self.gaussian_adapter.forward(
            pts_all.unsqueeze(-2),
            self.map_pdf_to_opacity(densities, global_step),
            rearrange(gaussian_parameters, "b v r srf c -> b v r srf () c"),
        )

        # IGF: build merged Gaussians via the same adapter on cloned-head output.
        merged_gaussians = None
        merged_opacities_per_pixel = None
        if igf_active:
            merged_raw = torch.stack(merged_other_params, dim=1)
            merged_raw = rearrange(merged_raw, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces)
            merged_densities = merged_raw[..., 0].sigmoid().unsqueeze(-1)
            merged_params = merged_raw[..., 1:]
            merged_opacity_mapped = self.map_pdf_to_opacity(merged_densities, global_step)
            # Per-pixel opacity for L_io BCE: collapse srf+spp dims to (b, v, h*w).
            merged_opacities_per_pixel = merged_opacity_mapped.flatten(-3, -1)

            # PLGC keeps a fraction of the points and every point carries all
            # lobes; budget mode reads the same rate as a total payload and
            # lets the allocator set the per-point lobe level instead.
            if budget_mode:
                assert self.sg is not None, "budget mode needs the SG colour head"
                assert self.cfg.num_surfaces == 1, "budget mode assumes one surface"
                from sg_wrapper import dense_floats, select_by_budget

                num_lobes = int(self.sg.num_lobes)
                sg_raw = merged_params[..., 0, 7:]  # (b, v, P, 3 + 7L)
                _, _, sharp_raw, rgb_raw = torch.split(
                    sg_raw, [3, 3 * num_lobes, num_lobes, 3 * num_lobes], dim=-1)
                # D = |a| (1 - e^{-2s})：与 sg_prune.lobe_importance 同一口径，
                # 分配器排序用的正是「删掉这一颗会损失多少」。
                sharpness = sharp_raw.reshape(*sharp_raw.shape[:-1], num_lobes, 1).abs()
                rgb = rgb_raw.reshape(*rgb_raw.shape[:-1], num_lobes, 3)
                lobe_value = rgb.abs().mean(dim=-1) * (
                    1.0 - torch.exp(-2.0 * sharpness.squeeze(-1)))
                rate = max(float(protect_rate), float(self.cfg.min_protect_rate))
                kappa_view = int(dense_floats(1, h, w, lobes_per_point=num_lobes) * rate)
                selection = select_by_budget(
                    merged_opacities_per_pixel.detach(), lobe_value.detach(),
                    kappa_view, lobes_per_point=num_lobes)
                top_idx = selection.index
                budget_stats = selection
                # 每点买到几颗 lobe 的直方图（只数有效槽位）。同一预算在不同
                # 场景里可能全花在点上、也可能给每个点配 lobe，这张直方图
                # 就是「预算怎么花」的分布，评估时按场景落盘。
                m_per_point = selection.keep_lobes.sum(-1).to(torch.long)
                budget_m_hist = torch.bincount(
                    m_per_point[selection.valid], minlength=num_lobes + 1).tolist()
            else:
                # Top-k by per-pixel opacity (paper PLGC selection); rendered set shrinks with rho.
                k = int(max(protect_rate, self.cfg.min_protect_rate) * h * w)
                _, top_idx = torch.topk(merged_opacities_per_pixel, k=k, dim=-1)
            pts_topk = torch.gather(
                pts_all, 2,
                top_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, pts_all.shape[-2], pts_all.shape[-1]),
            )
            params_topk = torch.gather(
                merged_params, 2,
                top_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, merged_params.shape[-2], merged_params.shape[-1]),
            )
            densities_topk = torch.gather(
                merged_densities, 2,
                top_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, merged_densities.shape[-2], merged_densities.shape[-1]),
            )

            if budget_mode:
                # 步骤1：保持预算选择不变，仅在显式开启时补回被删 lobe 的球面均值。
                # 补偿在原始颜色系数空间执行；补位槽位密度仍置零，不参与渲染。
                head = params_topk[..., 0, :]
                c0_m, dirs_m, sharp_m, rgb_m = torch.split(
                    head[..., 7:], [3, 3 * num_lobes, num_lobes, 3 * num_lobes], dim=-1)
                rgb_m = rgb_m.reshape(*rgb_m.shape[:-1], num_lobes, 3)
                if self.cfg.igf.get("budget_color_compensation", False):
                    from sg_wrapper.sg_prune import compensate_pruned_lobes

                    sharp_activated = sharp_m.reshape(
                        *sharp_m.shape[:-1], num_lobes, 1).abs()
                    compensated = compensate_pruned_lobes(
                        c0_m, rgb_m, sharp_activated, selection.keep_lobes)
                    c0_m = torch.where(selection.valid.unsqueeze(-1), compensated, c0_m)
                # 步骤2：先补偿再清零，保留训练时对颜色及锐度的梯度。
                rgb_m = rgb_m * selection.keep_lobes.to(rgb_m.dtype).unsqueeze(-1)
                params_topk = torch.cat(
                    [head[..., :7], c0_m, dirs_m, sharp_m, rgb_m.flatten(-2)],
                    dim=-1).unsqueeze(-2)
                densities_topk = densities_topk * selection.valid.unsqueeze(
                    -1).unsqueeze(-1).to(densities_topk.dtype)

            merged_gaussians = self.gaussian_adapter.forward(
                pts_topk.unsqueeze(-2),
                self.map_pdf_to_opacity(densities_topk, global_step),
                rearrange(params_topk, "b v r srf c -> b v r srf () c"),
            )


        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = depths_per_view

            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )
            visualization_dump["means"] = rearrange(
                gaussians.means, "b v (h w) srf spp xyz -> b v h w (srf spp) xyz", h=h, w=w
            ) 
            visualization_dump['opacities'] = rearrange(
                gaussians.opacities, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ) 

            
        if self.cfg.estimating_focal:
            intrinsics = estimate_intrinsics(rearrange(gaussians.means, "b v (h w) srf spp xyz -> b v h w (srf spp) xyz", h=h, w=w).squeeze(-2), h, w)
            pred_intrinsics = intrinsics.unsqueeze(1).repeat(1, v_cxt, 1, 1)
            pred_intrinsics_cwt = intrinsics.unsqueeze(1).repeat(1, v_cxt+v_tgt, 1, 1)


        encoder_output = dict()
        if budget_stats is not None:
            encoder_output["budget"] = {
                "rate": float(protect_rate),
                "n_points": budget_stats.n_points,
                "n_lobes": budget_stats.n_lobes,
                "achieved": budget_stats.achieved,
                "target": budget_stats.target,
                "beta": budget_stats.beta,
                "m_hist": budget_m_hist,
            }

        def _flatten_gaussians(g):
            if self.sg is not None:
                from sg_wrapper import flatten_sg_gaussians
                return flatten_sg_gaussians(g)
            return Gaussians(
                rearrange(g.means, "b v r srf spp xyz -> b (v r srf spp) xyz"),
                rearrange(g.covariances, "b v r srf spp i j -> b (v r srf spp) i j"),
                rearrange(g.rotations, "b v r srf spp i  -> b (v r srf spp) i "),
                rearrange(g.scales, "b v r srf spp i  -> b (v r srf spp) i "),
                rearrange(g.harmonics, "b v r srf spp c d_sh -> b (v r srf spp) c d_sh"),
                rearrange(g.opacities, "b v r srf spp -> b (v r srf spp)"),
            )

        ori_flat = _flatten_gaussians(gaussians)
        if igf_active:
            encoder_output["gaussians"] = _flatten_gaussians(merged_gaussians)
            if self.training:
                encoder_output["ori_gaussians"] = ori_flat
                means_pix = gaussians.means.reshape(b, v_cxt, h, w, 3)
                cov_pix = gaussians.covariances.reshape(b, v_cxt, h, w, 3, 3)
                distill_infos = self._cached_distill(
                    context, context_extrinsics, protect_rate, means_pix, cov_pix, depths_per_view
                )
                distill_infos["pred_opacity"] = merged_opacities_per_pixel
                encoder_output["distill_infos"] = distill_infos
                encoder_output["protect_rate"] = protect_rate
        else:
            encoder_output["gaussians"] = ori_flat

        if self.cfg.estimating_pose:
            encoder_output['extrinsics'] = {"c": pred_extrinsics}
            if target is not None:
                encoder_output['extrinsics']['cwt'] = pred_extrinsics_cwt

        if self.cfg.estimating_focal:
            encoder_output['intrinsics'] = {"c": pred_intrinsics}
            if target is not None:
                encoder_output['intrinsics']['cwt'] = pred_intrinsics_cwt

        return encoder_output

    def process_pose(self, pose_enc, context_index):
        # pose_enc: (b v 9)
        b, v = pose_enc.shape[:2]
        poses = convert_pose_to_4x4(rearrange(pose_enc, "b v ... -> (b v) ..."))
        poses = rearrange(poses, "(b v) ... -> b v ...", b=b, v=v)

        if self.cfg.pose_make_baseline_1:
            a = poses[:, context_index.argmin().item(), :3, 3]  # [b, 3]
            b = poses[:, context_index.argmax().item(), :3, 3]  #  [b, 3]

            scale = (a - b).norm(dim=1, keepdim=True)  # [b, 1]
            # scale = (a - poses[:, 1, :3, 3]).norm(dim=1, keepdim=True)
            poses[:, :, :3, 3] /= scale.unsqueeze(-1)

        if self.cfg.pose_make_relative:
            base_context_pose = poses[:,0] # [b, 4, 4]
            inv_base_context_pose = torch.inverse(base_context_pose)
            poses = inv_base_context_pose[:, None, :, :] @ poses # [b,1,4,4] @ [b,v,4,4]

        return poses      
    
    def process_depth(self, pose, pts3d):
        b, v, h, w, _ = pts3d.shape
        pts3d = rearrange(pts3d, "b v h w c -> (b v) (h w) c")
        pose = rearrange(pose, "b v ... -> (b v) ...")


        depths = depth_projector(pts3d, pose) # (bv, n, 1)
        depths = rearrange(depths, "(b v) (h w) 1 -> b v h w", b=b, v=v, h=h, w=w)
        return depths.contiguous()

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )

            return batch

        return data_shim
