下面给出一个**按论文 COCA 网络描述抽象出来的 PyTorch 风格伪代码**。论文只明确描述了整体结构和训练策略，没有公开精确的通道数、U-Net 深度、阈值融合公式等实现细节，所以这里把这些部分写成可替换的占位参数。COCA 的核心是两阶段：Stage I 用 U-Net 定位/粗分割结直肠 ROI 并裁剪影像；Stage II 在裁剪 ROI 内做“CRC 分割 + 患者级二分类”的多任务诊断，且把多尺度 decoder 特征做空间池化后拼接，送入分类分支预测 CRC。论文还说明采用混合监督、联合 loss，并用分类概率和分割体积共同给出最终诊断。

---

## 1. 输入与标签约定

```python
# image: 非增强 CT 体数据
# shape = [B, 1, D, H, W]

# mask: 对应体素级掩码
# shape = [B, D, H, W]
# 建议编码：
#   0 = background
#   1 = colorectum / healthy bowel tissue
#   2 = CRC tumor
#
# Stage I 的 ROI mask 可由：
#   roi_mask = (mask > 0)
# 得到，即结直肠 + 肿瘤区域都属于 ROI。

# label: 患者级类别标签
# shape = [B]
#   0 = normal
#   1 = CRC
```

如果某些样本没有体素级 mask，只含患者级 label，则可以用 `has_mask=False`，训练时只计算分类 loss；这对应论文的 mixed-supervised setting。论文提到 Site A 提供体素级和患者级标签，而 Site B 只有患者级标签，训练时结合两类监督信号。

---

## 2. 基础 U-Net 模块伪代码

论文图示中的 U-Net block 可概括为：`2 × (Conv + BatchNorm + ReLU)`，下采样用 pooling，上采样用 upsampling，最后接 softmax 分割头。

```python
class ConvBlock3D:
    """
    2 * (Conv3D + BatchNorm3D + ReLU)
    """
    def __init__(self, in_ch, out_ch):
        self.conv1 = Conv3D(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1   = BatchNorm3D(out_ch)
        self.relu1 = ReLU()

        self.conv2 = Conv3D(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2   = BatchNorm3D(out_ch)
        self.relu2 = ReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x
```

---

## 3. 通用 3D U-Net 伪代码

```python
class UNet3D:
    """
    3D U-Net:
    Encoder: ConvBlock + Pooling
    Decoder: Upsampling + Skip Connection + ConvBlock
    Output : segmentation logits

    return_decoder_features=True 时，
    返回每一级 decoder feature，用于 Stage II 的分类分支。
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=2,
        base_channels=32,
        depth=5,
        return_decoder_features=False
    ):
        self.return_decoder_features = return_decoder_features

        # -------- Encoder --------
        self.enc_blocks = []
        self.pools = []

        ch_in = in_channels
        for i in range(depth):
            ch_out = base_channels * (2 ** i)
            self.enc_blocks.append(ConvBlock3D(ch_in, ch_out))
            self.pools.append(MaxPool3D(kernel_size=2, stride=2))
            ch_in = ch_out

        # -------- Bottleneck --------
        self.bottleneck = ConvBlock3D(
            base_channels * (2 ** (depth - 1)),
            base_channels * (2 ** depth)
        )

        # -------- Decoder --------
        self.up_layers = []
        self.dec_blocks = []

        for i in reversed(range(depth)):
            ch_skip = base_channels * (2 ** i)
            ch_in   = base_channels * (2 ** (i + 1))

            self.up_layers.append(
                UpSample3D(scale_factor=2, mode="trilinear")
            )

            # concat 后通道数 = 上采样特征通道 + skip 特征通道
            self.dec_blocks.append(
                ConvBlock3D(ch_in + ch_skip, ch_skip)
            )

        # -------- Segmentation head --------
        self.seg_head = Conv3D(
            base_channels,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):
        skips = []

        # Encoder
        for enc, pool in zip(self.enc_blocks, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        decoder_features = []

        for up, dec, skip in zip(self.up_layers, self.dec_blocks, reversed(skips)):
            x = up(x)
            x = center_crop_or_pad_to_match(x, skip)
            x = concat([x, skip], dim=1)
            x = dec(x)
            decoder_features.append(x)

        seg_logits = self.seg_head(x)

        if self.return_decoder_features:
            return seg_logits, decoder_features
        else:
            return seg_logits
```

---

## 4. COCA 两阶段网络结构伪代码

```python
class COCA:
    """
    COCA:
    Stage I  : Localization U-Net
               输入完整 CT，输出结直肠/肠道 ROI 粗分割。
    Stage II : Diagnosis U-Net
               输入裁剪后的 ROI，输出：
                 1) CRC / colorectum segmentation
                 2) patient-level CRC classification
    """

    def __init__(
        self,
        num_seg_classes=3,      # 0 background, 1 colorectum, 2 tumor
        base_channels=32,
        depth=5,
        lambda_seg=1.0,
        tau_roi=0.5,
        tau_tumor=0.5,
        tau_final=0.5
    ):
        self.lambda_seg = lambda_seg
        self.tau_roi = tau_roi
        self.tau_tumor = tau_tumor
        self.tau_final = tau_final

        # -------------------------------
        # Stage I: localization network
        # -------------------------------
        # 输出 2 类：background / colorectal ROI
        self.loc_unet = UNet3D(
            in_channels=1,
            out_channels=2,
            base_channels=base_channels,
            depth=depth,
            return_decoder_features=False
        )

        # -------------------------------
        # Stage II: diagnosis network
        # -------------------------------
        # 输出 3 类：background / colorectum / CRC tumor
        self.diag_unet = UNet3D(
            in_channels=1,
            out_channels=num_seg_classes,
            base_channels=base_channels,
            depth=depth,
            return_decoder_features=True
        )

        # decoder 多尺度特征池化后拼接，送入分类分支
        # 这里的 in_dim 需要按 decoder feature 通道数计算；
        # 伪代码中用 get_decoder_feature_dim() 表示。
        cls_in_dim = get_decoder_feature_dim(
            base_channels=base_channels,
            depth=depth
        )

        self.classifier = MLP(
            in_dim=cls_in_dim,
            hidden_dims=[256, 64],
            out_dim=1          # binary CRC logit
        )

    def crop_by_roi(self, image, roi_prob):
        """
        根据 Stage I 输出的 ROI 概率图裁剪 CT。
        """
        roi_binary = roi_prob > self.tau_roi

        # 可加入连通域筛选、形态学处理、margin 扩展等
        roi_binary = keep_largest_components(roi_binary)
        bbox = bounding_box_3d(roi_binary, margin=(16, 32, 32))

        cropped_image = crop_3d(image, bbox)

        return cropped_image, bbox

    def forward(self, image):
        """
        image: [B, 1, D, H, W]
        """

        # =====================================================
        # Stage I: colorectal / intestinal ROI localization
        # =====================================================
        roi_logits = self.loc_unet(image)
        roi_prob = softmax(roi_logits, dim=1)[:, 1]   # [B, D, H, W]

        cropped_image, bbox = self.crop_by_roi(image, roi_prob)

        # =====================================================
        # Stage II: segmentation + classification
        # =====================================================
        seg_logits, decoder_features = self.diag_unet(cropped_image)

        # segmentation probability
        seg_prob = softmax(seg_logits, dim=1)

        # tumor probability map
        tumor_prob = seg_prob[:, 2]                   # [B, d, h, w]

        # 多尺度 decoder feature -> global embedding
        pooled_features = []
        for feat in decoder_features:
            # feat: [B, C, d, h, w]
            pooled = global_avg_pool_3d(feat)          # [B, C]
            pooled_features.append(pooled)

        global_embedding = concat(pooled_features, dim=1)

        # patient-level classification
        cls_logit = self.classifier(global_embedding)  # [B, 1]
        cls_prob = sigmoid(cls_logit)                  # [B, 1]

        # segmented tumor volume
        tumor_binary = tumor_prob > self.tau_tumor
        tumor_volume = voxel_count(tumor_binary)       # [B]

        # 论文只说明分类概率和分割体积共同用于最终诊断，
        # 未公开具体融合公式；这里用占位函数表示。
        final_score = fuse_classification_and_volume(
            cls_prob=cls_prob,
            tumor_volume=tumor_volume
        )

        final_pred = final_score > self.tau_final

        return {
            "roi_logits": roi_logits,
            "roi_prob": roi_prob,
            "cropped_image": cropped_image,
            "bbox": bbox,

            "seg_logits": seg_logits,
            "seg_prob": seg_prob,
            "tumor_prob": tumor_prob,

            "cls_logit": cls_logit,
            "cls_prob": cls_prob,

            "tumor_volume": tumor_volume,
            "final_score": final_score,
            "final_pred": final_pred
        }
```

---

## 5. Stage I 训练伪代码：ROI 定位 / 粗分割

```python
def train_stage1_localization(model, batch):
    """
    Stage I 只训练 localization U-Net。

    batch:
        image: [B, 1, D, H, W]
        mask : [B, D, H, W]
        label: [B]
    """

    image = batch["image"]
    mask  = batch["mask"]

    # Stage I 目标：结直肠 ROI
    # 只要是 colorectum 或 tumor，都算 ROI
    roi_target = (mask > 0).long()        # [B, D, H, W]

    roi_logits = model.loc_unet(image)    # [B, 2, D, H, W]

    loss_roi_ce   = cross_entropy_loss(roi_logits, roi_target)
    loss_roi_dice = dice_loss(softmax(roi_logits, dim=1), roi_target)

    loss_stage1 = loss_roi_ce + loss_roi_dice

    loss_stage1.backward()
    optimizer_step()

    return loss_stage1
```

---

## 6. Stage II 训练伪代码：分割 + 分类联合优化

论文的诊断阶段是多任务框架：一方面做 CRC 分割，另一方面做患者级 CRC 分类；多尺度 decoder features 被池化、拼接后输入分类分支。

```python
def train_stage2_diagnosis(model, batch):
    """
    Stage II 训练 diagnosis U-Net + classification branch。

    batch:
        image   : [B, 1, D, H, W]
        mask    : [B, D, H, W] or None
        label   : [B], 0 normal, 1 CRC
        has_mask: [B], 是否有体素级标注
    """

    image    = batch["image"]
    mask     = batch["mask"]
    label    = batch["label"].float()      # [B]
    has_mask = batch["has_mask"]           # [B], bool

    # -------------------------------
    # 1. 用 Stage I 定位并裁剪 ROI
    # -------------------------------
    with no_grad():
        roi_logits = model.loc_unet(image)
        roi_prob = softmax(roi_logits, dim=1)[:, 1]
        cropped_image, bbox = model.crop_by_roi(image, roi_prob)

    if mask is not None:
        cropped_mask = crop_3d(mask, bbox)       # [B, d, h, w]
    else:
        cropped_mask = None

    # -------------------------------
    # 2. Stage II forward
    # -------------------------------
    seg_logits, decoder_features = model.diag_unet(cropped_image)

    pooled_features = []
    for feat in decoder_features:
        pooled_features.append(global_avg_pool_3d(feat))

    global_embedding = concat(pooled_features, dim=1)

    cls_logit = model.classifier(global_embedding).squeeze(1)  # [B]

    # -------------------------------
    # 3. 分类 loss
    # -------------------------------
    loss_cls = binary_cross_entropy_with_logits(
        cls_logit,
        label
    )

    # -------------------------------
    # 4. 分割 loss
    # -------------------------------
    # mixed-supervised:
    # 有 mask 的样本计算 segmentation loss；
    # 只有患者级 label 的样本不计算 segmentation loss。
    if any(has_mask):
        seg_logits_masked = seg_logits[has_mask]
        target_masked = cropped_mask[has_mask].long()

        loss_seg_ce = cross_entropy_loss(
            seg_logits_masked,
            target_masked
        )

        loss_seg_dice = dice_loss(
            softmax(seg_logits_masked, dim=1),
            target_masked
        )

        loss_seg = loss_seg_ce + loss_seg_dice
    else:
        loss_seg = 0.0

    # -------------------------------
    # 5. joint loss
    # -------------------------------
    loss_total = loss_cls + model.lambda_seg * loss_seg

    loss_total.backward()
    optimizer_step()

    return {
        "loss_total": loss_total,
        "loss_cls": loss_cls,
        "loss_seg": loss_seg
    }
```

---

## 7. 论文中的 batch 采样策略伪代码

论文提到，为改善小肿瘤检测，每个 batch 包含 **8 个随机样本：3 个 CRC + 5 个 normal**，另外再加入 **2 个富集小肿瘤的 CRC 样本**。

```python
def coca_batch_sampler(dataset):
    """
    COCA small tumor enriched sampler
    """

    crc_cases = dataset.filter(label=1)
    normal_cases = dataset.filter(label=0)
    small_tumor_crc_cases = dataset.filter(
        label=1,
        tumor_size="<small_threshold"
    )

    while True:
        batch_crc = random_sample(crc_cases, n=3)
        batch_normal = random_sample(normal_cases, n=5)
        batch_small_crc = random_sample(small_tumor_crc_cases, n=2)

        batch = batch_crc + batch_normal + batch_small_crc

        yield collate(batch)
```

---

## 8. 五折训练流程伪代码

```python
def train_coca_with_5fold(dataset):
    """
    论文采用 five-fold cross-validation。
    """

    folds = make_5fold_split(dataset)

    trained_models = []

    for fold_id in range(5):

        train_set, val_set = folds[fold_id]

        model = COCA(
            num_seg_classes=3,
            base_channels=32,
            depth=5,
            lambda_seg=1.0
        )

        # -------------------------------
        # Stage I: train localization U-Net
        # -------------------------------
        for epoch in range(num_epochs_stage1):
            for batch in loader(train_set):
                train_stage1_localization(model, batch)

        # -------------------------------
        # Stage II: train diagnosis network
        # -------------------------------
        sampler = coca_batch_sampler(train_set)

        for epoch in range(num_epochs_stage2):
            for batch in sampler:
                train_stage2_diagnosis(model, batch)

        # -------------------------------
        # Validation:
        # 根据内部验证集选择 operating point
        # 例如 high-specificity 或 high-sensitivity
        # -------------------------------
        val_outputs = inference_on_validation_set(model, val_set)

        tau_high_specificity = select_threshold(
            val_outputs,
            target_specificity=0.99
        )

        tau_high_sensitivity = select_threshold(
            val_outputs,
            target_sensitivity=0.953
        )

        model.tau_final = tau_high_specificity

        trained_models.append({
            "fold_id": fold_id,
            "model": model,
            "tau_high_specificity": tau_high_specificity,
            "tau_high_sensitivity": tau_high_sensitivity
        })

    return trained_models
```

---

## 9. 推理阶段伪代码

```python
def inference_coca(model, image, mode="high_specificity"):
    """
    image: [1, 1, D, H, W]
    mode:
        "high_specificity" 用于大规模机会性筛查
        "high_sensitivity" 用于更高敏感度筛查
    """

    if mode == "high_specificity":
        model.tau_final = model.tau_high_specificity
    elif mode == "high_sensitivity":
        model.tau_final = model.tau_high_sensitivity

    outputs = model.forward(image)

    result = {
        "crc_probability": outputs["cls_prob"],
        "tumor_probability_map": outputs["tumor_prob"],
        "segmentation_mask": argmax(outputs["seg_prob"], dim=1),
        "tumor_volume": outputs["tumor_volume"],
        "final_score": outputs["final_score"],
        "prediction": "CRC" if outputs["final_pred"] else "normal"
    }

    return result
```

---

## 10. 总体结构一行总结

```text
Input CT
  → Stage I U-Net
      → colorectal / intestinal ROI mask
      → crop ROI
          → Stage II multi-task U-Net
              → segmentation head: background / colorectum / CRC tumor
              → decoder multi-scale pooling + concat
                  → classification head: normal vs CRC
              → classification probability + tumor segmented volume
                  → final diagnosis
```
