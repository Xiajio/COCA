"""
nii2npy_simple.py — 简单的 NIfTI 到 npy 转换脚本

功能:
- 直接将 NIfTI 文件转换为 npy 格式
- 同时处理图像数据和肿瘤分割掩码
- 不做任何预处理（无裁剪、无归一化、无重采样）
- 保持原始数据的完整性和分辨率
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm


# ============================================================
# 配置区 — 根据实际路径修改
# ============================================================
DICOM_DIR = r"G:\CMS\CMS\CMS_NAC\Dicom"
LABEL_DIR = r"G:\CMS\CMS\CMS_NAC\Lable"
OUTPUT_DIR = r"G:\CMS_class\CMS_npy_simple\Dicom"
OUTPUT_LABEL_DIR = r"G:\CMS_class\CMS_npy_simple\labels"

# 处理模式: 'both' (图像和标签), 'image' (仅图像), 'label' (仅标签)
PROCESS_MODE =  'label'


# ============================================================
# 主处理流程
# ============================================================

def process_and_save():
    if PROCESS_MODE in ['both', 'image']:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"输出目录: {OUTPUT_DIR}")
    if PROCESS_MODE in ['both', 'label']:
        os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
        print(f"标签输出目录: {OUTPUT_LABEL_DIR}")

    print(f"处理模式: {PROCESS_MODE}")
    print()

    processed_info = []
    processed_labels = []

    if PROCESS_MODE in ['both', 'image']:
        nii_files = [f for f in os.listdir(DICOM_DIR) if f.endswith('.nii.gz')]
        print(f"找到 {len(nii_files)} 个 NIfTI 文件")
        print(f"开始转换图像...")
        print()

        for filename in tqdm(nii_files, desc="转换图像中"):
            img_path = os.path.join(DICOM_DIR, filename)
            crf_id = os.path.splitext(os.path.splitext(filename)[0])[0]

            try:
                img_nii = nib.load(img_path)
                img_data = img_nii.get_fdata()

                save_name = f"{crf_id}.npy"
                save_path = os.path.join(OUTPUT_DIR, save_name)
                np.save(save_path, img_data)

                spacing = np.sqrt(np.sum(img_nii.affine[:3, :3] ** 2, axis=0))

                info = {
                    'filename': save_name,
                    'crf_id': crf_id,
                    'original_file': filename,
                    'spacing_d': float(spacing[0]),
                    'spacing_h': float(spacing[1]),
                    'spacing_w': float(spacing[2]),
                    'shape_d': img_data.shape[0],
                    'shape_h': img_data.shape[1],
                    'shape_w': img_data.shape[2],
                    'data_min': float(img_data.min()),
                    'data_max': float(img_data.max()),
                }
                processed_info.append(info)

            except Exception as e:
                print(f"转换 {filename} 时出错: {e}")

    if PROCESS_MODE in ['both', 'label']:
        label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.nii.gz')]
        print(f"\n找到 {len(label_files)} 个标签文件")
        print(f"开始转换标签...")
        print()

        for filename in tqdm(label_files, desc="转换标签中"):
            label_path = os.path.join(LABEL_DIR, filename)
            crf_id = os.path.splitext(os.path.splitext(filename)[0])[0].replace('-Tumor-label', '').replace('_Tumor-label', '')

            try:
                label_nii = nib.load(label_path)
                label_data = label_nii.get_fdata()

                save_name = f"{crf_id}_label.npy"
                save_path = os.path.join(OUTPUT_LABEL_DIR, save_name)
                
                label_data_uint8 = label_data.astype(np.uint8)
                np.save(save_path, label_data_uint8)

                label_info = {
                    'filename': save_name,
                    'crf_id': crf_id,
                    'original_file': filename,
                    'shape_d': label_data.shape[0],
                    'shape_h': label_data.shape[1],
                    'shape_w': label_data.shape[2],
                    'unique_values': len(np.unique(label_data)),
                }
                processed_labels.append(label_info)

            except Exception as e:
                print(f"转换标签 {filename} 时出错: {e}")

    if processed_info:
        info_df = pd.DataFrame(processed_info)
        info_csv_path = os.path.join(OUTPUT_DIR, 'dataset_info.csv')
        info_df.to_csv(info_csv_path, index=False)
        print(f"\n图像数据列表: {info_csv_path}")

    if processed_labels:
        label_df = pd.DataFrame(processed_labels)
        label_csv_path = os.path.join(OUTPUT_LABEL_DIR, 'label_info.csv')
        label_df.to_csv(label_csv_path, index=False)
        print(f"标签数据列表: {label_csv_path}")

    print("\n" + "=" * 60)
    print("转换完成！")
    if processed_info:
        print(f"共生成 {len(info_df)} 个图像 .npy 文件")
    if processed_labels:
        print(f"共生成 {len(label_df)} 个标签 .npy 文件")


if __name__ == "__main__":
    process_and_save()
