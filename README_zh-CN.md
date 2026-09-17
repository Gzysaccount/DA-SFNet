# DA-SFNet 复现代码

本仓库整理了论文 DA-SFNet 的完整可运行工程，包括修改后的 Ultralytics 源码、DA-FFT 与 MST-LADA 模块、最终模型配置、训练与测试脚本、发布权重、FloorPlanCAD 数据、双模块消融记录及 Gradio 可视化测试界面。

## 快速使用

```bash
# 请先按显卡与 CUDA 版本安装 PyTorch
pip install -e .
pip install -r requirements-app.txt

python scripts/check_dataset.py
python scripts/predict.py --source data/FloorPlanCAD/images/test
python app.py
```

训练与测试：

```bash
python scripts/train.py --device 0
python scripts/val.py --split test --device 0
```

默认训练参数与最终实验记录保持一致：200 epochs、imgsz 640、batch 8、SGD、初始学习率 0.01、momentum 0.937、warmup 3 epochs、seed 1、deterministic=True、最后 30 epochs 关闭 Mosaic。

完整说明、仓库结构、实验结果和许可信息请阅读 [README.md](README.md)。数据图片与模型权重已配置 Git LFS，首次推送前需执行 `git lfs install`。

