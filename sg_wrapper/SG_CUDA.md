# SG 推理颜色 CUDA 核

此扩展融合每个高斯、每个相机的 SG 颜色计算，输出仍交给 gsplat 标准光栅化路径。源码为项目自身实现，位于 `sg_wrapper/cuda/sg_colors.cu`；不复制或修改第三方包。当前 gsplat 的标准 3DGS SH 路径在像素混合前求颜色，并非逐像素求 SH。

## 使用

在原有推理命令前设置 `SG_COLOR_BACKEND=cuda`。默认 `torch` 保持原路径。扩展首次调用会 JIT 编译，需要 CUDA 工具链；失败会抛出错误，不静默掩盖。未实现 backward。

仅 eval 且关闭 autograd、CUDA FP32、dense lobe 表示时使用融合核。训练、有梯度的姿态对齐、CPU、非 FP32 和 packed 表示继续使用原路径。开启开关不保证评测的所有阶段都使用 kernel。

## 语义与成本

输入为 B 个场景、每场景 V 个相机；共享 lobe 张量不沿 V 复制。每个线程处理一个相机的一颗高斯，直接输出 RGB，避免方向权重和逐 lobe RGB 中间数组。视线为点到相机；视线分母为 norm+eps，lobe 方向分母为 max(norm,eps)。锐度与振幅已在适配器激活/缩放，基础颜色可已补偿，此处不重复处理。

任意槽位的全零 RGB 振幅跳过 exp 和方向运算；仍遍历 dense 槽位，不声称实现 packed 存储或真实变长数据布局。有限输入是正常模型契约，零振幅跳过不保留 NaN/Inf 输入的传播语义。未启用 fast-math，禁用 FMA 收缩以减小参考路径差异；不同求和顺序仍可能产生浮点误差。

这是一颗独立颜色求值核，并未融合进 gsplat 像素混合核。颜色计算不会变成零成本；性能与质量需验证，不能预先承诺毫秒数或与 EcoSplat 的差距。

## 验证

在仓库根目录、具有 torch 和 gsplat 的环境执行：

```bash
python sg_wrapper/tests/test_sg_cuda.py --cpu-only
python sg_wrapper/tests/test_sg_cuda.py
```

第二条首次包含构建，由实验执行者手动运行。覆盖 L=0/1/3/6、多场景多相机、零方向、极小方向、相机与高斯重合、任意删除槽位、clamp、空点集、梯度拒绝和微型 gsplat RGB/depth/alpha 对拍。颜色与图像绝对容差 1e-5。

通过后用同一 checkpoint、索引、预算、姿态对齐配置，分别设置 `SG_COLOR_BACKEND=torch` 与 `SG_COLOR_BACKEND=cuda` 跑现有评测。正式计时应先预热并完成 JIT，用 CUDA events 或同步计时；同时报告 sg_color、raster 和端到端时间。小样本对拍不代替 200 场景质量与性能验收。
