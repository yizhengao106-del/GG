# G：原版 ZoomNeXt 基础上的五级深度结构引导

本包是 **原版 EffB1 ZoomNeXt + G**，不是 F+G。五个 `hmu_5` 至 `hmu_1` 均部署 G。原来 `06_G_深度结构引导` 中的 F+G 版本保持不变；本次应上传本文件所在的新文件夹。

## 实際改动位置

核心文件为 `methods/zoomnext/depth_guided.py`。

- `EffB1_ZoomNeXt_DepthGuided` 继承原版 `EffB1_ZoomNeXt`，保留编码器、tra、MHSIU、相邻层相加及预测头。
- `DepthGuidedRGPU` 直接继承原版 `RGPU`。保留原版分组、顺序组间传播和残差输出；没有 F 的共享上下文、组间关系分支。
- 新增 `DepthStructureCondition`：从 RGB 中间特征学习深度结构表示，RGB 查询与深度特征键/值进行通道交叉注意力，生成空间相关的调制修正量。
- 原调制 logits 为 `[B,G*d,1,1]`，新修正量为 `[B,G*d,H,W]`；相加后按原 Softmax 的通道维归一化。不是仅在 G 个组之间做 Softmax，也不是空间位置之间的自注意力。
- 修正投影零初始化，初始 RGPU 输出与原版一致；训练后由结构特征修正原调制权重。
- 五级深度预测头用于辅助监督：`BCE + UAL + 0.1 * mean_stage(L1 + 1 - SSIM)`。0.1 是本次设置，不是已验证的最优值。

因此本次保留了深度辅助学习、跨特征通道交互和空间调制三个环节，但没有引入 F。训练需配对伪深度，测试仅需 RGB；测试跳过深度预测头，但深度特征分支和注意力仍运行，存在额外推理开销。

## 来源与区别

参考 [OVCamo 官方仓库](https://github.com/lartpang/OVCamo)，固定提交 `419c16beea40a9cf0458199f1172b8bdd37d505d`。论文为 *Open-Vocabulary Camouflaged Object Segmentation*，ECCV 2024；OVCamo 是数据集名，OVCoser 是模型名。

参考其 RGB 派生深度特征、结构增强中的通道交叉注意力和深度辅助损失。`ovcamo_loss.py` 保留其损失实现，许可证见 references。本包采用 GroupNorm，将结构信息用于 ZoomNeXt 原 RGPU 的输出调制，并扩展到五级；没有复现其整个开放词汇模型。

Depth Anything V2 Small 是本包可选的离线伪深度生成方案，不宣称是上述论文采用的教师。代码修改本身不能证明学术新颖性或精度提升。

## 三个可运行配置

| variant | 结构 | 训练深度标签 |
|---|---|---|
| baseline | 原版 ZoomNeXt，完全关闭 G | 不需要 |
| g_aux | 原版 + 深度辅助监督，关闭调制修正 | 需要 |
| g_only | 原版 + 完整 G（默认） | 需要 |

`g_aux` 保留完整分支参数便于控制变量，但关闭调制时注意力分支不会影响分割输出。

## 上传和运行

建议将整个本文件夹上传为 `/home/gyz/ZoomNeXt-G-Original`，保留相对目录。数据不包含在包中，沿用服务器原有数据。不要覆盖原来的 F+G 文件夹。

先检查代码（不会启动长训练）：

```bash
cd /home/gyz/ZoomNeXt-G-Original
/home/gyz/anaconda3/envs/zoomnext/bin/python check_g.py
```

若尚无伪深度，可在独立的教师环境安装 `requirements-depth.txt` 后执行（第一次需获取模型权重）：

```bash
CUDA_VISIBLE_DEVICES=1 python prepare_depth.py --data-cfg /home/gyz/ZoomNeXt/dataset.yaml --output /home/gyz/ZoomNeXt-G-Original/pseudo_depth_v1 --device cuda:0
```

上面仅处理训练集配对 RGB 图片。输出目录必须不存在，生成后留下教师和图片清单。已有对齐伪深度可直接在 YAML 的训练集条目添加 `depth: {path: 绝对目录, suffix: .npy}`；要求同名、同原图尺寸、二维、有限数值且归一化到 [0,1]。

完整 G 训练（确认 GPU 空闲后由你提交）：

```bash
cd /home/gyz/ZoomNeXt-G-Original
nohup /home/gyz/anaconda3/envs/zoomnext/bin/python -u run_g.py --variant g_only --gpu 1 --data-cfg /home/gyz/ZoomNeXt-G-Original/pseudo_depth_v1/dataset_with_depth.yaml > train_g_original.log 2>&1 < /dev/null &
```

每次训练自动保存源码、配置、环境和哈希快照到 `g_original_runs`。训练仍沿用原版配置的轮数、优化器等；新增深度损失是明确的实验变更。数据配对数量警告需要核对，不能因程序可运行就认为划分正确。

## 已完成验证与限制

CPU PyTorch 2.1.2：三组配置前向/反向通过；五级均调用；关闭 G 后严格加载原版参数且输出、损失一致；零初始化 RGPU 等价；调制和注意力梯度检查通过；384×384 RGB 推理通过；训练加载器的翻转对齐、深度不受颜色增强影响及缺失文件拒绝检查通过。

原版参数量 10,283,588；G 参数量 10,485,993，新增 202,405。详见 `validation_g.txt` 和 `validation_g.json`。

尚未运行真实教师生成、服务器 GPU/AMP 长训练及数据集精度评估，不保证提升。
