# Diffusion 模型核心学习笔记（从 U-Net 到 DiT 的桥梁）

> 定位：Diffusion（路线甲 · 站 E）的阶段总结，偏「直觉 + 工程落地」，不替代公式推导。
> 前置：`notes/知识点/VAE.md`（ELBO / 重参数化）、`notes/知识点/分层VAE.md`（T 层特例）。
> 配套代码：本目录 4 个 `.py` + 2 个 notebook，对应关系见 §7。
> 核对时间：2026-11-02；行号以当前文件为准。

## 目录

- [0. 速览](#0-速览)
- [1. 核心哲学](#1-核心哲学)
- [2. 前向与反向](#2-前向与反向)
- [3. U-Net 拆解](#3-u-net-拆解)
- [4. CFG（无分类器引导）](#4-cfg无分类器引导)
- [5. 工程技巧](#5-工程技巧)
- [6. 走向 DiT](#6-走向-dit)
- [7. 代码地图](#7-代码地图)
- [8. 炼丹心得](#8-炼丹心得)

---

## 0. 速览

| 问题 | 一句话答案 |
|---|---|
| 网络学什么 | 预测噪声 $\epsilon$（等价于学 score / 数据分布） |
| 训练怎么加噪 | 一步到位，不写 `for`：$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$ |
| 怎么生成 | 从 $x_T\sim\mathcal N(0,I)$ 倒着迭代 $T$ 步：算均值 + 加方差 |
| 条件怎么进 | 类别 → FiLM 加性注入；文本 → Cross-Attention |
| 质量怎么提 | CFG：训练丢 10% 条件，采样跑两次网络 |
| 工程三件套 | Cosine 调度、EMA（本仓库待补）、加载对权重 |

---

## 1. 核心哲学

- **Transformer 的本质**：序列到序列的映射，基于上下文预测下一个 Token。
- **Diffusion 的本质**：一个**无限自监督数据生成引擎**。它把清晰的图 $x_0$ 加噪变成 $x_t$，然后训练一个网络（U-Net / DiT）去预测加了多少噪声 $\epsilon$。
- **为什么是「无限」**：同一张图，每个时间步 $t$、每次采样到的 $\epsilon$ 都构成一个新样本。数据量随训练步数增长，且不需要人工标注。
- **曲线救国**：网络表面上学的是「预测噪声」，骨子里学的是「什么是真实的自然图像」。学「差异（噪声）」比学「本体（原图）」容易得多。
- **为什么预测 $\epsilon$ 而不是 $x_0$**：$\epsilon$ 与 $x_t$ 同尺度、数值稳定；且 $\epsilon$ 与 score $\nabla_{x_t}\log p(x_t)$ 只差一个系数，预测噪声等价于学数据分布的梯度场。

---

## 2. 前向与反向

### 2.1 前向过程（加噪）：一步到位，拒绝循环

- **理论**：一步步加高斯噪声，直到变成纯噪声 $x_T$。
- **工程落地**：绝对不写 `for` 循环。利用重参数化技巧，**一步到位**：

$$
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,I)
$$

- **`extract` 函数**：把一维系数数组按随机抽出的时间步 $t$ 取出，reshape 成 `[B, 1, 1, 1]`，再利用 PyTorch 广播与图片直接相乘。
  - 代码：`Diffusion_basic.py:20-26`（`torch.gather` + `view([B] + [1]*(ndim-1))`）。
- **训练一步的真实流程**（`Diffusion_basic.py:77-86`）：
  1. `t = torch.randint(T, ...)`：每个样本独立抽一个时间步；
  2. `noise = torch.randn_like(x_0)`；
  3. 用上面的公式合成 $x_t$；
  4. `loss = MSE(model(x_t, t), noise)`。

> **注意 loss 的归约方式**：代码写的是 `loss = trainer(x_0).sum() / 1000.`（`Diffusion_basic.py:393`），不是 `mean()`。
> 这意味着 loss 会随 batch size 线性放大；换 batch 时要同步重调学习率，不能只盯着 loss 数值判断好坏。

### 2.2 反向过程（去噪）：计算均值 + 加入方差

反向过程由贝叶斯公式推导，最终落地为一个极简公式：

$$
x_{t-1}
=\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\hat\epsilon_\theta(x_t,t)
\right)
+\sigma_t z
$$

- **两部分组成**：
  1. **均值（去除噪声）**：依赖网络预测的噪声 $\hat\epsilon_\theta$；
  2. **方差（重塑随机性）**：加上 $\sigma_t z$（$z\sim\mathcal N(0,I)$），保证生成多样性。**最后一步（$t=0$）不加噪声**，保证最终像素纯净。
- **代码对照**（`Diffusion_basic.py:89-130`）：
  - `coeff1 = 1/sqrt(alpha)`、`coeff2 = coeff1*(1-alpha)/sqrt(1-alpha_bar)`（L99-100），正好对应公式里的两个系数；
  - `predict_xt_prev_mean_from_eps` 只算均值（L103-108）；
  - 采样循环里 `if time_step > 0: noise = randn_like(x_t) else: noise = 0`（L123-126），最后 `clip(-1, 1)`（L130）。
- **一个容易忽略的细节：$\sigma_t$ 到底取什么？**
  - 理论后方方差：$\tilde\beta_t=\dfrac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}$（L101 的 `posterior_var`）；
  - 实际采样：`var = torch.cat([posterior_var[1:2], betas[1:]])`（L111），即 $t\ge 1$ 时直接用 $\beta_t$（DDPM 原实现的 `fixedlarge` 做法），$t=0$ 用哪一项都无所谓，因为那一步噪声置 0。
  - `mini_stable_diffusion.py:260-264` 也是同一套写法（`sqrt(beta_t)`，最后一步只取均值）。

---

## 3. U-Net 拆解

U-Net 是 CNN 时代的绝对霸主。核心结构是：**编码器（下采样）→ 瓶颈层（Attention）→ 解码器（上采样 + Skip Connection）**。

### 3.1 与 Transformer 的术语映射

| Transformer | U-Net（本仓库） | 代码位置 |
|---|---|---|
| Positional Encoding | **TimeEmbedding**：正弦/余弦查表 + 两层 `Linear` + Swish | `Diffusion_basic.py:141-167` |
| Token Embedding | **ConditionalEmbedding**：`num_labels+1`，0 号留给「无条件」 | `Classifier_free_guidance_diffusion.py:49-62` |
| `nn.Linear` | **1×1 Conv**：对每个像素独立做全连接 | `Diffusion_basic.py:203-206` |
| `LayerNorm` | **GroupNorm(32, C)**：按通道分组算均值方差，不受 batch size 影响 | `Diffusion_basic.py:238, 247` |
| FFN + Residual | **ResBlock**：GN → Swish → Conv → +temb → GN → Swish → Dropout → Conv → +shortcut | `Diffusion_basic.py:234-275` |

> **TimeEmbedding 的一个细节**：正弦表本身在 `Diffusion_basic.py` 里是**冻结**的（`nn.Embedding.from_pretrained(emb)` 默认 `freeze=True`），真正可学习的是后面两层 `Linear`；CFG 版本显式写了 `freeze=False`（`Classifier_free_guidance_diffusion.py:40`）。

### 3.2 外部条件如何注入？（重要！）

**不是把条件拼在输入层，而是在每个 ResBlock 里注入。** 但「注入」有两条不同路线，别混：

| 条件类型 | 注入方式 | 代码位置 |
|---|---|---|
| **类别标签**（CIFAR-10） | **FiLM 加性**：`h += temb_proj(temb)[:, :, None, None]`、`h += cond_proj(cemb)[:, :, None, None]` | `Classifier_free_guidance_diffusion.py:114-125` |
| **时间步** | 同样是 FiLM 加性，先 `Linear` 到 `out_ch` 再广播 | `Diffusion_basic.py:242-245, 269-275` |
| **文本**（mini-SD） | **Cross-Attention**：图像特征做 Q，文本 token 做 K/V，插在 ResBlock 末尾 | `mini_stable_diffusion.py:74-115, 131-140` |

- 加性 FiLM 的公式：
  $$
  h \leftarrow h+\mathrm{Linear}_{t}(t)[:, :, \mathrm{None}, \mathrm{None}]
  +\mathrm{Linear}_{c}(c)[:, :, \mathrm{None}, \mathrm{None}]
  $$
- **为什么每层都要注入**：网络一深就容易「忘了」条件，每一步都要提醒它当前的「火候（$t$）」和「目标（$c$）」。
- **为什么不在输入层拼**：拼在输入层只有第一层看得到，还会改变输入通道/空间结构；逐层调制更稳、更可控。
- **文本为什么不用 FiLM**：文本是变长 token 序列，需要保留 token 级信息（`a photo of a dog` 里关键是 `dog`）。Cross-Attention 让每个图像位置自己去「查」相关的词，比压成一个向量再加性更强。

> **无条件标签的实现细节**：`nn.Embedding(..., padding_idx=0)` 会把 0 号向量固定为零且不更新（`Classifier_free_guidance_diffusion.py:55`）。所以「无条件」实际走的是后面 `Linear` 的 bias，等价于一个常数条件向量；能工作，但和「学一个 null token」不是一回事。

### 3.3 Attention 放在哪 / Skip 怎么接

- Attention 位置：中间瓶颈层必开；`Diffusion_basic.py` 的配置 `attn=[1]` 让第 1 组下采样/上采样也开（`UNet` 里 `attn=(i in attn)`，L292-294、L310-312）。
- Skip Connection 是 **concat**，不是 add：`h = torch.cat([h, hs.pop()], dim=1)`（L342）。上采样路径的输入通道因此是「当前通道 + 对应下采样通道」。
- `mini_stable_diffusion.py` 的 MiniUNet 因为通道对齐，Skip 用的是 **add**（L223）——结构简化版，不要和标准 U-Net 的 concat 混淆。

---

## 4. CFG（无分类器引导）

### 4.1 核心公式

$$
\tilde\epsilon_\theta(z,c)
=(1+w)\cdot\epsilon_\theta(z,c)-w\cdot\epsilon_\theta(z,\varnothing)
$$

- $\epsilon_\theta(z,c)$：带条件预测（往死里画狗）。
- $\epsilon_\theta(z,\varnothing)$：不带条件预测（随便画个毛茸茸的）。
- $w$：CFG Scale（引导系数）。类别条件常用 3.0~5.0；文生图常用 7~12（SD 默认 7.5）。
- **换个写法更好懂**：
  $$
  \tilde\epsilon=\epsilon_\theta(z,c)+w\big(\epsilon_\theta(z,c)-\epsilon_\theta(z,\varnothing)\big)
  $$
  即在条件预测的基础上，沿「条件 − 无条件」的方向**外推** $w$ 倍。
- 代码注释里把它写作「原文公式 6」（`Classifier-free Guidance explanation.ipynb` 的 `p_mean_variance`）。

### 4.2 训练时：Condition Dropout

- 有 **10%** 的概率把条件 $c$ 替换为 0（无条件），这叫 Condition Dropout：
  ```python
  drop_mask = torch.rand(labels.shape, device=device) < 0.1
  labels = torch.where(drop_mask, torch.zeros_like(labels), labels)
  ```
  代码：`Classifier_free_guidance_diffusion.py:320-322`。
- 标签先 `+1`，让真实类别落在 1~10，**0 永远留给「无条件（Null）」**（L318；`ConditionalEmbedding` 的 `num_labels+1` 在 L55）。
- **关键**：训练时必须同时见过「有条件」和「无条件」，否则采样时无条件分支是瞎的，CFG 直接失效。
- 注意 dropout 是 **per-sample** 的 mask，不是整个 batch 一起丢。

### 4.3 采样时：网络跑两次

- 同一张 $z_t$ 前向两次：一次带条件，一次把标签置 0；再用公式融合：
  ```python
  eps    = model(x_t, t, labels)
  nonEps = model(x_t, t, torch.zeros_like(labels))
  eps    = (1. + w) * eps - w * nonEps
  ```
  位置：`Classifier-free Guidance explanation.ipynb` → `GaussianDiffusionSampler.p_mean_variance`。
- **差异提醒**：本目录的 `.py` 里只有 **CFG 的训练**（`Classifier_free_guidance_diffusion.py` 到 `train()` 结束）；**采样代码在 notebook**，不在 `.py`。找采样公式时别翻错文件。
- 本仓库只有「类别版」实现了 CFG；`mini_stable_diffusion.py` **没有** CFG（没有 null 条件、没有 dropout），它的条件只走 Cross-Attn。

### 4.4 代价与调参

- $w$ 越大：图像越锐利/保真，但多样性下降，甚至出现色彩饱和（塑料感）、色块。
- $w=0$：退化成普通条件采样；$w$ 太大：条件方向被过度放大，容易过饱和/崩坏。
- 调参顺序：先确认 $w=0$ 能出合理图，再逐步加到 3~5；不要用大 $w$ 掩盖训练不足。

---

## 5. 工程技巧

### 5.1 Cosine Noise Schedule（余弦调度）

- 早期的线性调度（Linear）在后期毁图太快；余弦调度让 $\bar\alpha_t$ 下降得更均匀，**最后才归零**，因此高频细节保留得更久。
- 代码公式（`Classifier_free_guidance_diffusion.py:198-205`）：
  $$
  \bar\alpha_t=
  \frac{\cos^2\!\left(\dfrac{t/T+s}{1+s}\cdot\dfrac{\pi}{2}\right)}
  {\cos^2\!\left(\dfrac{s}{1+s}\cdot\dfrac{\pi}{2}\right)},
  \qquad s=0.008
  $$
  $$
  \beta_t=1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}},
  \qquad \beta_t\in[10^{-4},\,0.9999]
  $$
- 先 `linspace(0, T, T+1)` 建表，再除以 `alphas_bar[0]` 归一化，保证 $t=0$ 时 $\bar\alpha=1$。
- **重要差异**：Cosine 只出现在 CFG 脚本里；`Diffusion_basic.py`、`Diffusion_with_upsampler.py`、`mini_stable_diffusion.py` 仍然是**线性 $\beta$**（$10^{-4}\to0.02$）。改调度时要四个文件分别看，别以为全局生效。

### 5.2 采样方差：posterior 还是 $\beta_t$？

- 理论后方方差 $\tilde\beta_t$ 与「直接固定为 $\beta_t$」在 $T=1000$、$\beta$ 很小时几乎一样。
- 本仓库统一采用 `fixedlarge`：$t\ge1$ 用 $\beta_t$（`Diffusion_basic.py:111`、`mini_stable_diffusion.py:256-262`）。
- 这不影响训练目标，只影响采样随机性的大小；换实现时不必纠结。

### 5.3 EMA（指数移动平均）

- **灵魂所在**：训练时维护一份「平滑版」权重。训练后期的参数在最优点附近疯狂震荡，EMA 取的就是它们的中轴线，相当于免费集成。
- **参考写法**（本仓库尚未接入）：
  ```python
  ema.update_model_average(ema.model, net_model)
  torch.save({
      'net_model': net_model.state_dict(),
      'ema_state_dict': ema.model.state_dict(),
  }, path)
  # 采样时
  model.load_state_dict(ckpt['ema_state_dict'])
  ```
- **测试时务必加载 `ema_state_dict`**，否则图会充满噪点/发糊。
- **本仓库现状**：4 个 `.py` 的 checkpoint 里都只有普通权重，没有 EMA（basic / CFG / upsampler 存 `net_model`，mini-SD 存 `vae` + `unet` + `text_encoder` 三份：`Diffusion_basic.py:406`、`Classifier_free_guidance_diffusion.py:343-345`、`Diffusion_with_upsampler.py:443, 550`、`mini_stable_diffusion.py:374-378`）。要加就在 `optimizer.step()` 之后更新，并让 eval 优先加载 EMA 权重。
- 两个坑：
  1. EMA 的 buffer 也要一起同步/加载（本仓库用 GroupNorm，没有 BN running stats，问题较小）；
  2. 训练步数少时别用 0.999 这种大 decay（跟不上）；短训练先用 0.99。

---

## 6. 走向 DiT

搞懂 U-Net 后，可以直接把「骨架」扔掉，换成 Transformer：

1. **Patchify**：把图片切成小块（Token 化）。（本仓库还没实现；`mini_stable_diffusion.py` 用 VAE 把 $32\times32$ 压到 $8\times8$ 潜变量，是另一条「减少 token」的路。）
2. **去掉下采样**：全程保持 Token 数量不变，所有 block 同尺寸。
3. **adaLN-Zero**：把时间编码和条件编码拆成 $\gamma$ 和 $\beta$，直接调制 LayerNorm：
   $$
   \mathrm{adaLN}(h)=\gamma\odot\mathrm{LN}(h)+\beta
   $$
   $\gamma,\beta$ 零初始化（再加一个门控 $\alpha$），让每个 block 初始是恒等映射，训练更稳。
   
   - 对比本仓库的 FiLM：FiLM 是「加一个 bias 向量」，adaLN 是「缩放 + 平移」，表达力更强。
4. **优势**：彻底摆脱 CNN 的局部视野限制，可以无缝吃透 **Scaling Law**（堆参数、堆数据）。Sora 的底层架构就是 DiT（视频 patch 成 token）。

---

## 7. 代码地图

**进化路线（建议按这个顺序读）：**

```
Diffusion_basic.py               无条件 DDPM（CIFAR-10 32×32）
  → Classifier_free_guidance_diffusion.py   + 类别条件 + CFG 训练 + Cosine
  → Diffusion_with_upsampler.py            + SRNet 4× 放大（两阶段）
  → mini_stable_diffusion.py               VAE 潜空间 + 文本 Cross-Attn（迷你 LDM）
```

| 文件 | 内容 | 关键位置 | 对应章节 |
|---|---|---|---|
| `Diffusion_basic.py` | 无条件 DDPM：U-Net + 训练 + 采样 | `extract` L20-26；Trainer L66-86；Sampler L89-130；UNet L278-346 | §2, §3 |
| `Classifier_free_guidance_diffusion.py` | 类别条件 + CFG 训练 + Cosine 调度 | 条件编码 L49-62；FiLM L114-125；Cosine L198-212；丢条件 L320-322 | §3.2, §4.2, §5.1 |
| `Classifier-free Guidance explanation.ipynb` | CFG 采样与逐行讲解 | `GaussianDiffusionSampler.p_mean_variance` / `forward` | §4.1, §4.3 |
| `Diffusion_with_upsampler.py` | DDPM 生成 32×32，再 SRNet 4× 放大到 128×128 | `SRNet` L360-385；造数据 L446-482；`train_sr` L485-551 | §7 注 3 |
| `mini_stable_diffusion.py` | 迷你 Latent Diffusion：VAE + 文本编码 + Cross-Attn U-Net | `MiniVAE` L47-69；`CrossAttnBlock` L74-115；`MiniUNet` L170-225；`Diffusion` L231-265 | §3.2, §6 |
| `codeblock_explanation.ipynb` | 基础 DDPM 代码逐块解释（约 1 MB，含图） | — | 全部 |

**三个读代码时容易误解的点：**

1. `mini_stable_diffusion.py` 的 `MiniVAE` 实际是 **AE**：`encode` 直接返回确定性 $z$，没有 KL、没有重参数化，损失只有 `MSE(recons, imgs)`（L319-322）。它只负责「压缩/还原」，训练完就冻结（L329-331），不承担生成。
2. mini-SD 的 `TextEncoder` 只有 2 层 Transformer、词表 14 个词（L11-12），本质是「把类别名 token 化」，不是 CLIP。
3. `Diffusion_with_upsampler.py` 的 HR 目标是把真实 32×32 图 **bicubic 放大**到 128×128（L477），不是原生高清图；所以这里的 SR 实质是「放大 + 去生成伪影」，上限受 bicubic 限制。

---

## 8. 炼丹心得

1. 图模糊说明 Loss 没降下去，继续加 epoch；但先确认 EMA/权重加载正确。
2. 图变纯黑/纯白色块，通常是 CFG 的 $w$ 太大，或者权重加载损坏。
3. 永远相信 Cosine 调度和 EMA——但要注意本仓库只有 CFG 脚本用了 Cosine，EMA 还没接。
4. 换 batch size 时，记得 loss 是 `sum()/1000` 而不是 `mean`，等效学习率会变。
5. 采样慢是正常的：朴素 DDPM 要跑 $T$ 次前向（$T=1000$）。

---

