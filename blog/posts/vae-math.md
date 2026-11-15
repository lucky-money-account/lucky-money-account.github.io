# VAE — 从数学理论到算法实现

> **记号**：$x$ 观测数据；$z$ 隐变量；$\phi$ 解码器（生成）参数；$\theta$ 编码器（变分）参数。
> $p(z)=\mathcal N(0,I)$ 先验；$p_\phi(x\mid z)$ 解码器；$q_\theta(z\mid x)$ 编码器（近似后验）。
> **目标**：最大化数据似然（evidence）$p_\phi(x)$ —— 即"模型产生数据 $x$ 的概率"。

## 1. 起点：积分不可解

$$
\log p_\phi(x) = \log \left[ \int p_\phi(x, z)\, dz \right]
$$

该积分要对所有 $z$ 积分，无法直接计算。于是引入一个"替身"分布——变分后验 $q_\theta(z\mid x)$：

$$
\log p_\phi(x)
= \log \left[ \int q_\theta(z\mid x)\, \frac{p_\phi(x, z)}{q_\theta(z\mid x)}\, dz \right]
= \log \mathbb E_{z \sim q_\theta(z\mid x)} \left[ \frac{p_\phi(x, z)}{q_\theta(z\mid x)} \right]
$$

## 2. 套 Jensen 不等式 → ELBO

$g(y)=\log y$ 是凹函数，Jensen 不等式为 $g(\mathbb E[y]) \ge \mathbb E[g(y)]$，即 $\log \mathbb E[y] \ge \mathbb E[\log y]$：

$$
\log p_\phi(x)
\;\ge\; \mathbb E_{z \sim q_\theta(z\mid x)} \left[ \log \frac{p_\phi(x, z)}{q_\theta(z\mid x)} \right]
\;=:\; \mathrm{ELBO}
$$

（取等条件见第 5 节。）

## 3. 拆成两项：重构 + KL

把联合分布按 $p_\phi(x,z)=p_\phi(x\mid z)\,p(z)$ 展开：

$$
\begin{aligned}
\mathrm{ELBO}
&= \mathbb E_q \left[ \log p_\phi(x\mid z) + \log p(z) - \log q_\theta(z\mid x) \right] \\[2pt]
&= \underbrace{\mathbb E_q\left[\log p_\phi(x\mid z)\right]}_{\text{重构项}}
\;-\; \underbrace{D_{KL}\!\left(q_\theta(z\mid x)\,\|\,p(z)\right)}_{\text{正则项}}
\end{aligned}
$$

- **重构项**：在编码器给出的 $z$ 上，解码器还原 $x$ 的期望对数似然；
- **正则项**：让编码分布贴近先验 $p(z)$，使潜空间规整、可采样。

## 4. 实际计算

### 4.1 重构项：单样本蒙特卡洛 + 重参数化

重构项是期望，用蒙特卡洛估计；VAE 采用**单样本**（SGVB 估计器）：

$$
\mathbb E_{q_\theta(z\mid x)}\!\left[\log p_\phi(x\mid z)\right] \;\approx\; \log p_\phi(x \mid z^*),
\qquad z^* \sim q_\theta(z\mid x)
$$

但 $z^*$ 是采样得到的，**梯度在采样这一步断流**。重参数化把随机性外移到一个与参数无关的噪声上：

$$
z = \mu_\theta(x) + \sigma_\theta(x) \odot \varepsilon,
\qquad \varepsilon \sim \mathcal N(0, I)
$$

于是期望对象从"依赖 $\theta$ 的 $q$"变成了"**固定的 $\varepsilon$**"，梯度可以直接移进期望、顺着 $\mu_\theta,\sigma_\theta$ 回传：

$$
\nabla_\theta\, \mathbb E_{q_\theta(z\mid x)}[f(z)]
= \mathbb E_{\varepsilon}\!\left[ \nabla_\theta\, f\!\left(\mu_\theta(x) + \sigma_\theta(x)\odot\varepsilon\right) \right]
$$

（这也是为什么每个样本每次只采一个 $\varepsilon$ 就够：随机性只在 $\varepsilon$ 上，batch 平均会把它抹平。）

### 4.2 重构项代入高斯 → MSE

设解码器为高斯 $p_\phi(x\mid z)=\mathcal N(\hat x,\, \sigma^2)$（$\hat x$ 为解码器输出；这里的 $\sigma$ 是**观测噪声**，与潜变量的 $\sigma$ 无关）：

$$
\begin{aligned}
\log \left[\frac{1}{\sqrt{2\pi}\,\sigma}\exp\!\left(-\frac{(x-\hat x)^2}{2\sigma^2}\right)\right]
&= \log \frac{1}{\sqrt{2\pi}\,\sigma} - \frac{(x-\hat x)^2}{2\sigma^2}
\end{aligned}
$$

只有平方项依赖参数，所以 **最大化高斯似然 $\iff$ 最小化 $(x-\hat x)^2$（MSE）**。
（若数据是二值，改用伯努利似然 → **BCE**。$\sigma$ 固定时 MSE 权重是常数；可学习时模型自己平衡两项。）

### 4.3 KL 项：高斯闭式解

设 $q_\theta(z\mid x)=\mathcal N(\mu,\Sigma)$、先验 $p(z)=\mathcal N(0,I)$：

$$
D_{KL} = \frac{1}{2}\left( \mathrm{Tr}[\Sigma] + \mu^\top \mu - D_z - \log|\Sigma| \right)
$$

若 $q$ 取**对角**协方差 $\Sigma=\mathrm{diag}(\sigma_i^2)$：

$$
\mathrm{Tr}[\Sigma] = \sum_i \sigma_i^2, \qquad \mu^\top\mu = \sum_i \mu_i^2
$$

$$
\therefore\;
D_{KL} = \frac{1}{2}\sum_{i=1}^{D_z}\left(\sigma_i^2 + \mu_i^2 - 1 - \log\sigma_i^2\right)
$$

## 5. 为什么是"下界"：gap 等式

用贝叶斯 $p_\phi(x,z)=p_\phi(z\mid x)\,p_\phi(x)$ 回代：

$$
\begin{aligned}
\mathrm{ELBO}
&= \mathbb E_q\left[\log p_\phi(x,z) - \log q_\theta(z\mid x)\right] \\
&= \mathbb E_q\left[\log p_\phi(z\mid x) + \log p_\phi(x) - \log q_\theta(z\mid x)\right] \\
&= \log p_\phi(x) - D_{KL}\!\left(q_\theta(z\mid x)\,\|\,p_\phi(z\mid x)\right)
\end{aligned}
$$

$$
\therefore\quad
\log p_\phi(x) = \mathrm{ELBO} + \underbrace{D_{KL}\!\left(q_\theta(z\mid x)\,\|\,p_\phi(z\mid x)\right)}_{\text{gap}\;\ge\;0}
$$

- gap $=$ KL（近似后验 ‖ 真后验）$\ge 0$，所以 ELBO 是 $\log p(x)$ 的**下界**；
- 取等（gap $=0$）当且仅当 $q_\theta(z\mid x)$ 恰为真后验。

## 6. 从公式到代码

- 负重构项 $-E_q[\log p_\phi(x\mid z)]$ → `((imgs - out)**2).sum()`（高斯似然 → MSE）
- KL $D_{KL}(q_\theta(z\mid x)\;\|\;p(z))$ → 手写 KL 那一行（对角高斯闭式）
- 负 ELBO → `loss = reconstructed_loss + kl`（最小化负 ELBO = 最大化 ELBO）
- 重参数化采样 → `z = mu + std * torch.randn_like(mu)`（$z=\mu+\sigma\odot\varepsilon$）

## 7. 结论

**最大化 ELBO = 最小化（重构误差 + KL）**，同时优化两个网络：

- $\phi$（解码器 / 生成参数）：提高数据似然；
- $\theta$（编码器 / 变分参数）：让 $q$ 贴近真后验、收紧下界。

## 8. 两个易混点

1. **两个 KL**：ELBO 内部是 $q\,\|\,p(z)$（对**先验**）；gap 是 $q\,\|\,p(z\mid x)$（对**真后验**）。方向都是 $q\,\|\,\cdot$。
2. **两个 $\mu$**：$q$ 里的 $\mu$ 是**潜变量均值**；重构公式里的 $\hat x$ 是**解码器输出**（原稿两者都写作 $\mu$，容易混，这里改用 $\hat x$）。

## 9. 自测

1. Jensen 那一步为什么是 $\ge$？取等条件是什么？
2. 不做重参数化、改用 REINFORCE 估计，代价是什么？
3. KL 权重 $\beta$ 很大 / 很小时，分别会发生什么？
