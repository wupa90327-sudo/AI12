# Warmup 技术介绍

## 什么是 Warmup

Warmup 通常指在训练开始阶段逐步增大学习率，而不是一开始就使用目标学习率。它常用于深度学习模型训练，尤其是 Transformer、大模型预训练、迁移学习和大 batch 训练场景。

简单来说，warmup 的目标是让模型在训练初期先稳定适应数据和参数更新，再进入正常训练节奏。

## 为什么需要 Warmup

训练刚开始时，模型参数通常还没有形成稳定表示。如果直接使用较大的学习率，可能导致：

- loss 剧烈震荡；
- 梯度更新过大；
- 训练不稳定甚至发散；
- 前几轮训练破坏已有预训练权重。

Warmup 通过在前若干 step 或 epoch 内从较小学习率逐步增加到目标学习率，降低训练初期的不稳定风险。

## 常见 Warmup 策略

### 线性 Warmup

线性 warmup 是最常见的方式。学习率从 0 或一个较小值开始，按训练步数线性增加到目标学习率。

示例：

```text
lr = target_lr * current_step / warmup_steps
```

当 `current_step >= warmup_steps` 后，学习率进入正常调度阶段。

### Constant Warmup

在 warmup 阶段使用一个固定的小学习率，经过指定步数后切换到目标学习率。这种方式实现简单，但切换点可能带来学习率突变。

### Cosine Warmup

Warmup 后接 cosine decay 是常见组合：

1. 前期学习率逐步升高；
2. 达到目标学习率后按余弦曲线逐步衰减。

这种方式适合较长周期训练，也常用于大模型训练。

## 关键参数

- `target_lr`：warmup 后要达到的目标学习率。
- `warmup_steps`：warmup 持续的训练步数。
- `warmup_ratio`：用总训练步数的比例表示 warmup 长度，例如 `0.03` 表示前 3% step 用于 warmup。
- `scheduler`：warmup 后使用的学习率调度策略，如 linear decay、cosine decay 等。

## 使用建议

- 小模型或简单任务不一定需要 warmup。
- 大 batch、Transformer 和预训练模型微调通常建议开启 warmup。
- `warmup_ratio` 可先从 `0.03` 到 `0.1` 尝试。
- 如果训练初期 loss 波动很大，可以适当增加 warmup 步数。
- 如果模型收敛太慢，可以减少 warmup 步数或提高初始学习率。

## 简单示例

下面是一个线性 warmup 的伪代码：

```python
def get_learning_rate(step, warmup_steps, target_lr):
    if step < warmup_steps:
        return target_lr * step / warmup_steps
    return target_lr
```

实际项目中通常会把 warmup 与学习率 scheduler 结合使用，例如在 warmup 结束后继续执行 cosine decay 或 linear decay。

## 总结

Warmup 是一种简单但有效的训练稳定化技术。它通过控制训练初期的学习率变化，减少不稳定更新带来的风险，帮助模型更平稳地进入正式训练阶段。
