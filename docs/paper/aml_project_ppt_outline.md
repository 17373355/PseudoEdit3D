# AML 项目汇报 PPT（Markdown 版）

> 说明：本文件按幻灯片结构撰写，可直接转成 PPT 内容。
> 文字解释以中文为主，学术名词与方法词保留 English。

---

## Slide 1. 标题页

**Learning an Atomic Motion Language for Structured Motion Annotation, Control, and Transfer**

中文副标题：
- 学习一种用于动作标注、控制与迁移的原子动作语言

一句话定位：
- 我们希望构建一个介于自然语言与 motion tokens 之间的语义层，使机器人既能理解人类语言，也能把自己的 action intent 转成动作。

---

## Slide 2. 背景问题

标题：为什么现有 motion-language 学习还不够好？

要点：
- 现有很多方法默认把整段 motion 输入 encoder，学习 difficulty 很高。
- 对比文本，LLM 成功的重要原因之一是有成熟的 tokenizer / subword system。
- 在 motion 场景中，raw HumanML3D captions 存在明显噪声：
  - 同一段 motion 多条 caption 互相不一致
  - 粒度不统一
  - 有时描述并不准确
- 如果直接把 noisy captions 当监督，模型容易学到 benchmark-specific language，而不是真正的 motion semantics。

核心问题：
- 什么样的 pose / motion token 才是好的？
- 如何在 tokenization 之前，先定义清楚 motion 的 atomic semantic units？

---

## Slide 3. 我们的核心想法

标题：从 noisy captions 到 Atomic Motion Language

要点：
- 我们不把 HumanML3D captions 当 strict ground truth。
- 我们把它们当作 noisy prior。
- 目标不是直接优化一句 prompt，而是构建一个更原子化、更一致、更 motion-faithful 的语义层。

统一表述：
- 短期：固定 MoMask tokenizer，重点学习上层的 Atomic Motion Language (AML)
- 长期：AML 反过来指导更好的 motion tokenizer 设计

一句话：
- `motion tokenizer` 负责底层压缩
- `Atomic Motion Language` 负责语义组织与控制接口

---

## Slide 4. 项目总框架

标题：当前项目主线

```text
HumanML3D motion + all captions
    -> caption prior inventory
    -> motion kinematic analysis
    -> auto_program
    -> auto_prompt
    -> MoMask generation probe
    -> qualitative / bad-case analysis
    -> pattern update
```

解释：
- `auto_program` 是核心结构化输出
- `auto_prompt` 只是文本渲染层 / probe layer
- `MoMask` 当前只是 semantic probe，不是最终 structured model

---

## Slide 5. 为什么需要 AML，而不是只做 motion tokenization

标题：AML 与 motion tokenization 的关系

要点：
- 单纯的 motion tokenization 只解决“如何压缩和生成动作”。
- 但机器人真正需要的是：
  - 可解释的 semantics
  - 可组合的 control units
  - 可对齐 natural language 与 action intent 的桥梁
- 因此我们需要三层结构：

```text
Natural Language / Robot Intent
    <-> Atomic Motion Language (AML)
    <-> Motion Tokens
    <-> Motion / Robot Action
```

结论：
- AML 不是替代 tokenizer
- AML 是 tokenizer 之上的 semantic interface

---

## Slide 6. 从 LLM 的视角理解 motion

标题：为什么要把 motion 看成“字母 -> 半词 -> 语言”

文本领域：
```text
characters -> subwords -> words -> sentences
```

motion 领域：
```text
frame observables -> micro-events -> sub-motion units -> atomic programs
```

关键观点：
- 字母不是“姿态”，而是“微变化”
- 真正的 motion alphabet 应该是短时间窗内、方向明确、量级可量化的变化事件

例子：
- `ROOT_UP`
- `TURN_LEFT`
- `L_ELBOW_UP`
- `TORSO_BEND_FWD`
- `STOP_LOW_VEL`

---

## Slide 7. 当前技术路线：Layered Design

标题：从 Frame Observables 到 Atomic Program

### Layer 0: Frame Observables
- root height
- root heading
- root xz velocity
- pelvis-to-ankle compression
- torso bend
- left/right arm raise
- left/right elbow lift
- wrist-to-chest distance
- contact / support proxies

### Layer 1: Micro-Events
- 按方向变化切分成最小变化事件
- 例：`ROOT_DOWN_M`, `LEFT_ELBOW_UP_S`

### Layer 2: Sub-Motion Units
- 由高频局部组合 merge 得到
- 例：`hop_ascent`, `crouch_descent`, `arm_lift_front`

### Layer 2.5: Phase Patterns
- 识别 repeated / alternating phases
- 例：repeated hop, repeated squat, repeated arm up/down

### Layer 3: Atomic Programs
- `type / part / direction / magnitude / unit / count / start / end / confidence`

---

## Slide 8. 当前已经完成了什么

标题：当前进展

### 已经完成
- 固定 600-case regression benchmark
  - Batch 1: 100
  - Batch 2: 500
- motion-only `auto_prompt`
  - 不再允许 same-case caption retrieval
- `good / soft_bad / hard_bad` triage
- Layer 0 frame observables
- Layer 1 micro-events
- Layer 2 初版 sub-motion lexicon
- Layer 2.5 初版 phase detector
- Batch 1 和 Batch 2 的 representative visualization

### 已经修掉的一些典型错误
- stairs vs bounce
- crouch vs stairs
- repeated squat vs repeated bounce
- some stop semantics
- some limb-level up/down patterns

---

## Slide 9. 当前发现的主问题

标题：当前 bottleneck

### 1. false positive
- false crouch / false squat
- bend 被误识别成 bounce
- 轻微 posture change 被误打成 crouch

### 2. false negative
- walk_backward
- stop_pause edge cases
- hop / bounce / jump count

### 3. coarse semantics
- too much whole-body description
- missing limb details:
  - arm circling
  - elbow flap
  - hand support / rail support

### 4. missing number / angle awareness
- `3 steps`
- `twice`
- `180 degrees`
- `4 stairs`

---

## Slide 10. Semantic Merge, Numeric Retention

标题：为什么 merge 后仍要保留数值信息

原则：
- 语义上可以 merge
- 数值、方向、次数、时间跨度必须可检索

所以每个 `sub-motion` / `atomic event` 仍要保留：
- support tokens
- start / end
- signed deltas
- magnitude summary
- count
- confidence

目标例子：
- “右手再向外打开 30 度”
- “向前走 2 步，再向右跳 3 步，中间不要转身”

---

## Slide 11. 当前实验设计

标题：实验设计概览

### Experiment 1: HumanML3D Re-annotation Benchmark
- original HumanML3D captions + MoMask text-conditioned model trained on original captions
- AutoPrompt-HumanML3D + same MoMask architecture retrained on AutoPrompt supervision
- compare FID and consistency

### Experiment 2: Cross-Dataset Generalization
- motion_B -> AutoPrompt
- AutoPrompt-conditioned model trained on AutoPrompt-HumanML3D
- generation on dataset B
- compare with GT_B

### Current annotation-layer regression
- 600-case fixed benchmark
- `good / soft_bad / hard_bad`
- representative qualitative cases

---

## Slide 12. 当前从规则修补走向表示学习

标题：我们现在不是在修 prompt，而是在建表示层

现在主线已经变成：

```text
Layer 0 observables
-> Layer 1 micro-events
-> Layer 2 sub-motion candidates
-> Layer 2.5 phase patterns
-> Layer 3 atomic programs
```

这意味着：
- not just prompt engineering
- but a structured motion semantic representation problem

---

## Slide 13. 4 个月规划

标题：4个月路线图

### By 2026-06-10
- 把 heuristic pipeline 提升成 paper-writable mechanism
- 形成：
  - iterative refinement
  - confidence-based denoising
  - consistency-driven relabeling
  - structured latent induction

### 2026-06-10 ~ 2026-09-10
- 稳定 600-case benchmark
- 跑 benchmark / transfer experiments
- 补 structured-condition prototype
- 准备 figures / tables / artifact summary

### From 2026-09-10
- 开始正式写 paper

---

## Slide 14. 现在的关键任务

标题：What is next?

1. strengthen Layer 1 and Layer 2
2. build a more stable sub-motion vocabulary
3. improve repeated phase detection
4. add finer limb patterns
5. introduce stronger number / angle / count fields
6. later connect AML to a structured-condition model

---

## Slide 15. 总结

标题：一句话总结

我们当前不是在直接训练最终 generator，
而是在构建一层位于 noisy captions 与 motion tokens 之间的 **Atomic Motion Language**：

- 更原子化
- 更一致
- 更 motion-faithful
- 更适合未来的 structured control

它将成为：
- human language
- robot intent
- motion tokens
之间的语义桥梁。
