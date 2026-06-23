# TMR 对 AML 的启示与可耦合方向

## 背景

本文档记录 HumanTOMATO 中 TMR（Text-Motion Retrieval model）对 AML 的启发，供后续工作 session 判断如何逐步耦合到当前 Motion-BPE / pattern family / naming 审核流程中。

HumanTOMATO 的 TMR 可以理解为 motion-aware 的 text-motion 对齐器。它不是图文 CLIP，而是专门训练 text encoder 和 motion encoder，把文本描述与人体动作序列映射到同一个 embedding space。

HumanTOMATO 使用 TMR 的核心动机是：CLIP 或普通 LLM text embedding 更擅长静态视觉/语言语义，不足以稳定表达 motion 的方向、顺序、轨迹和动态差异。例如 clockwise / counter-clockwise、forward / backward、left / right、walk then turn / turn then walk 等。

## HumanTOMATO 中 TMR 的三个角色

### 1. Motion-aware text encoder

HumanTOMATO 用 TMR text encoder 替代普通 CLIP text encoder，作为 Hierarchical-GPT 的文本输入。这样输入 embedding 更贴近 motion 分布，而不是图像语义分布。

### 2. Training-time alignment critic

训练生成模型时，HumanTOMATO 不只预测离散 motion token，还把生成 motion 与输入 text 分别送入 TMR motion encoder 和 TMR text encoder，通过 text-motion alignment loss 约束生成结果与文本整体语义一致。

抽象为：

```text
text -> TMR text encoder
generated motion -> TMR motion encoder
alignment loss(text, generated_motion)
```

### 3. Evaluation metric

HumanTOMATO 提出 TMR-R-Precision 和 TMR-Matching-score，用 TMR 替代旧 HumanML3D retriever 来评价 text-motion alignment。更大的 retrieval set，例如 256，比 32 更难，也更能暴露语义对齐差异。

## 对 AML 的核心启示

TMR 不应替代 AML。它更适合作为 AML 与 motion 之间的 learned semantic judge。

AML 的核心职责仍然是：

```text
motion -> interpretable events -> pattern family -> AML program
```

TMR 类模块的职责可以是：

```text
motion / AML / text 之间的语义一致性评分
```

因此推荐关系是：

```text
AML = 可解释、可编辑、可组合的 grounded interface
TMR = 学习式 motion-text / motion-AML 对齐器与审核器
```

## 可耦合方向 A：TMR 审核 pattern naming

当前 Motion-BPE / coordination motif 流程已经能从 motion-only evidence 中提出结构节点，例如：

```text
COORD_SIG[...] -> <COM_0036> -> jumping_jack candidate
```

目前 naming 主要依赖 caption alias purity、pseudo-GT audit、WordNet / text-BPE naming evidence。TMR 可以作为额外审核信号，而不是决定结构的来源。

可加入的审核问题：

```text
motion 与 proposed name 是否相似？
motion 与 AML rendered prompt 是否相似？
motion 与 competing labels 的相对排名如何？
```

示例：

```text
score(motion, "jumping jack")
score(motion, "jump rope")
score(motion, "jumping in place while raising both arms")
score(motion, AML_rendered_description)
```

如果 proposed name 在 TMR ranking 中明显高于 competing labels，可以提高 naming confidence。如果 TMR 更支持另一个文本表达，则该 motif 应进入 review，而不是直接命名。

建议新增字段可以是：

```json
{
  "tmr_naming_audit": {
    "proposed_name": "jumping_jack",
    "proposed_score": 0.0,
    "competing_names": [
      {"name": "jump_rope", "score": 0.0},
      {"name": "cheer_dance", "score": 0.0}
    ],
    "rank": 1,
    "policy": "diagnostic_only"
  }
}
```

注意：TMR score 不能创建 motion node。motion node 仍必须来自 motion evidence。

## 可耦合方向 B：辅助处理 approximate / unknown pattern

一些 HumanML3D action name 并不完全由 skeleton motion 决定，例如：

```text
jump rope
karate
tennis
mimic chicken
play instrument
dribble basketball
```

这些 label 可能包含 object、scene、intent、style 或 cultural convention。当前 AML 应避免把这些词直接变成 motion-only family。

TMR 可以帮助区分三类情况：

### 1. Motion-supported semantic family

motion evidence 与 name 都强，例如 jumping_jack 中的 vertical up/down + bimanual raise/spread。

```text
motion evidence strong
caption/name evidence strong
TMR supports proposed name
```

可以作为 named pattern family candidate。

### 2. Approximate motion family

motion evidence 支持某种近似动作结构，但缺少对象或场景证据。例如 jump rope skeleton 可能只显示 repeated jumping + arm rotation，但没有 rope。

```text
motion evidence supports approximate structure
TMR supports object/action label
visual/object evidence missing
```

应标为：

```text
approximate_semantic_family
needs_visual_or_object_grounding
```

### 3. Context-only / weakly grounded label

TMR 或 caption 支持某个词，但 skeleton motion evidence 不稳定，或者多个不同动作结构都被叫同一个词。

应标为：

```text
context_or_intent_label
not_motion_tree_authority
```

这对未来 VLM / VAM / world action model 很重要：AML 可以承认当前 skeleton-only 证据不足，把 object / visual / tactile / scene grounding 留给多模态系统补全。

## 可耦合方向 C：未来 AML-aware TMR

普通 TMR 只学习 natural text 与 motion 的对齐。AML 需要更细粒度的对齐能力，例如 part、direction、count、span、angle、contact/state。

未来可以训练 AML-aware TMR：

```text
motion encoder:
  raw motion / joints / latent motion tokens

AML encoder:
  structured AML fields
  rendered AML text
  pattern family id
  numeric residues

training pairs:
  motion <-> AML program
  motion <-> AML rendered prompt
  motion <-> HumanML3D caption
```

目标不是让 AML 变成普通 caption，而是让模型学会：

```text
part-level correctness
direction correctness
count correctness
temporal ordering
span-level grounding
approximate vs exact semantic family
```

可能的训练/评价任务：

```text
motion -> retrieve correct AML program
AML program -> retrieve correct motion
motion -> rank correct name above competing names
motion + candidate AML -> binary consistency score
source motion + edit instruction + target motion -> edit consistency score
```

## 与当前 Motion-BPE 主线的推荐关系

当前主线应保持：

```text
motion cluster + Motion-BPE -> motion pattern tree
text-BPE + WordNet -> naming layer
TMR -> learned naming / alignment audit
```

也就是说：

```text
motion decides structure
language decides names
TMR audits semantic alignment
WordNet supplies lexical hierarchy
```

TMR 不应参与以下步骤：

```text
create base motion events
select Motion-BPE merges
create motion tree nodes by itself
```

TMR 可以参与以下步骤：

```text
rank candidate names
audit AML rendered prompt faithfulness
identify approximate semantic labels
score generated target motion vs AML edit program
build future AML-aware retrieval benchmark
```

## 给工作 session 的最小落地建议

第一阶段只做 audit，不训练新模型。

可以先接入已有或可复用的 TMR / text-motion retriever，对当前 pattern family proposal 做离线打分：

```text
input:
  pattern_family_proposal.json
  support case motion ids
  proposed name and competing aliases
  AML rendered descriptions if available

output:
  tmr_naming_audit.json
  tmr_naming_audit.md
```

优先测试对象：

```text
jumping_jack
jump_rope
kick
lunge
sit_down
karate / martial_arts
```

审核重点：

```text
TMR 是否能把 proposed name 排在 competing labels 前面
TMR 是否能区分 exact family 与 approximate family
TMR 是否能发现 caption alias purity 高但 motion 结构不一致的情况
TMR 是否能发现 motion evidence clean 但 caption alias noisy 的情况
```

第二阶段再考虑 AML-aware TMR。训练前必须先稳定 AML program schema 和 rendered prompt，否则模型会学习到不断变化的目标。

## 一句话总结

TMR 给 AML 的价值不是替代 symbolic AML，而是补上一层 learned semantic alignment：用它审核命名、识别 approximate pattern，并为未来 motion / AML / text 三方对齐模型打基础。
