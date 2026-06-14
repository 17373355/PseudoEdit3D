# AML 作为多模态动作接口的长期设想

## 背景

AML 当前的核心对象是 human motion。它从连续运动中抽取可解释、可组合、可编辑的中间表示：

```text
continuous human motion
-> motion observables
-> micro-events
-> sub-motion units
-> semantic patterns
-> editable AML program
```

但长期来看，AML 不应该只停留在 motion annotation language。更自然的方向是让 AML 成为一种 grounded action interface，用来连接肢体动作、视觉信号、触觉/接触信号，以及动作对世界状态造成的影响。

未来 AML 可以扩展为：

```text
vision / touch / proprioception / motion
-> grounded action evidence
-> AML action program
-> prediction / editing / execution / correction
```

也就是说，AML 不只是描述身体怎么动，还应该描述身体动作如何与物体、场景、接触状态和世界后果发生关系。

## 核心主张

AML 可以发展成一种 body-centric multimodal action interface。

它绑定的不只是 motion facts，还包括：

- 身体部位和肢体运动；
- 物体和场景区域；
- 接触、支撑和触觉状态；
- 动作发生的前提条件；
- 动作造成的世界状态变化；
- 可编辑的时间区间；
- 可执行的控制参数；
- 多模态证据来源和置信度。

一个未来的 AML event 可以表示为：

```text
event = <
  body_part,
  temporal_span,
  motion_direction,
  magnitude,
  count,
  contact_state,
  object_reference,
  scene_region,
  tactile_state,
  precondition,
  effect,
  confidence,
  evidence_sources
>
```

## 与 Visual Language Model 的关系

Visual Language Model 可以为 AML 提供物体、场景和空间关系的 grounding。

例如，motion-only AML 可能只能检测到：

```text
SIT_DOWN_CANDIDATE:
  pelvis lowers
  knees flex
  terminal low posture
```

这说明身体正在下降到类似坐姿的状态，但不能证明人真的坐在椅子上。因为仅凭 motion，没有 chair 的视觉证据，也没有 pelvis-chair contact 的场景证据。

如果加入视觉 grounding，AML 可以进一步得到：

```text
SIT_DOWN_ON_CHAIR:
  pelvis lowers
  knees flex
  chair is behind body
  pelvis contacts chair surface
  terminal seated support is achieved
```

这时，语义状态可以从 motion-only candidate 变成 grounded action pattern。

类似地，motion-only AML 可能检测到：

```text
JUMP_ROPE_LIKE_CANDIDATE:
  repeated vertical jumps
  bimanual rhythmic hand motion
  in-place root trajectory
```

但这仍然不能证明人在跳绳，因为 rope object 没有被观测到。Visual Language Model 可以补充：

```text
rope is visible
rope rotates around the body
hand rhythm is synchronized with rope motion
```

只有当这些视觉和交互证据存在时，AML 才应该把该 pattern 提升为稳定的 `JUMP_ROPE`，否则更合理的表达是 `jump_rope_like` 或 `jump_rope_mime_candidate`。

## 与 Visual Action Model 的关系

Visual Action Model 需要能够理解和生成可执行动作。AML 可以作为其中的显式 action program。

例如，一个面向物体交互的 AML program 可以写成：

```text
reach(right_hand, target=cup_handle)
grasp(right_hand, force=light)
lift(cup, height=10cm)
keep(cup_tilt < threshold)
```

这和直接学习一个 latent policy 不同。AML 提供了一个中间结构，使模型或用户可以检查：

- 当前意图是什么；
- 哪个身体部位参与动作；
- 目标物体是什么；
- 哪些时间段需要修改；
- 哪些约束需要保持；
- 哪些动作参数可以被控制。

例如，以下自然语言反馈可以被映射到 AML slots：

```text
raise the cup higher
grip the handle more firmly
do not tilt the cup
stop the left hand from touching the table
```

这些反馈不应该只作为普通 caption 输入模型，而应被解析成：

```text
target_event
target_span
target_body_part
target_object
slot_to_modify
constraint_to_preserve
```

这样 AML 就能成为 Visual Action Model 中可解释、可编辑、可执行的动作接口。

## 与 World Action Model 的关系

World Action Model 关注动作如何改变环境状态，以及动作是否物理可行、安全、稳定。

AML 可以在 motion 和 world effect 之间提供结构化桥梁。

例如：

```text
precondition: chair behind pelvis
action: lower pelvis
effect: seated support achieved
failure: no support contact
```

或者：

```text
precondition: cup on table
action: push cup forward
effect: cup translates
failure: excessive force may cause cup to fall
```

这说明 AML 不只是运动描述语言，也可以成为 action consequence reasoning 的接口。它可以帮助模型判断：

- 动作的前提是否满足；
- 目标物体是否存在；
- 接触是否成功；
- 支撑是否稳定；
- 物体状态是否按预期变化；
- 动作是否违反物理或安全约束。

## 证据来源需要分离

为了避免把不可靠的语义强行写进 motion layer，AML 应该明确区分不同证据来源。

```text
Motion evidence:
  joints, velocity, direction, magnitude, count, temporal span

Perceptual evidence:
  objects, scene regions, affordances, spatial relations

Interaction evidence:
  contact, force/tactile proxy, support state, object state change

World evidence:
  preconditions, effects, failure modes, physical feasibility
```

semantic pattern 不应该替代这些底层证据，而应该是对这些证据的绑定。

例如：

```text
jump_rope_like:
  motion evidence: repeated jumps + hand cycles
  visual evidence: missing rope
  interaction evidence: no rope contact
  status: candidate
```

而不是直接写成：

```text
jump_rope:
  status: stable
```

当视觉和交互证据足够时，才可以提升为：

```text
jump_rope:
  motion evidence: repeated jumps + hand cycles
  visual evidence: rope visible
  interaction evidence: rope rotation synchronized with body motion
  status: stable
```

## 对当前 AML 分层的要求

当前 motion-only AML 应继续保持 conservative。

Layer 0-3 应该保存 grounded motion facts：

```text
part
span
direction
magnitude
count
state/contact proxy
supporting events
```

更高层的 semantic pattern 应作为 hypothesis 附着在底层 motion evidence 之上：

```text
semantic_pattern
-> supporting Layer3 events
-> supporting Layer2 sub-motion units
-> supporting Layer1 micro-events
-> visual / tactile / world evidence
-> editable slots
```

这种设计可以保证：

- motion layer 不被人类 action name 污染；
- semantic pattern 可以解释、可以审计；
- pattern 的置信度可以随证据来源变化；
- motion-only candidate 将来可以被视觉、触觉和 world evidence 提升为 grounded action；
- AML 可以自然进入 embodied action、robot control 和 world model 场景。

## 设计原则

未来 AML 中的人类动作语义不应该被视为单一标签，而应该被视为 evidence binding。

例如：

```text
SIT_DOWN_CANDIDATE
= body lowering + knee flexion + terminal low posture
```

而 grounded version 是：

```text
SIT_DOWN_ON_CHAIR
= body lowering
+ knee flexion
+ chair behind pelvis
+ pelvis-chair contact
+ stable support state
```

再例如：

```text
JUMP_ROPE_LIKE_CANDIDATE
= repeated vertical jumps
+ bimanual periodic hand motion
+ in-place root trajectory
+ missing rope evidence
```

而 grounded version 是：

```text
JUMP_ROPE
= repeated vertical jumps
+ bimanual periodic hand motion
+ visible rope
+ synchronized rope rotation
+ object-motion interaction evidence
```

因此，AML 的长期设计应坚持：

```text
motion facts stay low-level and grounded
semantic patterns stay evidence-linked and uncertainty-aware
multimodal grounding upgrades candidate semantics into stable action programs
```

## 总结

AML 可以从 motion annotation language 进一步发展为 embodied intelligence 中的 grounded action interface。

一个简洁定义是：

> AML is a grounded action interface that binds body motion, scene objects, contact states, and expected world effects into editable and executable programs.

中文表述可以写为：

> AML 是一种具身动作接口，它把身体运动、场景物体、接触状态和预期世界后果绑定成可解释、可编辑、可执行的程序。

这个方向使 AML 不只服务于 human motion annotation，也可以进一步连接 Visual Language Model、Visual Action Model、robot control 和 World Action Model。
