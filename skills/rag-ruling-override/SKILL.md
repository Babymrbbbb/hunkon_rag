---
name: rag-ruling/ruling-override
description: 在 LLM 生成之后用真相源做确定性口径覆盖，并登记覆盖路径确保闸落在每处执行路径（模式 A）。
mode: A
entrypoints: [apply_ruling_overrides, register_ruling_path, verify_ruling_coverage]
---

# rag-ruling/ruling-override（模式 A）

## 触发条件
任何「LLM 生成后要对齐权威口径」的场景，且你希望**保证**覆盖闸不是只改了一处、
而是落在了所有实际输出路径上。

## 为什么需要它
口径正确与否不能押在 LLM 自觉上。参考实现的做法是：裁决文档写了什么就是什么，
用确定性后处理在生成之后强制覆盖。但如果在 N 个输出路径里只接了 1 个，
其余 N-1 个仍会输出旧口径——「只改一处 = 形同虚设」。

## 怎么用
1. 在每处实际产出对外文本的函数里，写入前调用 `apply_ruling_overrides(text)`。
2. 同一处紧接着调用 `register_ruling_path("module.func")`，声明「此处已覆盖」。
3. CI 里调用 `verify_ruling_coverage()`，返回空列表即通过；否则说明有声明必需却未登记的路径。

## 反模式
- ❌ 只在主路径接覆盖闸，特殊页 / 聚合页漏接。
- ❌ 用 prompt 提醒 LLM「请遵守口径」代替确定性后处理。
- ❌ 覆盖逻辑写死业务数字而非由真相源 JSON 驱动。
