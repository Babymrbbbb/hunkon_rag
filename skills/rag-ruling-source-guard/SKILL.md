---
name: rag-ruling/source-guard
description: 聚合 / 生成入口的有效源为空时必须显式失败，绝不允许拿 0 个源产出空壳页（模式 B）。
mode: B
entrypoints: [require_nonempty_sources]
---

# rag-ruling/source-guard（模式 B）

## 触发条件
聚合页 / LLM 生成入口收到「源列表」参数，且你必须在生成前确认源非空。

## 为什么需要它
内部实战教训：拿 0 个源去问 LLM 会产出**空壳页**，而且这种页面能过字段校验
（字段都在，只是内容为空），从而覆盖掉完整的旧版页面——比直接报错危险得多。
「空输入比报错更危险」：报错至少阻断，空输入会静默污染。

## 怎么用
在生成 / 聚合函数的最前面调用：

```python
from app.rules import require_nonempty_sources, RulingError

def build_aggregate(sources):
    require_nonempty_sources(sources)  # 空源直接抛 RulingError，阻断后续
    ...
```

## 反模式
- ❌ 用 `if not sources: return ""` 静默返回（空壳页会被下游当正常结果）。
- ❌ 只在单元测试里假设源非空，不在入口做前置校验。
- ❌ 把「相邻维度」当「目标指标」冒充（缺失就显式说明不可得，不编造）。
