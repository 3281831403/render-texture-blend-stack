# render-texture-blend-stack

材质换成多层混合栈后，层序乱了、法线还越混越平：`normalize_layers` 直接按插入顺序返回层列表，`order` 字段被彻底忽略；`blend_over` 在预乘时把 alpha 乘了两遍，半透明层边缘发暗；`blend_normal_map` 把两个法线直接相加后不归一，光照整体过亮；法线混合只保留 base 法线，细节层的起伏全部丢失；`apply_mask` 无视 `invert` 参数，反相遮罩与正向遮罩结果一模一样；`resolve` 在合成时把遮罩与 opacity 各乘两次，整层透明过度；`merge_layers` 合并相邻层时对遮罩取并集，上层覆盖范围被放大；`validate_stack` 不校验 opacity 越界，颜色分量溢出到 0..1 之外；`stack_batch` 让各材质共用同一份层列表，前一个材质的状态串到下一个；`layers` 为空时不报错，返回一堆未定义值。matstack/stack.py 是核心，tests/test_stack.py 15个红。要修的是：按 `order` 稳定排序（缺失 order 追加末尾）、over 只乘一次 alpha、法线混合重新定向后归一、遮罩支持 invert、mask 与 opacity 各生效一次、合并层遮罩相乘、opacity 越界与空栈必须报 `StackError`、批量材质互不串味且失败整体回滚。

## 运行方式

```console
python -m pytest tests -q
```

初始状态必须恰好是：

```console
15 failed in 0.09s
```

15 条测试全部调用 `matstack/stack.py` 里的桩，失败来源是 `NotImplementedError`，不允许出现 collection / import / syntax error。修完之后同一命令应当变成 `15 passed`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 数据约定

层是字典，固定五个键 `{"id", "color", "mask", "opacity", "normal"}`，可选 `"order"`：

| 键 | 类型 | 含义 |
| --- | --- | --- |
| `id` | `str` | 同一栈内唯一，重复即在 `validate_stack` 中报错 |
| `color` | `(r, g, b)` | 0..1 的颜色 |
| `mask` | `float` | 0..1 的层内覆盖率 |
| `opacity` | `float` | 0..1 的材质不透明度 |
| `normal` | `(x, y, z)` | -1..1 的法线，长度非零，混合结果为长度 1 |
| `order` | `int` / `float` | 可选；升序合成，缺失则排在所有有序层之后 |

层序是画家顺序：`order` 最小的是底层，最后合成的是顶层；有效 alpha 为 `opacity * mask`，只生效一次。浮点比较容差 `1e-6`。

## 约束

- `tests/test_stack.py` 不可修改：只能改 `matstack/stack.py` 让 15 条测试变绿。
- 函数签名与异常类 `StackError` 不可改；不要新增第三方依赖。
- 每个函数的契约写在它的 docstring 里，按 docstring 实现即可。
