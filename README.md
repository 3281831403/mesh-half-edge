# mesh-half-edge

换成半边结构后，模型边界处遍历死循环、翻边还把网格撕开：`build` 的边键用有序对 `(min,max)`，反向半边丢失、twin 只有一半；`build` 使用 `(a,b)` 但同一无向边出现三次（非流形）时静默覆盖，非流形未报；`twin_map` 的边界半边填成自身而不是 `-1`，一环遍历死循环；`vertex_ring` 只走 twin 不走 next，漏面内边；`vertex_ring` 不限制步数，开放边界上死循环；`face_ring` 用共享顶点判断邻接，非邻接面被误连；`flip_edge` 翻转后不更新外围 twin，破坏一致性；`euler_characteristic` 的 E 用半边数而不是无向边数，结果差一倍；`validate` 不检查面绕序（法线朝向），内向网格未被报；`mesh_batch` 复用边表，第二网格边键冲突。halfedge/mesh.py 是核心，tests/test_mesh.py 15个红。要修的是：…

难点是拓扑不变量：twin 对称、边界 −1、欧拉示性数在翻边后不变。

## 运行方式

在仓库根目录执行：

```bash
python -m pytest tests -q
```

初始失败环境必须恰好是 `15 failed`，且没有 collection error、import error 或 syntax error；输出末尾形如：

```text
15 failed in 0.05s
```

测试文件本身已经写好，不要求新增测试；修完 `halfedge/mesh.py` 后同一命令应变为 `15 passed`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 目录

```text
mesh-half-edge/
├── README.md
├── conftest.py
├── .gitignore
├── halfedge/
│   ├── __init__.py
│   └── mesh.py
└── tests/
    └── test_mesh.py
```

`halfedge/mesh.py` 只保留 8 个待实现函数：`build`、`twin_map`、`vertex_ring`、`face_ring`、`flip_edge`、`euler_characteristic`、`validate`、`mesh_batch`。核心约定写在模块与函数 docstring 中：面/顶点索引约定、边界半边 `twin == -1`、半边/面的局部顺序，以及浮点比较使用的 `1e-9` 容差。

## 约束

- `tests/test_mesh.py` 不可修改：不得增删测试、跳过测试或放宽断言。
- 只使用 Python 标准库（`math`、`collections` 等）与 `pytest`；离线运行、不使用随机数。
- 不得通过修改测试或伪造 `15 passed` 绕过拓扑检查；验收同时看欧拉示性数、twin 对合、边界完整性和翻边一致性。
