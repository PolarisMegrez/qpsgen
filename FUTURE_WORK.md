# fpgen 后续工作

本文是项目唯一的规划文档。当前可执行示例见 `example_local.ipynb`，公共接口以
`fpgen.__all__`、对象 docstring 和测试为准。

## 当前基线

- 结构化主方程可推导 Wigner、P、Q 和 positive-P 的完整 Kramers--Moyal series。
- 二阶 flux 截断可生成 Fokker--Planck 和增广 Itô Langevin 对象。
- Wigner/P/Q 在严格双线性闭合条件下可生成 Hermitian 正常协方差动力学。
- 实状态按 mode 声明索引排序，并具有版本化布局、显式参数顺序和推导来源记录。
- `MomentDynamicsSpec` 可生成 metadata-only JSON manifest 和 float64 NumPy 参考模型。
- 数值模型提供 state/parameter Jacobian、JVP/VJP 和按需混合方向导数。
- 线性消元可联合枚举变量/方程分块，记录稀疏性和代价，并在正则分支上生成
  fraction-free 标量残差、分母、清除因子和重根条件。

## 优先事项

### 1. 与 qphase 联调当前规格

先用 2-mode VDP、Kerr、cross-Kerr 和 3-mode Kerr 验证：

- batch `rhs(x, p)`、state/parameter Jacobian 与 qphase 数值模型逐点一致；
- manifest fingerprint 能用于缓存和结果追溯；
- 消元候选排序在实际模型上选择低代价、条件良好的分块；
- qphase 结果使用 `coverage=heuristic|continuation|algebraic|interval_certified`，
  不把有限预算内未找到候选解释为不存在。

根据联调反馈再冻结 schema。冻结前不增加跨语言表达式 DAG。

### 2. 扩展二阶矩状态

当前 Hermitian 正常协方差在反常 Hamiltonian 非零时会明确拒绝闭合。后续需要：

- 为简并参量振荡器加入正常与反常二阶矩的增广状态；
- 为 positive-P 加入 alpha/beta 独立矩，并在数值边界拆分实部和虚部；
- 让所有矩动力学共享 `state_spec`、`parameter_spec`、`rhs`、方向导数和
  `MomentDynamicsSpec`，但保留不同的布局版本与闭合来源。

不得隐式采用 Gaussian/Wick 分解；任何近似闭合都必须显式命名并写入 provenance。

### 3. 增加 Langevin 数值规格

只有在 qphase 的 SDE adapter 契约确定后再实现 `LangevinModelSpec`。它至少需要：

- 增广状态、漂移、真实独立噪声 factor 和 dtype；
- 明确区分 FPE 的复对称扩散、正常关联、反常关联及具体噪声实现；
- 标记 `drift = A(z,p)z` 分解的有效性，常数驱动或非唯一分解时拒绝 Cayley 生成；
- 先生成 NumPy Euler--Maruyama 参考能力，再由 qphase 负责 CuPy/CUDA、RNG、
  积分器融合、缓存和运行时调度。

fpgen 不直接生成依赖 qphase 运行时的 fused CUDA kernel，也不引入 JAX 作为首选
运行时。

### 4. 提高消元与高阶判据的可扩展性

- 在 3-mode 及更大模型上评估变量/方程全分块枚举的组合成本，必要时增加结构剪枝；
- 增加不展开 determinant 的符号秩条件和可供后端装配的凝聚求解规格；
- 对无法标量消元的模型，保留 exact RHS 和混合方向收缩，由 qphase 实现 bordered
  Lyapunov--Schmidt 系统、伪弧长延拓和任意精度复核；
- 明确区分 `equilibrium_multiplicity`、`zero_eigenvalue_defect` 和一般 Jacobian
  exceptional point。第一目标仍是单零空间下的平衡点高重根。

## 提交前验证

```powershell
python -m pytest tests -q
python -m compileall -q fpgen tests
```

涉及 notebook 展示或公共 API 时，还应完整执行 `example_local.ipynb`，并验证生成的
NumPy 模型在单样本和前导 batch 维下与 SymPy 表达式一致。
