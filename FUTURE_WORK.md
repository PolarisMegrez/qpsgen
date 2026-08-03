# fpgen 后续工作

本文只记录尚未完成的符号推导与下游接口工作。`fpgen` 的公开示例止于动力学对象、
Jacobian 和符号约化方程；定点搜索、延拓、分岔判定和 GPU 执行属于 `qphase`。

## 当前基线

- 支持 Wigner、P、Q 和 positive-P 表象的完整 Kramers--Moyal series。
- 在 flux 形式下进行二阶截断，并生成 Fokker--Planck 与增广 Itô Langevin 对象。
- 直接从 FPE 扩散矩阵给出正常关联 `D` 和反常关联 `M`。
- Wigner/P/Q 使用 Hermitian 增广矩 `R_tilde=[[R,C],[C*,R*]]`，分别生成
  `dot(R)`、`dot(C)`、实向量化状态及 Jacobian；经符号验证后可自动回退到 normal 布局。
- 严格区分 `exact` 与 `factorized_bilinear` 闭合策略，并记录在 provenance 中。
- positive-P 使用 alpha/beta 独立的复对称二阶矩布局，不施加共轭约束。
- 支持仿射平衡方程块的符号消元，并生成真正的约化动力学
  `dot(q)=G(q;p)=P(q;p)/Q(q;p)`、正则条件与重根条件。
- `MomentDynamicsSpec` 提供版本化状态/参数布局、推导来源、稀疏结构、指纹及 NumPy
  参考实现，供 `qphase` 等下游项目验证和接入。

## 后续优先事项

### 1. 扩展矩闭合策略

- 评估是否需要加入包含一阶矩的仿射闭合，以支持无法写成 `A(z)z` 的常数驱动。
- 为高于双线性的相位多项式设计可插拔闭合接口；默认仍应明确拒绝。
- 如引入 Gaussian/Wick 等近似闭合，必须使用独立 API、明确命名，并写入
  provenance，不能隐式启用。

### 2. 稳定 qphase 交接规格

- 用 two-mode VDP、Kerr、cross-Kerr、three-mode Kerr 和线性增广 DPO 逐点核对
  RHS、state/parameter Jacobian、JVP/VJP 与批处理行为。
- 根据实际接入反馈冻结 schema；冻结前不引入跨语言表达式 DAG。
- 明确 qphase 返回结果的覆盖等级，避免把有限预算内未找到解解释为不存在解。

### 3. 提高符号消元的可扩展性

- 为三模及更大系统增加结构剪枝，减少变量/方程分块的组合枚举成本。
- 提供不展开大行列式和大有理式的凝聚表示与符号秩条件。
- 区分平衡点重根、零特征值缺陷和一般 Jacobian exceptional point；当前重根 API
  只覆盖正则标量约化分支。

### 4. 评估 Langevin 模型规格

待 qphase 的 SDE adapter 契约明确后，再决定是否增加 `LangevinModelSpec`。该规格至少
需要记录增广状态、漂移、真实独立噪声因子、dtype，以及 FPE 复对称扩散、正常关联、
反常关联和具体噪声实现之间的区别。

`fpgen` 不负责 qphase 的积分器、随机数生成、CUDA 算子融合、缓存或运行时调度。

## 提交前验证

```powershell
python -m pytest tests -q
python -m compileall -q fpgen tests
```

涉及 notebook 或公开 API 时，还应从空命名空间顺序执行 `example.ipynb` 与
`models.ipynb` 的全部代码单元。
