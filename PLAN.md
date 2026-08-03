# fpgen 结构化推导重构

## 总结

将项目重构为基于 SymPy 的结构化量子主方程工具，完整流程为：

```python
km = derive_kramers_moyal(master, representation="positive-p")
fpe = km.truncate(2)
langevin = fpe.to_langevin()
```

统一支持 Wigner、P、Q 和 positive-P；以增广复状态为标准输出，保留完整 Kramers–Moyal Series，显式记录 Fokker–Planck 截断，并生成可直接模拟的 Itô Langevin 方程。

## 公共接口与数据模型

- 新增非对易算符多项式模型。`boson_modes("a", "b")` 创建模式，支持乘法、加法、幂、标量 SymPy 系数和 `.dag`；取 dagger 时反转算符顺序并复共轭系数。
- `MasterEquation` 显式接收模式顺序，并支持：
  - `hamiltonian`，自动生成 `-I[H,rho]`；
  - `LindbladChannel(jump=L, rate=gamma)`，约定为 `gamma*(L rho L† - {L†L,rho}/2)`；
  - `extra_terms`，由 `LiouvillianTerm(coefficient, left, right)` 表示任意 `coefficient * left * rho * right`。
- `derive_kramers_moyal(...) -> KramersMoyalSeries`。结果以
  `dP/dt = sum_mu (-1)^|mu| / mu! * partial^mu[K_mu P]`
  为唯一符号约定，保存变量顺序、所有阶数、各多重指标 `mu` 和 SymPy 系数 `K_mu`。
- `KramersMoyalSeries.truncate(2) -> FokkerPlanckEquation`。结果保留被舍弃项、原始最高阶和 `is_exact`，任何截断都在 repr/LaTeX 中明确显示。
- `FokkerPlanckEquation.to_langevin() -> LangevinEquation` 输出：
  - 增广状态 `z`；
  - 漂移 `A_i = K_{e_i}`；
  - 复对称 FPE 扩散 `G_ij = K_{e_i+e_j}`；
  - 规范噪声因子 `B`，满足经 SymPy 化简后 `B B^T = G`；
  - `xi = B eta`，其中 `eta` 为相互独立的实 Gaussian 白噪声；
  - 反常关联 `M=<xi xi^T>=G`；
  - 当前规范噪声实现下的正常关联 `D=<xi xi†>=B B†`。
- 所有结果对象提供结构化字段、可读 `repr`、`to_latex()` 和 notebook `_repr_latex_`，首版不定义 CLI 或 JSON schema。
- 这是破坏性 API 更新：移除公开的 `opr2fp`、`to_fp_normal_form`、字符串解析及指数元组契约，版本升级到 `0.2.0`。

## 推导实现

- 用统一的多变量符号微分算符代数替换 `float` 指数元组；系数使用 `sympy.Expr`，通过多变量 Leibniz 法则组合左、右乘对应关系。
- 明确区分两种微分算符表示：normal 形式为 `f_mu(z) partial^mu P`（系数在导数左侧），flux（通量/散度）形式为 `partial^mu[K_mu(z) P]`（导数作用于系数与分布的乘积）。Kramers–Moyal 阶数以及 Fokker–Planck 截断均以 flux 形式中的 `|mu|` 定义。
- 截断顺序固定为：先保留全部阶数并精确完成 normal → flux 重排，合并由 Leibniz 法则产生的所有低阶项；再在 flux 形式中舍弃 `|mu|>2` 的项。禁止在 normal 形式中预先截断，因为例如 `f(z) partial^3 P` 重排后除三阶通量外还会产生由 `f` 的导数组成的二阶及更低阶通量，提前删除会改变漂移和扩散。
- 所有模式在整条主方程上使用显式全局顺序，修复当前不同加和项缺少部分模式时指数块错位的问题。
- Cahill–Glauber 表象使用增广顺序
  `z=(alpha_1,...,alpha_N,alpha_bar_1,...,alpha_bar_N)`；`alpha_bar` 在符号运算中视为独立变量。
- positive-P 使用
  `z=(alpha_1,...,alpha_N,beta_1,...,beta_N)`，其中 `beta` 明确不是 `alpha` 的复共轭，并实现：
  `a rho -> alpha P`、`a† rho -> (beta-partial_alpha)P`、
  `rho a -> (alpha-partial_beta)P`、`rho a† -> beta P`。
- 表象名称使用严格枚举和规范别名；未知名称直接报错，不再像当前 `Positive P` 那样静默回退到 Wigner。
- 零阶 KM 项若不为零，FPE 仍可展示，但 `to_langevin()` 拒绝转换并报告主方程不满足概率守恒。
- `B` 使用稳健稀疏规范分解：对角元各用一个噪声列，非对角元各用两个带相对相位的列，使多余对角贡献相消。该分解不保证最少噪声数，但对任意符号复对称 `G` 均可构造。
- 条件输出增广有效 Hamiltonian `H_eff(z)`：
  - 先将 `A_i` 中含 `z_i` 的项规范分配到对角元；
  - 剩余仅允许是状态变量的一次齐次项，并唯一分配到相应非对角元；
  - 满足时保证 `A=-i H_eff(z) z`；
  - 出现常数驱动或不能按上述规则分配的非线性交叉项时，`H_eff=None` 并返回诊断；
  - 文档明确该矩阵包含耗散漂移，可非 Hermitian，不等同于输入主方程中的物理 Hamiltonian。

## 测试计划

- 验证四种表象的全部左右乘对应关系、符号复系数、dagger、算符幂和多模式全局排序。
- 谐振子与单光子损耗：验证 positive-P 漂移、零扩散及增广 `H_eff`。
- Kerr Hamiltonian：验证
  `A_alpha=-i chi beta alpha^2`、
  `A_beta=i chi alpha beta^2` 及对角扩散
  `(-i chi alpha^2, i chi beta^2)`。
- 二次参量驱动和多模式线性耦合：验证增广交叉 Hamiltonian 元素。
- Wigner 阻尼与非线性模型：验证正常/反常扩散块以及高阶项截断记录。
- 三体相互作用或损耗：验证三阶以上 KM 项被保留、FPE 标记为近似。
- 构造含变量系数的三阶 normal 微分项，验证完整 normal → flux 转换后产生的二阶及低阶项被保留，并验证“先在 normal 形式截断”的错误路径不会重新出现。
- 对一般符号对称扩散矩阵逐元素验证 `simplify(B*B.T-G)==0`，并验证 `M`、`D` 的定义。
- 验证非守恒通用 Liouvillian、不可分解漂移、未知表象和非法模式引用均给出明确异常。
- 更新 README 和示例 notebook，展示结构化主方程、四种表象、KM/FPE/Langevin 三步流程及 breaking migration。

## 假设与依赖

- SDE 统一采用 Itô 解释；首版不提供 Stratonovich 转换。
- 增加 `sympy>=1.12,<2` 运行时依赖、pytest 开发依赖，并将 Python 要求修正为 `>=3.10`。
- 模式顺序由 `MasterEquation(modes=...)` 明确指定，不依据表达式或字母排序猜测。
- 实现建立在当前未提交工作树之上，不回退现有修改；旧 Mathematica notebook 仅作为公式参考，不作为新版 API 的兼容目标。

## 后续 Jacobian 支持

- 定义 Hermitian 协方差矩阵 `R_ij = <alpha_i alpha_j^*>`，并由 Wigner/P/Q 的物理 Hamiltonian 与直接 FPE 扩散块构造闭合方程 `F(R) = -i(H(R)R - R H(R)^dagger) + D(R)`。
- Hermitian 实坐标向量固定为：先按 mode 顺序排列全部对角元；再按 mode 字典序排列上三角非对角元实部；最后按相同字典序排列上三角非对角元虚部。Jacobian 定义为该顺序下 `vectorize(F).jacobian(vectorize(R))`。
- 第一版闭合映射仅接受常数以及单个双线性因子 `alpha_j abar_i -> R_ji`；遇到更高阶矩或无法唯一配对的单项式时明确报告 closure error，不静默采用 Gaussian/Wick 分解。
- 若 anomalous Hamiltonian 或 anomalous diffusion 非零，仅靠 Hermitian `R` 不闭合；此时拒绝生成该 Jacobian，并提示未来使用包含正常与反常二阶矩的增广协方差系统。
