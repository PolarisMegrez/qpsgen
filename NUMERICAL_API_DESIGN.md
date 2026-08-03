# fpgen 数值动力学接口设计说明（评审稿）

> 状态：接口提案，尚未实现。本文用于 fpgen 与数值求解模块之间的接口评审。
>
> 面向对象：负责定态求解、分岔延拓、高阶简并点搜索、GPU 加速和模型代码生成的开发者。
>
> 本文中的“当前接口”指 fpgen 0.2.0 已有实现；“建议接口”均为后续设计，不应被理解为当前可调用 API。

## 1. 目标与边界

fpgen 的职责是从量子主方程得到可靠、可追溯的符号动力学对象。下游数值模块的职责是选择参数区域、生成适合特定后端的模型文件，并执行定态搜索、延拓和高阶分岔分析。

建议维持如下边界：

- fpgen 负责主方程、表象、Fokker--Planck 截断、Langevin 方程、矩方程闭合、实状态向量化、符号导数、结构分析和后端中立的模型描述。
- fpgen 可以提供代码生成所需的表达式和元数据，但不直接依赖 JAX、PyTorch、CUDA 或具体延拓器。
- 下游模块负责数组后端、自动微分、GPU 批处理、线性代数策略、初值管理、延拓、约束优化和高精度复核。
- notebook 继续承担“从主方程推导并展示 Hamiltonian、Diffusion 和 Jacobian”的用途，不暴露代码生成细节。

这一区分可以使展示接口保持简洁，同时避免下游 agent 再次解析 LaTeX、猜测变量顺序或从矩阵表达式反推动力学。

## 2. 数学对象与符号约定

### 2.1 Langevin 层

Fokker--Planck 截断后，fpgen 当前生成增广复状态上的 Itô 方程

\[
\dot{\boldsymbol z}=\boldsymbol A(\boldsymbol z)+\boldsymbol\xi,
\qquad
\langle\boldsymbol\xi(t)\boldsymbol\xi^{\mathsf T}(t')\rangle
=\boldsymbol G_{\mathrm{aug}}(\boldsymbol z)\,\delta(t-t').
\]

这里必须区分两个名称相近但含义不同的矩阵：

- \(\boldsymbol G_{\mathrm{aug}}\) 是增广 Fokker--Planck 扩散矩阵，是复对称矩阵。
- 对 Wigner/P/Q 表象，\(\boldsymbol D=\langle\boldsymbol\xi\boldsymbol\xi^\dagger\rangle\) 是 \(\boldsymbol G_{\mathrm{aug}}\) 的正常关联块；\(\boldsymbol M=\langle\boldsymbol\xi\boldsymbol\xi^{\mathsf T}\rangle\) 是反常关联块。

当漂移可以分解时，当前接口还给出

\[
\dot{\boldsymbol\alpha}
=-i\boldsymbol H(\boldsymbol\alpha,\bar{\boldsymbol\alpha})
\boldsymbol\alpha+\boldsymbol\xi.
\]

这里的 \(\boldsymbol H\) 是包含耗散漂移的有效 Hamiltonian，可以是非厄米矩阵，不等同于输入主方程中的物理 Hamiltonian。

### 2.2 Hermitian 协方差层

当前的协方差计算器定义

\[
R_{ij}=\langle\alpha_i\alpha_j^*\rangle,
\]

并在闭合条件满足时构造

\[
\dot{\boldsymbol R}
=-i\left(\boldsymbol H(\boldsymbol R)\boldsymbol R
-\boldsymbol R\boldsymbol H^\dagger(\boldsymbol R)\right)
+\boldsymbol D(\boldsymbol R).
\]

注意第一项是 \(\boldsymbol H\boldsymbol R\)，不能省略右乘的 \(\boldsymbol R\)。定义实坐标向量 \(\boldsymbol r=\operatorname{vec}_{H}(\boldsymbol R)\) 后，数值问题统一写成

\[
\dot{\boldsymbol r}=\boldsymbol F(\boldsymbol r;\boldsymbol p),
\qquad
\boldsymbol J(\boldsymbol r;\boldsymbol p)
=\frac{\partial\boldsymbol F}{\partial\boldsymbol r}.
\]

当前约定的 Hermitian 向量化顺序为：

1. 对角元，按 `MasterEquation.modes` 的声明顺序；
2. 上三角非对角元实部，按 mode 名称对的字典序；
3. 上三角非对角元虚部，采用相同的 mode 对顺序。

若有 \(N\) 个 mode，则 \(\boldsymbol r\in\mathbb R^{N^2}\)。该顺序必须成为可版本化的接口契约，不能由后端重新排序。

### 2.3 当前闭合条件

当前实现只允许 \(\boldsymbol H\) 和 \(\boldsymbol D\) 的每个相空间单项式是：

- 与相空间变量无关的常数；或
- 单个双线性项 \(\alpha_j\bar\alpha_i\)，映射为 \(R_{ji}\)。

当前实现不会隐式采用 Gaussian/Wick 分解。若存在更高阶矩、无法唯一配对的单项式、positive-P 独立变量，或者非零反常 Hamiltonian 使正常协方差 \(R\) 不闭合，则明确抛出 `CovarianceClosureError`。

## 3. 要搜索的“高阶点”需要先区分

下游接口不应把以下两个问题混为一谈。

### 3.1 不动点合并或高重根

定态满足

\[
\boldsymbol F(\boldsymbol r;\boldsymbol p)=0.
\]

若消元后只剩一个序参量 \(q\) 和一个标量方程 \(P(q;\boldsymbol p)=0\)，则 \(m\) 重根满足

\[
P=\partial_qP=\cdots=\partial_q^{m-1}P=0,
\qquad
\partial_q^mP\ne0.
\]

这是固定点分支合并的代数条件。

### 3.2 动力学 Jacobian 的 exceptional point

另一个问题是 \(\boldsymbol J\) 的特征值和特征向量合并。高阶 EP 需要检查几何重数或 Jordan 链，而不只是特征多项式有重根。以 \(\lambda\) 为目标特征值时，可采用增广条件

\[
(\boldsymbol J-\lambda\boldsymbol I)\boldsymbol v_0=0,
\qquad
(\boldsymbol J-\lambda\boldsymbol I)\boldsymbol v_k=\boldsymbol v_{k-1},
\]

再加尺度和相位规范条件。稳态分岔通常关注 \(\lambda=0\)，但非零 \(\lambda\) 的 EP 也可能存在。

不动点高重根和 Jacobian EP 可以相关，但并不等价。数值模块应显式声明搜索目标，例如：

- `fixed_point_multiplicity`：约化定态方程的根重数；
- `zero_eigenvalue_degeneracy`：定态 Jacobian 在零特征值处的简并；
- `jacobian_exceptional_point`：指定特征值处的 Jordan 缺陷。

## 4. 当前可用接口

当前 notebook 和 Python API 的主路径为：

```python
km = derive_kramers_moyal(master, representation="wigner")
fpe = km.truncate(2)
langevin = fpe.to_langevin()
jac = langevin.to_covariance_jacobian()
```

`jac` 是 `HermitianCovarianceJacobian`，已经提供：

| 字段 | 含义 |
| --- | --- |
| `modes` | mode 名称及声明顺序 |
| `covariance` | 由实坐标重建的 Hermitian 矩阵 \(R\) |
| `coordinates` | 实坐标符号，顺序即公共向量化顺序 |
| `pair_order` | 上三角 mode 对的字典序 |
| `hamiltonian` | 完成双线性闭合后的 \(H(R)\) |
| `diffusion` | 完成双线性闭合后的正常扩散 \(D(R)\) |
| `flow_matrix` | \(\dot R\) |
| `flow_vector` | \(F(r;p)\) |
| `jacobian` | 稠密符号 Jacobian \(\partial F/\partial r\) |

### 4.1 当前接口不适合直接 codegen 的部分

1. `coordinates` 使用 `R_{aa}` 和 `\operatorname{Re}(R_{ab})` 一类展示名称作为 SymPy 符号名。它们适合 LaTeX，不适合作为 Python、JSON、JAX 或 CUDA 标识符。
2. 参数没有显式声明顺序。目前只能从表达式的 `free_symbols` 推断，容易把状态、参数、共轭参数和辅助符号混在一起。
3. Jacobian 在创建对象时立即完整符号化。对大系统而言，构造稠密矩阵可能比计算 `Jv` 昂贵得多。
4. 没有 RHS-only 的模型契约、稀疏结构、JVP/VJP、公共子表达式或后端代码生成信息。
5. 没有记录数值尺度、参数范围、物理解域或状态约束。
6. 没有用户所需的“选择序参量并线性消去其余变量”接口。
7. 反常 Hamiltonian 和 positive-P 的二阶矩动力学还没有相应的增广状态模型。

## 5. 建议的对象层次

建议让 `CovarianceDynamics` 成为数值接口的核心对象，Jacobian 则改为按需派生。现有 `HermitianCovarianceJacobian` 保留为兼容和展示外观。

```python
dynamics = langevin.to_covariance_dynamics(
    parameters=[omega_a, omega_b, g, gamma_a, gamma_b, Gamma],
)

dynamics.covariance       # R(r)
dynamics.hamiltonian      # H(r, p)
dynamics.diffusion        # D(r, p)
dynamics.rhs_matrix       # dot(R)
dynamics.rhs              # F(r, p), real flat vector
dynamics.state_spec       # stable state metadata
dynamics.parameter_spec   # stable parameter metadata
dynamics.provenance       # representation/truncation/closure information

J = dynamics.jacobian()   # only when a dense symbolic J is requested
plan = dynamics.linear_reduce(order_parameters=["r_diag_0"])
spec = dynamics.to_model_spec(name="two_mode_vdp")
```

现有调用保持有效：

```python
jac = langevin.to_covariance_jacobian()
```

其内部可以改为创建 `CovarianceDynamics`，再物化兼容对象。因此 notebook 的展示风格无需变化。

### 5.1 `StateVariableSpec`

每个数值状态不能只用一个字符串表示，建议至少包含：

```python
@dataclass(frozen=True)
class StateVariableSpec:
    id: str                 # ASCII 内部 ID，如 r_diag_0、r_re_0_1
    index: int              # 在实状态向量中的位置
    symbol: sympy.Symbol    # 机器符号，如 x0、x1，而不是 LaTeX 名称
    latex_name: str         # 展示标签
    kind: str               # diag | offdiag_re | offdiag_im
    matrix_index: tuple[int, int]
    mode_names: tuple[str, str]
    real: bool = True
```

建议内部 ID 使用矩阵索引而不是拼接 mode 名称，以避免 Unicode、复合名称和命名碰撞。示例：

| `id` | `matrix_index` | `mode_names` | `latex_name` |
| --- | --- | --- | --- |
| `r_diag_0` | `(0, 0)` | `("a", "a")` | `R_{aa}` |
| `r_re_0_1` | `(0, 1)` | `("a", "b")` | `\operatorname{Re}R_{ab}` |
| `r_im_0_1` | `(0, 1)` | `("a", "b")` | `\operatorname{Im}R_{ab}` |

`symbol` 建议使用 `x0, x1, ...`。显示名称和计算符号彻底分离后，notebook 与 codegen 不再互相限制。

### 5.2 `ParameterSpec`

参数顺序必须由模型作者显式给出。数值导出不应依赖 `free_symbols` 的排序。

```python
@dataclass(frozen=True)
class ParameterSpec:
    id: str                 # p0, p1, ...
    index: int
    symbol: sympy.Symbol    # 原始 SymPy 参数
    name: str               # 人类可读名称
    latex_name: str
    domain: str             # real | positive | nonnegative | complex_split
    default: float | None
    scale: float | None
    bounds: tuple[float | None, float | None] | None
```

在本项目当前模型中，\(\omega,g,\kappa,\chi\) 等未特别声明为复数的参数按实数处理。不过通用导出器不应静默假定未知参数为实数。建议采用以下规则：

- SymPy 已有 `real=True`、`positive=True` 等假设时直接保留；
- 假设未知时，数值导出要求调用者补充 `parameter_domains`；
- 真正的复参数在实数内核中拆成实部和虚部两个参数；
- 若表达式中仍出现未消去的 `conjugate(parameter)`，导出前报错。

### 5.3 `CovarianceDynamics`

建议字段如下：

```python
@dataclass(frozen=True)
class CovarianceDynamics:
    representation: str
    modes: tuple[str, ...]
    state_spec: tuple[StateVariableSpec, ...]
    parameter_spec: tuple[ParameterSpec, ...]
    covariance: sympy.Matrix
    hamiltonian: sympy.Matrix
    diffusion: sympy.Matrix
    rhs_matrix: sympy.Matrix
    rhs: sympy.Matrix
    provenance: DerivationProvenance
```

建议按需方法为：

```python
dynamics.jacobian(sparse=False)
dynamics.jacobian_sparsity()
dynamics.jvp(direction)
dynamics.parameter_jacobian()
dynamics.directional_derivative(direction, order=2)
dynamics.linear_reduce(...)
dynamics.to_model_spec(...)
```

不建议默认创建完整 Hessian 或更高阶导数张量。高阶分岔算法通常只需要沿零空间方向的收缩量，可以通过重复 JVP 或方向导数得到，避免 \(O(n^k)\) 的张量存储。

### 5.4 `DerivationProvenance`

下游模型必须知道自己求解的是精确方程还是截断/闭合后的方程。建议记录：

```text
representation
phase_space_variable_order
fpe_truncation_order
original_kramers_moyal_order
discarded_term_count
fpe_is_exact
stochastic_interpretation = "ito"
moment_closure = "exact_bilinear" | ...
closure_assumptions
hamiltonian_factorization_diagnostics
fpgen_version
state_layout_version
```

若 FPE 截断丢弃了高阶项，或者矩闭合使用了近似，生成模型时应提供机器可读标志，不能只在 notebook 文本中提示。

## 6. 后端中立的 `ModelSpec`

`ModelSpec` 是 fpgen 与下游 agent 的主要交接对象。第一版可以是只在 Python 内存中使用的冻结 dataclass，不必立即承诺稳定 JSON 格式。

```python
@dataclass(frozen=True)
class ModelSpec:
    schema_version: str
    name: str
    state: tuple[StateVariableSpec, ...]
    parameters: tuple[ParameterSpec, ...]
    rhs: tuple[sympy.Expr, ...]
    hamiltonian: sympy.Matrix | None
    diffusion: sympy.Matrix | None
    jacobian_sparsity: SparsityPattern
    provenance: DerivationProvenance
```

导出时应把原始状态/参数符号替换为位置符号 `x0...`、`p0...`。这样表达式图与显示名称无关，并可直接生成数组索引代码。

不建议把 `sympy.srepr()` 或可执行 Python 字符串直接作为长期交换格式：它们不构成稳定、安全的 schema。若未来确实需要跨进程或跨语言传递，再定义带版本号的白名单表达式 DAG，例如只允许：

```text
constant, rational, state_ref, parameter_ref,
add, mul, integer_power, exp, log, sqrt, sin, cos
```

不在白名单中的 SymPy 函数必须在导出阶段失败，而不是在 GPU 编译时失败。

## 7. 建议的生成模型协议

以 JAX 为例，生成的模型文件建议提供以下纯函数。核心函数只处理单个样本，批处理交给 `vmap`，以保持形状和编译缓存稳定。

```python
STATE_SIZE = n_state
PARAMETER_SIZE = n_parameter
STATE_NAMES = (...)
PARAMETER_NAMES = (...)
MODEL_INFO = {...}  # 普通 Python 元数据，不进入 jit 内核

def rhs(x, p):
    """(n_state,), (n_parameter,) -> (n_state,), real float64"""

def jvp(x, p, tangent):
    """Return d(rhs)/d(x) @ tangent."""

def jacobian(x, p):
    """Optional dense Jacobian for small systems."""

def hamiltonian(x, p):
    """Optional complex (n_mode, n_mode) diagnostic output."""

def diffusion(x, p):
    """Optional complex (n_mode, n_mode) diagnostic output."""

def reconstruct_covariance(x):
    """Return complex Hermitian R for diagnostics and constraints."""
```

接口上的重要约束：

- 数值求根只依赖实函数 `rhs(x, p)`；复杂矩阵仅作为可选观测量输出。
- 内核使用平坦数组，不在 `jit` 函数内部传递字典或 SymPy 对象。
- 默认使用 float64。GPU 粗筛可允许显式选择 float32，但高阶简并判定不能默认使用 float32。
- `jacobian` 是可选入口。大模型优先用自动微分 JVP/VJP 和 Newton--Krylov。
- `rhs`、`hamiltonian`、`diffusion` 分开生成，避免只求 RHS 时计算全部展示量。若下游确实需要算子融合，可另生成 `rhs_and_observables` 并联合执行公共子表达式消除。
- 公共子表达式消除应在每个实际入口的表达式集合上执行，不能强制把未请求的 \(H,D,J\) 都融合进 RHS。
- JAX 后端可由 `jax.jvp`/`jax.vjp` 产生导数；NumPy/C 后端可以使用 fpgen 给出的符号 JVP。两者需有一致性测试。

主接口可设计为：

```python
spec = dynamics.to_model_spec(name="two_mode_vdp")
spec.generate(
    backend="jax",
    path="generated/two_mode_vdp.py",
    include=("rhs", "jvp", "hamiltonian", "diffusion"),
    dtype="float64",
)
```

文件写入属于显式的 `generate` 操作；创建 `ModelSpec` 本身不应产生文件系统副作用。

## 8. 线性消元接口

用户此前采用的方法是合理的，并且非常适合作为 fpgen 的结构分析功能。设状态和方程分别分块为

\[
\boldsymbol r=(\boldsymbol q,\boldsymbol y),
\qquad
\boldsymbol F=(\boldsymbol g,\boldsymbol h),
\]

其中 \(\boldsymbol q\) 是保留的序参量，而 \(\boldsymbol y\) 是待消元变量。若

\[
\boldsymbol h(\boldsymbol q,\boldsymbol y;\boldsymbol p)
=\boldsymbol A(\boldsymbol q;\boldsymbol p)\boldsymbol y
+\boldsymbol b(\boldsymbol q;\boldsymbol p),
\]

则在 \(\boldsymbol A\) 可逆的正则分支上

\[
\boldsymbol y_*(\boldsymbol q;\boldsymbol p)
=-\boldsymbol A^{-1}\boldsymbol b,
\qquad
\boldsymbol G(\boldsymbol q;\boldsymbol p)
=\boldsymbol g(\boldsymbol q,\boldsymbol y_*;\boldsymbol p).
\]

实现中不应真的先构造 `A.inv()`。应使用符号线性求解、分数无消元或数值 `solve(A, -b)`。

建议接口：

```python
plan = dynamics.linear_reduce(
    order_parameters=["r_diag_0"],
    eliminated_equations="matching",  # 或显式给出 residual indices
    verify_linearity=True,
)
```

建议返回 `LinearReductionPlan`：

```python
@dataclass(frozen=True)
class LinearReductionPlan:
    order_parameters: tuple[str, ...]
    eliminated_variables: tuple[str, ...]
    retained_equations: tuple[int, ...]
    eliminated_equations: tuple[int, ...]
    A: sympy.Matrix
    b: sympy.Matrix
    retained_residual: sympy.Matrix
    linearity_residual: sympy.Matrix
    regularity_conditions: tuple[sympy.Expr, ...]
    singular_branch_condition: sympy.Expr | None
    diagnostics: tuple[str, ...]
```

### 8.1 两种执行模式

`LinearReductionPlan` 应支持两种模式，而不是强制生成显式有理式。

**符号代入模式**适合小矩阵：

```python
reduced = plan.materialize(method="fraction_free")
reduced.eliminated_solution
reduced.residual
reduced.denominators
```

它适合进一步求 resultant、subresultant、GCD 或多项式重根条件。

**凝聚数值模式**适合较大模型和 GPU：

```python
def reduced_rhs(q, p):
    A, b = assemble_linear_block(q, p)
    y = solve(A, -b)
    return retained_residual(q, y, p)
```

这种方式避免显式逆矩阵和巨大的符号分式。JAX 可以对线性求解自动微分；也可以使用隐式公式

\[
\frac{d\boldsymbol y_*}{dq}
=-\boldsymbol A^{-1}
\left(\frac{\partial\boldsymbol b}{\partial q}
+\frac{\partial\boldsymbol A}{\partial q}\boldsymbol y_*\right)
\]

构造约化 JVP。若 \(A\) 稀疏且规模较大，后端可以根据 `A` 的稀疏结构替换线性求解器。

### 8.2 不可丢失奇异分支

上述消元只在 \(\boldsymbol A\) 可逆时成立。`LinearReductionPlan` 必须保留这一前提，并显式暴露奇异分支信息：

\[
\det\boldsymbol A=0.
\]

对大矩阵不建议真的展开行列式。符号侧可以只记录“要求 `A` 满秩”，数值侧监控最小奇异值、秩揭示 QR 或条件数。若求解目标靠近该分支，应回到完整方程或改换消元变量。

任何通过通分得到的约化多项式都必须同时记录被乘掉的分母。否则会引入伪根，并可能误删真正位于奇异分支上的解。

### 8.3 标量约化后的解析路线

若只保留一个序参量，且约化残差是有理函数

\[
G(q;\boldsymbol p)=\frac{P(q;\boldsymbol p)}{Q(q;\boldsymbol p)},
\]

则在 \(Q\ne0\) 和 `A` 满秩的分支上，只需研究分子 \(P\)。不需要求出高次多项式的根式表达式；可直接使用：

- \(P,P',\ldots,P^{(m-1)}\) 的公共零点；
- `gcd(P, P')` 判断重复因子；
- resultant 或 subresultant 条件消去 \(q\)；
- 对候选参数点使用高精度数值代回验证。

该路线通常比“先求全部定态解析根，再分析根的合并”更稳健。

### 8.4 多序参量约化

当 \(\boldsymbol q\) 有多个分量时，Gröbner basis 或多变量 resultant 只适合变量少、次数低的模型。一般情形建议：

1. 数值定位满足 \(G=0\) 且约化 Jacobian 近奇异的候选点；
2. 求左右零向量并做局部 Lyapunov--Schmidt 约化；
3. 通过方向导数和辅助线性方程递归计算低维分岔系数；
4. 用高精度算术验证消失条件和非退化条件。

fpgen 应提供 RHS、JVP、VJP 所需的符号信息和方向导数，但不必在第一版内置所有分岔正规形。

## 9. 建议的数值求解框架

### 9.1 阶段一：GPU 粗筛

在参数网格或低差异采样点上并行执行：

- 多初值定态求解；
- 残差范数；
- Jacobian 最小奇异值或若干最小模特征值估计；
- 线性消元块 \(A\) 的条件估计；
- 物理约束和有限值检查。

GPU 的主要收益来自大量同构小问题的 `vmap + jit`，而不是单个很小矩阵的线性代数。

### 9.2 阶段二：候选点精修

- 小系统使用带缩放的 trust-region Newton；
- 大系统使用 Newton--Krylov 和 JVP；
- 对奇异根使用增广方程、伪弧长条件或 deflation；
- 同时优化状态和待调参数，而不是只在固定参数下反复碰撞奇异点；
- 对实变量、参数和残差分别设置尺度，避免阈值依赖量纲。

### 9.3 阶段三：分支延拓

用伪弧长延拓跟踪固定点分支，记录：

- `rhs` 残差；
- Jacobian 谱或最小奇异值；
- 约化分母和 `A` 的条件数；
- 物理状态约束；
- 分岔测试函数的符号变化或极小值。

仅做参数网格扫描很容易漏掉窄分支和转折点，因此扫描应主要用于提供延拓起点。

### 9.4 阶段四：高精度复核

高阶点对舍入误差极其敏感。建议 GPU/float64 用于筛选，最终候选点使用 CPU arbitrary precision 复核：

- 完整方程和约化方程残差；
- 根重数条件或 Jordan 链残差；
- 相关矩阵的秩和奇异值间隙；
- 正则分支条件及所有被清除的分母；
- 提高精度后候选参数和判据是否收敛。

## 10. 稀疏性、自动微分和算子融合

对于 \(N\) 个 mode，Hermitian 状态已有 \(N^2\) 个实变量，稠密 Jacobian 有 \(N^4\) 个元素。接口应从一开始允许 matrix-free 用法。

建议 `ModelSpec` 至少给出 RHS 的结构稀疏性：

```python
row_ptr / col_ind
or
tuple[(row, col), ...]
```

但数值后端不一定要显式生成稀疏 Jacobian。优先级建议为：

1. 纯 `rhs`；
2. JVP/VJP；
3. 稀疏模式；
4. 小系统的稠密 Jacobian；
5. 按需的高阶方向导数；
6. 最后才是完整高阶导数张量。

代码生成时应在实向量化之后做代数化简和公共子表达式消除。若先对复杂的矩阵展示表达式分别化简，再拼成 RHS，可能错失跨矩阵元的公共子表达式。

同时要限制符号化简的成本。建议提供可配置等级：

```text
none        只做必要替换
basic       cancel/together 的受控组合
cse         在目标入口上做公共子表达式消除
aggressive  仅由用户显式请求
```

## 11. 增广二阶矩与 positive-P 的后续扩展

当前 `HermitianCovarianceJacobian` 只处理正常协方差 \(R\)。简并参量振荡器等模型含有反常 Hamiltonian 块，\(R\) 会与反常二阶矩耦合，因此必须扩充状态。positive-P 的第二组变量又不是第一组的复共轭，也不能套用 Hermitian \(R\) 的布局。

建议不要把这些情况硬塞进 `CovarianceDynamics` 的现有闭合规则，而是在公共基类下引入更一般的对象：

```text
MomentDynamics
├── HermitianCovarianceDynamics       当前 R 状态
├── AugmentedSecondMomentDynamics     正常 + 反常二阶矩
└── PositivePMomentDynamics           独立 alpha/beta 矩
```

三者最终都必须提供相同的数值最小接口：

```python
state_spec
parameter_spec
rhs
jvp(...)
to_model_spec(...)
```

差异由 `state_layout.kind`、矩重建映射和约束元数据表达。数值后端只消费实向量 RHS，不需要理解量子表象；展示层仍可针对不同对象给出对应矩阵。

对复独立矩变量，数值边界应明确拆成实部和虚部，不应把后端是否支持 complex autodiff 当作模型正确性的前提。

## 12. 物理约束与数值尺度

fpgen 可以提供约束的结构，但不应擅自把所有约束变成求解器硬约束。建议 `ModelSpec` 支持可选信息：

```text
state_scales
parameter_scales
state_bounds
parameter_bounds
matrix_constraints: Hermitian, PSD, symmetry, ...
constraint_provenance: exact | user_supplied | heuristic
```

例如 \(R\) 的 Hermitian 性已经由向量化构造保证，无需数值约束；半正定性可以作为候选解过滤条件。表象、截断和矩定义不同可能影响可接受区域，因此未由推导严格得到的约束必须标为用户配置。

## 13. 验证要求

新增接口至少应通过以下测试。

### 13.1 符号层

- `vectorize(reconstruct(r)) == r`，并保持既定顺序；
- \(\dot R=-i(HR-RH^\dagger)+D\) 与 `rhs` 的向量化完全一致；
- 对声明为实的状态和参数，`rhs` 的每个元素都能证明为实；
- `jacobian()` 等于 `rhs.jacobian(state)`；
- 符号 JVP 等于稠密 Jacobian 与方向向量的乘积；
- 未满足闭合条件时明确失败，不静默引入 Wick 分解；
- 状态 ID、参数 ID 和顺序在重复运行中稳定。

### 13.2 线性消元

- 精确验证 `h - A*y - b == 0`；
- 将消元解代回完整方程后，约化残差与保留方程一致；
- 非线性依赖于 `y` 时拒绝生成 plan，并指出具体单项式；
- 正则条件和通分分母不为空时必须保留；
- 在随机非奇异数值点上，完整定态解与凝聚残差一致；
- 在 `A` 近奇异的点发出结构化诊断，而不是返回无穷或 NaN 后继续。

### 13.3 代码生成

- NumPy/JAX 结果与 SymPy 在随机实点逐项一致；
- JAX JVP 与符号 JVP、有限差分在合理容差内一致；
- float64 模式确实启用；
- 可选输出不会改变 RHS；
- CSE 前后数值结果一致；
- 生成文件不依赖 LaTeX 符号名或 `free_symbols` 的偶然顺序。

### 13.4 回归模型

至少保留以下模型作为端到端基准：

- 两模 van der Pol：当前 Hermitian 协方差 Jacobian；
- 两模 Kerr 和 cross-Kerr：非线性 \(H(R)\)；
- pair hopping：非对角非线性 Hamiltonian；
- 简并参量振荡器：反常 Hamiltonian，当前正常协方差闭合必须拒绝，未来增广矩接口应通过；
- positive-P Kerr：独立增广变量，不能误用 Hermitian 协方差布局。

## 14. 建议实施顺序

### 阶段 A：不改变 notebook 行为的内部重构

1. 将机器符号与 LaTeX 标签分离；
2. 新增 `StateVariableSpec`、`ParameterSpec` 和 `CovarianceDynamics`；
3. 让 `to_covariance_jacobian()` 成为兼容外观；
4. 增加按需 dense Jacobian、JVP 和稀疏模式；
5. 补齐 provenance 和严格的参数实数检查。

### 阶段 B：数值交接接口

1. 新增内存中的 `ModelSpec`；
2. 先实现一个参考 NumPy 后端，作为正确性基准；
3. 与数值 agent 共同实现 JAX 后端及 `jit/vmap/jvp` 测试；
4. 再决定是否需要稳定 JSON/DAG schema。

### 阶段 C：结构化线性消元

1. 识别并验证 \(A(q,p)y+b(q,p)\)；
2. 同时提供符号代入与凝聚数值模式；
3. 保留正则条件、分母和奇异分支诊断；
4. 为标量约化提供 numerator/denominator 和重根条件辅助函数。

### 阶段 D：更一般的矩动力学

1. 增广正常/反常二阶矩；
2. positive-P 独立矩变量；
3. 统一到 `MomentDynamics -> ModelSpec` 最小数值协议。

## 15. 请数值计算 agent 重点评审的问题

1. `rhs(x, p)` 使用两个固定顺序的一维数组，是否适合计划中的 GPU 批处理和模型文件组织？
2. 批处理应由调用侧 `vmap` 完成，还是生成函数需要原生接受 `(..., n_state)` 的前导批维？
3. 下游主要需要 JVP、VJP、稠密 Jacobian还是显式稀疏 Jacobian？预期状态规模是多少？
4. `LinearReductionPlan` 的凝聚模式在内核中调用 `solve(A, -b)` 是否可接受？是否需要暴露 `A` 的块结构以便替换求解器？
5. 首个后端是否确定为 JAX，还是需要同时支持 NumPy/PyTorch？
6. 模型交接只需 Python 内存对象和生成的 `.py` 文件，还是必须支持 JSON/跨语言加载？
7. 目标“高阶不动点分岔”具体指约化方程高重根、零特征值高阶简并，还是一般非零特征值 EP？这会决定增广求解条件。
8. 候选点最终是否会用 arbitrary precision 复核？若不会，需要共同制定 float64 下可辩护的条件数与残差标准。
9. 参数范围、缩放和物理解域由模型作者提供，还是由统一配置文件提供？
10. 生成模型是否需要同时返回 \(H,D,R\) 供在线诊断，还是这些量只在离线验证时使用？

## 16. 建议的最小可用版本

对第一轮联调，建议只承诺以下内容：

```python
dynamics = langevin.to_covariance_dynamics(parameters=[...])
spec = dynamics.to_model_spec(name=...)

spec.state
spec.parameters
spec.rhs
spec.jacobian_sparsity
spec.provenance
```

以及一个参考生成器：

```python
spec.generate(backend="numpy", path=...)
```

该版本即可让数值 agent 验证状态/参数布局、模型文件协议和求根流程。JAX、线性消元、高阶方向导数及增广矩动力学可以在接口契约确认后逐步加入，避免在数值侧尚未评审前固化过多实现细节。

