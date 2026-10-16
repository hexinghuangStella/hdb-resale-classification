# HDB 转售房价格分类

使用房屋属性预测 `price_category`：Below Median = 0，Above Median = 1。

## 项目结构

```text
root/
├── eda.ipynb
├── README.md
├── requirements.txt
├── data/
│   └── data.csv
├── src/
│   ├── data_preparation.py
│   ├── model_training.py
│   └── config.yaml
└── main.py
```

## 安装与运行

需要 Python 3.10 或以上版本，在项目根目录运行：

```bash
python -m venv .venv
# Windows PowerShell:
.venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
python main.py
# 自定义配置：
python main.py --config src/config.yaml
# 仅训练指定模型（默认训练全部）：
python main.py --model decision_tree
# 探索数据：
jupyter lab eda.ipynb
```

默认数据为原始 `resale_transactions_categorised.csv`，现已重命名为
`data/data.csv`；数据仍由 .gitignore 排除，不随仓库提交。其他环境需要自行放置同样的 CSV。
数据至少需要 `month`, `flat_type`, `block`, `street_name`, `storey_range`,
`floor_area_sqm`, `lease_commence_date`, `remaining_lease`, `town_id`,
`flatm_id`, `price_category` 列。

## 处理和训练

- 规范列名、移除全空行及完全重复行；缺失或未知目标标签会报错。
- 将剩余租约转换为月数，交易月份拆为年/月，楼层范围转换为中点。
- 使用 town_id / flatm_id 作为类别特征；不使用记录 id 和冗余名称列。
  不需要从全量数据推导 ID 到名称的映射。
- 按标签分层划分训练/验证/测试集，默认 80% / 10% / 10%，随机种子为 42。
- 缺失值填补、数值标准化和类别独热编码仅在训练集拟合，未知类别可直接预测。
- 保留逻辑回归和决策树两种模型，使用相同训练/验证/测试划分，分别输出 accuracy、F1、ROC AUC、混淆矩阵和分类报告。
  逻辑回归会标准化数值特征，决策树跳过标准化；两者独立拟合预处理。
  混淆矩阵的行是真实类别、列是预测类别，顺序为 [Below Median, Above Median]。
- `src/config.yaml` 的 `models` 下按名称配置 `type` 和 `params`，参数直接传给相应 sklearn 分类器。
  支持 `logistic_regression` 和 `decision_tree`；随机种子默认继承 `split.random_state`，可通过模型参数覆盖。
  决策树已配置用户提供的 notebook 最佳参数：`max_depth=20`、`min_samples_leaf=10`、
  `min_samples_split=50`。原实验报告的交叉验证 ROC AUC 为 `0.9709776364989262`，
  此分数未在当前流程中重新做交叉验证；当前运行独立生成验证集和测试集指标。
  原搜索参数的 `classifier__` 前缀属于 Pipeline 参数命名，在这里直接传给分类器，因此需去掉前缀。
- 每个模型分别保存到 `artifacts/<模型名称>/`：
  - `pipeline.joblib`：拟合后的完整预处理和分类管道，推荐用于预测。
  - `preprocessor.joblib`：独立保存的预处理管道。
  - `model.joblib`：独立保存的分类器，输入必须先经过对应预处理器转换。
  - `metrics.json`：该模型的验证集、测试集指标及样本量。
  - `config.yaml`：该次运行的数据路径、划分及所选模型参数。
  同名模型再次运行会覆盖该目录中的产物；旧版根目录下的产物不再更新。
  调参使用验证集，最终测试集仅用于最终评估。

## 加载模型预测

```python
import joblib
import pandas as pd
from src.data_preparation import engineer_features

model = joblib.load("artifacts/decision_tree/pipeline.joblib")
raw = pd.read_csv("data/data.csv").head()
predictions = model.predict(engineer_features(raw))  # 输入不需要目标列
```

## Review 与限制

原有 Python 模块、配置、依赖和 README 为空，notebook 中存在记录 ID 入模、
目标映射不校验、清洗逻辑无法复用等问题。本次补齐完整流程，并让 EDA 复用清洗与切分函数。
notebook 保留原有探索内容；原有名称清洗仅作为 EDA 展示，不参与训练。

当前划分衡量的是同一数据分布下的随机留出表现。如果要预测未来交易，应使用按时间划分，
并确认价格中位数阈值的来源和计算时点；现有 CSV 无法验证标签生成过程是否使用未来信息。
完全重复行会移除，但不推断不同 ID 的相似记录是否为同一笔交易。
依赖使用兼容范围而非锁文件；复现实验时应固定实际安装版本。
