# agent.md

## UrbanGraph-SG Agent Specification

**project_name:** UrbanGraph-SG
**project_goal:** Build a GraphRAG-powered urban knowledge navigator for Singapore, integrating multi-source open data (LTA, NEA, SingStat, URA, OneMap, HDB) into a unified knowledge graph and enabling natural-language QA with provenance-grounded answers and interactive graph visualization.
**primary_objectives:** knowledge graph construction, GraphRAG indexing pipeline, natural-language QA, answer provenance and traceability, interactive graph visualization, reproducible deployment.

**project_type:** Knowledge Graph + LLM + RAG（非时序预测，非传统 MLOps）

## §1 项目定位与边界

### 1.1 一句话定位

> A GraphRAG-powered urban knowledge navigator that transforms Singapore's open data ecosystem into a queryable knowledge graph, answering natural-language questions with source-attributed, evidence-bound responses and interactive graph visualizations.

### 1.2 与现有项目的差异化

| 维度 | UrbanFlow-AU | UrbanGraph-SG |
|---|---|---|
| 数据类型 | 结构化表格（行人传感器 + 天气） | 多源异构 → 统一知识图谱 |
| 核心方法 | 监督学习回归（LightGBM） | LLM + 知识图谱 + RAG |
| 输出形态 | 数值预测 + 折线图 | 自然语言答案 + 溯源引用 + 图谱可视化 |
| 评价方式 | MAE / RMSE / WAPE | 答案准确性 / 溯源完整性 / 覆盖度 |
| 前沿性 | ML 工程（成熟技术栈） | GraphRAG + Agent（2024-2026 前沿） |
| 训练范式 | 有监督训练 + 回测 | 零样本 / 检索增强（无需训练模型） |

### 1.3 明确不做

- 不训练任何预测模型（回归、分类、时序）
- 不做传统 MLOps（MLflow、Evidently 与 UrbanFlow-AU 重复）
- 不做实时流处理
- 不构建通用聊天机器人
- 不把 LLM 用于生成未经溯源的内容——所有答案必须绑定知识图谱来源
- 不追求覆盖新加坡所有数据领域——聚焦交通 × 天气 × 城市空间三大维度

---

## §2 全局约定

### 2.1 数据约定

- 所有原始数据缓存在 `data/raw/` 下按来源分子目录：`lta/`、`nea/`、`singstat/`、`ura/`、`onemap/`、`hdb/`
- 每次数据采集生成 manifest（`data/manifests/`），记录时间范围、记录数、文件哈希
- 不允许把 API Key 或 token 写入仓库
- 所有时间戳使用 `Asia/Singapore`（UTC+8），转换必须显式记录
- 图数据（Neo4j dump / CSV nodes & edges）放在 `data/graph/`

### 2.2 代码约定

- Python 3.11+，类型注解覆盖率 ≥ 80%
- 所有外部 API 调用必须带超时 + 指数退避重试（最多 3 次）
- LLM 调用必须记录：模型名、prompt 版本、token 用量、响应时间
- GraphRAG 每次 index run 必须记录 config hash 和所用 prompt 版本
- 不做 `print` 调试——用 `logging` 模块，至少分 INFO / WARNING / ERROR 三级

### 2.3 回答约定

- 每个系统回答必须附带来源引用（source citation）：至少包含数据来源 + 实体名称 + 时间范围
- 如果知识图谱无法覆盖某个问题，系统必须显式说"I don't know"并给出缺失范围，禁止 LLM 自由发挥
- 数值类回答必须带单位和时间上下文（如"2025 年 11 月 CBD 区域，暴雨天（降雨 >20mm/h）MRT 客流比晴天同期高 18-32%——来源：LTA + NEA，2025 年数据"）

### 2.4 不把 LLM 或 RAG 强行加入 UrbanFlow-AU

- UrbanGraph-SG 是独立的 LLM/KG 项目，不污染 UrbanFlow-AU 的纯 ML 定位
- 两个项目代码库完全分离，各自有独立的依赖、Docker 配置、CI

---

## §3 Agent 体系

```
                    ┌─────────────────────┐
                    │  Orchestrator       │
                    │  (手动 / Prefect)    │
                    └──────┬──────────────┘
           ┌───────────────┼───────────────────┐
           │               │                   │
    ┌──────▼──────┐ ┌──────▼──────┐   ┌────────▼────────┐
    │ Ingestion   │ │ Processing  │   │ GraphRAG Index  │
    │ Agents (×6) │ │ Agent       │   │ Pipeline (×4)   │
    └──────┬──────┘ └──────┬──────┘   └────────┬────────┘
           │               │                   │
           └───────┬───────┘                   │
                   ▼                           ▼
           ┌──────────────┐           ┌──────────────┐
           │  Neo4j        │◄──────────│ Graph Store  │
           │  Graph Store  │           │ Agent        │
           └──────┬───────┘           └──────────────┘
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
  ┌─────────┐ ┌───────┐ ┌─────────┐
  │ Local   │ │Global │ │ Cypher  │
  │ Search  │ │Search │ │ Query   │
  │ Agent   │ │Agent  │ │ Agent   │
  └────┬────┘ └───┬───┘ └────┬────┘
       │          │          │
       └──────────┼──────────┘
                  ▼
         ┌───────────────┐
         │  Answer Gen   │
         │  Agent        │
         └───────┬───────┘
                 ▼
         ┌───────────────┐
         │  Streamlit    │
         │  UI Agent     │
         └───────────────┘
```

### 3.1 Ingestion Agent: LTA

**职责**
从 LTA DataMall API 采集 MRT 线路/站点、巴士线路/站点/到站时间、出租车可用性、交通流量/速度数据。历史回填 + 增量更新双模式。

**触发条件**
- 手动 `make ingest-lta`
- 增量：每小时 scheduler（Prefect）
- 历史回填：`workflow_dispatch`

**输入**
- `start_date` / `end_date`
- `data_types`: `mrt_stations`, `bus_stops`, `bus_services`, `bus_arrival`, `taxi_availability`, `traffic_speed`
- LTA AccountKey（环境变量 `LTA_ACCOUNT_KEY`）

**输出**
- `data/raw/lta/mrt_stations.parquet`
- `data/raw/lta/bus_stops.parquet`
- `data/raw/lta/bus_services.parquet`
- `data/raw/lta/bus_arrival/*.parquet`
- `data/raw/lta/taxi_availability/*.parquet`
- `data/raw/lta/traffic_speed/*.parquet`
- `data/manifests/lta_manifest.json`

**失败处理**
- API 限流 (429)：指数退避 1s/5s/25s，最多 3 次
- 单 data_type 失败不阻塞其他
- 连续 3 次失败 → 写入告警 + 中断下游

### 3.2 Ingestion Agent: NEA (Weather)

**职责**
从 data.gov.sg NEA API 采集降雨、温度、湿度、PSI、风速等气象数据。

**输出**
- `data/raw/nea/rainfall.parquet`
- `data/raw/nea/temperature.parquet`
- `data/raw/nea/psi.parquet`
- `data/raw/nea/wind.parquet`
- `data/manifests/nea_manifest.json`

### 3.3 Ingestion Agent: SingStat

**职责**
从 SingStat API 采集人口统计、家庭收入、区域规划人口等结构化统计数据。

**输出**
- `data/raw/singstat/population.parquet`
- `data/raw/singstat/household.parquet`
- `data/manifests/singstat_manifest.json`

### 3.4 Ingestion Agent: URA / OneMap

**职责**
从 URA 开放数据和 OneMap API 采集规划区域边界、用地类型、POI 坐标。

**输出**
- `data/raw/ura/planning_areas.geojson`
- `data/raw/ura/land_use.parquet`
- `data/raw/onemap/poi.parquet`
- `data/manifests/geo_manifest.json`

### 3.5 Ingestion Agent: HDB

**职责**
从 data.gov.sg HDB API 采集组屋转售价格、租金中位数。

**输出**
- `data/raw/hdb/resale_prices.parquet`
- `data/raw/hdb/rental_median.parquet`
- `data/manifests/hdb_manifest.json`

### 3.6 Ingestion Agent: School Holidays & Public Holidays

**职责**
通过 `holidays` Python 包和 MOE 学校假期日历生成新加坡假期/学期标记。

**输出**
- `data/raw/calendar/sg_holidays.csv`
- `data/raw/calendar/school_terms.csv`

---

### 3.7 Data Processing Agent

**职责**
对所有 raw 数据执行以下处理：
1. **统一时间索引**：所有时间戳转 `Asia/Singapore`
2. **空间标准化**：经纬度统一 WGS84，验证坐标在新加坡边界内（lat: 1.15-1.48, lon: 103.60-104.10）
3. **实体名称去重 + 规范化**：MRT 站名统一（如 "Dhoby Ghaut" vs "Dhoby Ghaut Station"）
4. **缺失值处理**：标记而非填充（保留数据质量信息）
5. **异常值标记**：超出合理范围的值打 tag（如负降雨量、超出新加坡范围的 GPS）
6. **GraphRAG 输入格式转换**：生成 entity description CSV 和 relationship CSV

**触发条件**
- Ingestion Agent 成功后自动触发
- 手动 `make process`

**输入**
- `data/raw/*`
- `configs/schema_mapping.yaml`（原始字段 → 标准字段映射）

**输出**
- `data/processed/entities/*.csv`（每种实体类型一个文件）
- `data/processed/relationships/*.csv`
- `data/processed/community_texts/*.txt`（每个区域/主题一段描述文本，供 GraphRAG community summarization）
- `reports/data_quality/processing_report.json`

**关键规则**
- 实体 ID 必须全局唯一，前缀标识来源：`lta-mrt-xxx`, `nea-station-xxx`, `ura-area-xxx`
- 同一个物理实体（如一个 MRT 站）在不同数据源中可能以不同名称出现——必须做实体对齐（entity resolution），至少处理精确匹配 + 模糊匹配（Levenshtein < 3）
- 关系必须有方向性且带时间范围（如 `[AFFECTS_DURING]` 关系附带 `{start_date, end_date}`）

**失败处理**
- 实体对齐冲突：生成冲突报告，人工审核后处理
- 单个数据源处理失败：隔离，不阻塞其他源
- 关键字段缺失率 > 10%：阻断并标记

---

### 3.8 GraphRAG Indexing Pipeline

这是项目的核心引擎。基于微软 GraphRAG 框架，但做新加坡城市数据的定制化改造。

#### 3.8.1 Entity Extraction Agent

**职责**
调用 LLM 从结构化数据和文本描述中提取城市实体，生成标准 GraphRAG entity table。

**自定义 prompt 设计**
与默认 GraphRAG 不同，这里需要引导 LLM 识别**城市领域的实体类型**：

| GraphRAG 默认类型 | UrbanGraph-SG 映射 |
|---|---|
| ORGANIZATION | 政府机构 (LTA, NEA, URA, HDB) |
| PERSON | N/A（不使用个人数据） |
| GEO | MRT 站、巴士站、规划区域、POI、邮区 |
| EVENT | 暴雨事件、节假日、MRT 故障、大型活动 |

**prompt 设计原则**
- 提供新加坡城市实体的 few-shot 示例
- 明确排除个人数据
- 要求提取数值属性（如 `daily_ridership`, `rainfall_mm`, `avg_resale_price`）
- 位置信息要求 `latitude` + `longitude` + `planning_area`

**输入**
- `data/processed/community_texts/*.txt`
- `configs/graphrag/entity_extraction_prompt.yaml`

**输出**
- `data/graphrag/input/entities.parquet`
- `logs/graphrag/entity_extraction_stats.json`（提取实体数、类型分布、LLM token 用量）

**失败处理**
- LLM API 调用失败 → 重试 3 次（指数退避），仍失败则降级到基于规则的结构化数据直接转换（跳过非结构化文本的实体提取）
- 实体数量 < 预期 50% → 发告警，检查 prompt 是否需要调优

#### 3.8.2 Relationship Extraction Agent

**职责**
调用 LLM 识别实体间的关系，构建图谱的边。

**预定义关系类型：**

| 关系类型 | 示例 | 来源 |
|---|---|---|
| `CONNECTS_TO` | MRT 站 A → MRT 站 B | LTA 线路数据（确定性，可规则生成） |
| `LOCATED_IN` | MRT 站 → 规划区域 | OneMap 空间 join（确定性） |
| `AFFECTS` | 暴雨事件 → MRT 站 | LLM 从文本描述中推断 |
| `NEAR` | POI → MRT 站 | 空间距离 < 500m（确定性） |
| `CORRELATES_WITH` | 降雨量 → 巴士客流 | LLM 从统计摘要中推断 |
| `PART_OF` | 巴士站 → 巴士线路 | LTA 数据（确定性） |

**设计原则**
- 确定性关系（从结构化数据可直接推导的）用规则生成，不浪费 LLM token
- LLM 只处理推断性关系（CORRELATES_WITH, AFFECTS）
- 每个 LLM 生成的关系必须附带 `evidence` 字段（引用来源文本/数据）

**输入**
- `data/processed/relationships/*.csv`（确定性关系）
- `data/processed/community_texts/*.txt`（文本描述）
- `data/graphrag/input/entities.parquet`

**输出**
- `data/graphrag/input/relationships.parquet`
- `logs/graphrag/relationship_stats.json`

#### 3.8.3 Community Detection Agent

**职责**
使用 Leiden 算法对构建好的图谱做社区检测，识别紧密耦合的实体群组（如 "CBD-交通-天气" 群组、"Jurong-工业-物流" 群组），为 GraphRAG 的 Global Search 提供分层摘要基础。

**参数**
- `max_cluster_size`: 20（防止单个社区过大，摘要质量下降）
- `resolution`: 1.0（默认）

**输出**
- `data/graphrag/output/communities.parquet`
- `reports/graphrag/community_map.png`（社区可视化图）

#### 3.8.4 Community Summarization Agent

**职责**
对每个社区的实体和关系生成自然语言摘要，供 Global Search 使用。

**prompt 要求**
- 摘要长度 200-500 tokens
- 必须包含：社区主题、关键实体（≤5个）、代表性关系、数值亮点
- 示例："This community centers on the CBD transportation hub during peak monsoon periods. Key entities include Orchard MRT (avg daily ridership: 85,000), Somerset MRT (avg daily ridership: 62,000), and Nov 2025 Northeast Monsoon events (max rainfall: 45.2mm/h). Rainfall events >30mm/h correlate with 25-40% MRT ridership increase in this cluster."

**输出**
- `data/graphrag/output/community_reports.parquet`

---

### 3.9 Neo4j Graph Storage Agent

**职责**
将 GraphRAG 索引结果写入 Neo4j 图数据库，支持 Cypher 查询和可视化。

**图 Schema**

```
节点标签:
- TransportNode {id, name, type[mrt/bus/taxi_stand], lat, lon, planning_area}
- WeatherStation {id, station_id, name, lat, lon}
- WeatherEvent {id, date, rainfall_mm, temperature_c, psi, category[normal/rain/heavy_rain/haze]}
- PlanningArea {id, name, region, population, area_sqkm}
- POI {id, name, category, lat, lon}
- HDBTown {id, name, region, avg_resale_price, rental_median}
- Holiday {id, date, name, type[public/school]}
- EntityCommunity {id, title, summary, level}

关系:
- (TransportNode)-[:CONNECTS_TO {line}]->(TransportNode)
- (TransportNode)-[:LOCATED_IN]->(PlanningArea)
- (WeatherEvent)-[:AFFECTS {confidence}]->(TransportNode)
- (WeatherStation)-[:RECORDS]->(WeatherEvent)
- (POI)-[:NEAR {distance_m}]->(TransportNode)
- (HDBTown)-[:LOCATED_IN]->(PlanningArea)
- (EntityCommunity)-[:CONTAINS]->(entity)
```

**索引策略**
- 在 `TransportNode.id`、`WeatherEvent.date`、`PlanningArea.name` 上建索引
- 在 `TransportNode` 的 `lat, lon` 上建空间索引（支持距离查询）

**输入**
- `data/graphrag/output/entities.parquet`
- `data/graphrag/output/relationships.parquet`
- `data/graphrag/output/communities.parquet`
- `data/graphrag/output/community_reports.parquet`

**输出**
- Neo4j 数据库（Docker: `neo4j:5-community`）
- `reports/graph/schema_validation.json`

**失败处理**
- Neo4j 连接失败 → 写本地 CSV 导出作为 fallback
- schema 不匹配 → 报告 mismatch 详情

---

### 3.10 Retrieval Agents

#### 3.10.1 Local Search Agent

**职责**
对用户问题做局部检索：提取问题中的实体提及 → 在知识图谱中定位相关实体 → 提取 1-2 跳邻居的子图 → 将子图上下文 + 原始文本注入 LLM prompt → 生成答案。

**检索流程**
```
用户问题: "下雨天 Orchard MRT 会有多拥挤？"
    │
    ▼
实体识别: "Orchard MRT", "下雨天"（雨事件）
    │
    ▼
Neo4j 子图检索: 
  MATCH (s:TransportNode {name:"Orchard MRT"})-[r]-(n)
  WHERE n:WeatherEvent OR n:WeatherStation
  RETURN s, r, n
    │
    ▼
上下文组装: 子图 + 原始文本片段 + 社区摘要
    │
    ▼
LLM 回答: 带溯源引用
```

**参数**
- `top_k_entities`: 5
- `max_hops`: 2
- `context_window`: 4000 tokens

#### 3.10.2 Global Search Agent

**职责**
对宏观/比较型问题（如"新加坡哪些区域的交通系统最容易受暴雨影响？"），遍历社区摘要做 Map-Reduce 式回答。

**检索流程**
```
用户问题 → 与所有社区摘要做语义匹配 → 选取 Top-5 社区
    → 每个社区生成部分回答（Map）
    → 汇总为最终回答（Reduce）
```

#### 3.10.3 Cypher Query Agent

**职责**
将自然语言中的结构化查询意图翻译为 Cypher 查询，直接在图数据库中执行。

**支持查询类型**
- "列出所有在 CBD 区域的 MRT 站" → `MATCH (s:TransportNode {type:"mrt"})-[:LOCATED_IN]->(:PlanningArea {name:"CBD"}) RETURN s`
- "2025 年降雨量 > 30mm/h 的天数" → `MATCH (e:WeatherEvent) WHERE e.rainfall_mm > 30 AND e.date STARTS WITH '2025' RETURN count(e)`
- "从 Jurong East 到 City Hall 经过哪些站" → 路径查询

**安全约束**
- 禁止 `DELETE`、`DETACH DELETE`、`DROP`
- 禁止写入操作
- 查询超时 5 秒
- 返回行数上限 1000

---

### 3.11 Answer Generation Agent

**职责**
基于检索结果（子图 + 文本上下文）调用 LLM 生成最终答案。

**prompt 模板（核心）**

```
You are UrbanGraph-SG, an AI assistant grounded in Singapore's urban knowledge graph.

CONTEXT (from knowledge graph):
{retrieved_subgraph}

SOURCES:
{source_citations}

RULES:
1. Answer ONLY using information present in the CONTEXT above.
2. If the CONTEXT does not contain enough information to answer, say "I don't have enough data to answer this question. The knowledge graph currently covers: [list available domains]."
3. Every factual claim MUST cite its source using [Source: entity_name, dataset, date_range] format.
4. For numerical claims, include the time range the data covers.
5. If the answer involves comparison, provide numbers for both sides.
6. Do NOT generate speculative answers beyond the evidence.
7. Keep answers concise but complete. Target 100-300 words unless the question requires more detail.

QUESTION: {user_query}

ANSWER:
```

**输出字段**
- `answer_text`: 最终回答
- `sources_used`: 引用的实体 + 数据集 + 时间范围列表
- `confidence`: HIGH / MEDIUM / LOW（取决于检索覆盖度）
- `graph_entities`: 答案中涉及的实体 ID 列表（用于前端高亮）
- `llm_model`: 使用的 LLM 模型名
- `tokens_used`: 消耗的 token 数

**失败处理**
- LLM 调用失败 → 重试 1 次，仍失败返回 "Service temporarily unavailable"
- 检索结果为空 → 返回 "I don't know" 消息 + 知识图谱覆盖范围说明
- Context 超过 token 限制 → 自动截断，优先保留与问题最相关的部分

---

### 3.12 Streamlit UI Agent

**职责**
提供对话界面 + 知识图谱可视化。

**界面布局（三栏）**

```
┌──────────────┬────────────────────────┬──────────────┐
│  📁 预设问题   │                        │  🗺️ 知识图谱   │
│              │   💬 对话区域            │  可视化       │
│  ▸ CBD暴雨     │                        │              │
│    对MRT影响  │   User: 下雨天 Orchard  │  [PyVis 子图] │
│              │   MRT 有多拥挤？         │              │
│  ▸ Jurong    │                        │  节点: 12     │
│    房价趋势   │   System: 根据 LTA 和   │  关系: 18     │
│              │   NEA 数据...            │              │
│  ▸ 跨区通勤    │   [Source: LTA, 2025]  │              │
│    模式      │                        │              │
│              │                        │              │
└──────────────┴────────────────────────┴──────────────┘
```

**预设问题库（至少 15 个）**

| 类别 | 示例问题 |
|---|---|
| 交通×天气 | "暴雨天 CBD 区域的 MRT 客流比平时增加多少？巴士客流有何变化？" |
| 空间分析 | "哪些 MRT 站周边 500m 内 HDB 转售价格最高？与交通便利性有何关系？" |
| 时间模式 | "学校假期期间，Orchard Road 区域的交通模式与工作日有何不同？" |
| 跨域关联 | "Jurong East 的工业用地比例与公交车次频率之间是否存在关联？" |
| 历史对比 | "2025 年东北季风期间，哪些区域的 MRT 客流波动最大？" |

**交互功能**
- 用户输入自然语言问题
- 回答渲染为 Markdown，来源引用可点击展开
- 右侧子图实时联动：展示当前答案涉及的实体和关系
- 点击图中节点可查看实体详情卡片
- 支持追问（上下文保持）

---

### 3.13 Evaluation Agent

**职责**
评估系统回答质量。与 UrbanFlow-AU 不同，这里不评分数值预测误差，而是评估语义质量。

**评估维度**

| 维度 | 指标 | 评估方式 |
|---|---|---|
| 答案准确性 | 事实正确率 | 人工抽样（50 题）× 双重标注 |
| 溯源完整性 | 每回答引用数 ≥ 1，来源可验证 | 自动检查 |
| 覆盖度 | 预设问题集中可回答的比例 | 自动统计 |
| 拒答诚实性 | "I don't know" 率（不应答的问题不编造） | 人工评估 |
| 延迟 | 端到端响应时间（P50 / P95） | 自动记录 |
| Token 效率 | 每回答平均 token 消耗 | 自动记录 |

**触发条件**
- 每次 GraphRAG 重新 index 后
- 预设问题库更新后
- 手动 `make evaluate`

**输出**
- `reports/evaluation/eval_report.md`
- `reports/evaluation/answer_samples.json`

**失败处理**
- 准确率 < 70% → 需要调优 prompt 或扩展知识图谱
- 覆盖率 < 60% → 需要扩充数据源

---

### 3.14 Deployment Agent

**职责**
Docker Compose 一键启动：Streamlit + Neo4j + GraphRAG API。

**Docker Compose 服务**

```yaml
services:
  neo4j:
    image: neo4j:5-community
    environment:
      NEO4J_AUTH: neo4j/password
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs

  streamlit:
    build:
      context: .
      dockerfile: docker/ui.Dockerfile
    ports:
      - "8502:8502"
    environment:
      NEO4J_URI: bolt://neo4j:7687
      LLM_API_KEY: ${LLM_API_KEY}
      LLM_MODEL: ${LLM_MODEL:-deepseek-chat}
    depends_on:
      - neo4j
```

**LLM Provider 支持**
| Provider | 模型推荐 | 用途 |
|---|---|---|
| DeepSeek | deepseek-chat | 低成本主力模型 |
| OpenAI | gpt-4o-mini | 高质量需求 |
| Ollama (local) | qwen2.5:7b / llama3.1:8b | 零成本开发测试 |

---

## §4 完整命令流程

```bash
# === 环境初始化 ===
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/base.txt

# === 配置 ===
cp .env.example .env  # 编辑填入 LTA_ACCOUNT_KEY, LLM_API_KEY

# === 数据采集（全部 6 个来源） ===
make ingest-all
# 等价于：
# python -m src.ingestion.lta --data_types all
# python -m src.ingestion.nea
# python -m src.ingestion.singstat
# python -m src.ingestion.ura_onemap
# python -m src.ingestion.hdb
# python -m src.ingestion.calendar

# === 数据处理 + 实体对齐 ===
make process
# python -m src.processing.run_all

# === GraphRAG 索引管道 ===
make index
# 内部执行:
# 1. python -m src.graphrag.extract_entities
# 2. python -m src.graphrag.extract_relationships
# 3. python -m src.graphrag.detect_communities
# 4. python -m src.graphrag.summarize_communities

# === 写入 Neo4j ===
make load-graph
# python -m src.graph.write_to_neo4j

# === 启动服务 ===
make up
# docker compose up --build

# === 访问 ===
# Streamlit: http://localhost:8502
# Neo4j Browser: http://localhost:7474 (neo4j/password)
```

---

## §5 目录结构

```text
urbangraph-sg/
├── README.md
├── agent.md
├── UrbanGraph-SG-report.md
├── LICENSE
├── .gitignore
├── .env.example
├── Makefile
├── pyproject.toml
├── requirements/
│   ├── base.txt          # 核心依赖
│   ├── graphrag.txt      # GraphRAG + Neo4j 相关
│   ├── llm.txt           # LLM SDK (openai / langchain / deepseek)
│   └── dev.txt           # pytest, ruff, pre-commit
├── configs/
│   ├── data_sources.yaml           # 数据源配置
│   ├── schema_mapping.yaml         # 字段映射规则
│   ├── entity_types.yaml           # 知识图谱实体类型定义
│   ├── relationship_types.yaml     # 关系类型定义
│   ├── graphrag_config.yaml        # GraphRAG 管道配置
│   ├── search_config.yaml          # 检索参数
│   └── preset_questions.yaml       # 预设问题库
├── data/
│   ├── raw/
│   │   ├── lta/
│   │   ├── nea/
│   │   ├── singstat/
│   │   ├── ura/
│   │   ├── onemap/
│   │   ├── hdb/
│   │   └── calendar/
│   ├── manifests/
│   ├── processed/
│   │   ├── entities/
│   │   ├── relationships/
│   │   └── community_texts/
│   └── graphrag/
│       ├── input/
│       └── output/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── lta.py
│   │   ├── nea.py
│   │   ├── singstat.py
│   │   ├── ura_onemap.py
│   │   ├── hdb.py
│   │   └── calendar.py
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── time_normalizer.py
│   │   ├── spatial_validator.py
│   │   ├── entity_resolution.py
│   │   ├── graphrag_formatter.py
│   │   └── quality_reporter.py
│   ├── graphrag/
│   │   ├── __init__.py
│   │   ├── entity_extractor.py
│   │   ├── relationship_extractor.py
│   │   ├── community_detector.py
│   │   ├── summarizer.py
│   │   └── prompts/
│   │       ├── entity_extraction.yaml
│   │       ├── relationship_extraction.yaml
│   │       └── summarization.yaml
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── neo4j_client.py
│   │   ├── schema.py
│   │   └── write_to_neo4j.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── local_search.py
│   │   ├── global_search.py
│   │   └── cypher_agent.py
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── answer_generator.py
│   │   └── prompts/
│   │       └── answer_prompt.yaml
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── streamlit_app.py
│   │   └── components/
│   │       ├── chat_panel.py
│   │       ├── graph_viz.py
│   │       └── preset_questions.py
│   └── evaluation/
│       ├── __init__.py
│       ├── run_eval.py
│       └── metrics.py
├── tests/
│   ├── unit/
│   │   ├── test_entity_resolution.py
│   │   ├── test_time_normalizer.py
│   │   ├── test_cypher_generation.py
│   │   └── test_answer_prompt.py
│   ├── integration/
│   │   ├── test_ingestion_pipeline.py
│   │   ├── test_graphrag_pipeline.py
│   │   └── test_retrieval_flow.py
│   └── fixtures/
│       ├── sample_entities.csv
│       ├── sample_relationships.csv
│       └── sample_questions.json
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_graph_schema_design.ipynb
│   ├── 03_entity_extraction_tuning.ipynb
│   └── 04_evaluation_analysis.ipynb
├── reports/
│   ├── data_quality/
│   ├── graph/
│   ├── evaluation/
│   └── figures/
├── docker/
│   ├── ui.Dockerfile
│   └── neo4j.Dockerfile
├── docker-compose.yml
└── .github/
    └── workflows/
        ├── ci.yml          # lint + unit tests
        └── index-check.yml # GraphRAG 索引冒烟测试
```

---

## §6 测试策略

### 6.1 单元测试
- `test_entity_resolution.py`：同名 MRT 站在不同数据源的对齐逻辑
- `test_time_normalizer.py`：时间戳转换正确性
- `test_cypher_generation.py`：NL→Cypher 转换覆盖主要查询类型
- `test_answer_prompt.py`：prompt 模板正确填充、上下文截断逻辑

### 6.2 集成测试
- `test_ingestion_pipeline.py`：端到端 mock API → 数据落盘
- `test_graphrag_pipeline.py`：小样本数据走完 extract → summarize
- `test_retrieval_flow.py`：问题 → 检索 → 回答 全链路

### 6.3 CI
- GitHub Actions: lint (ruff) + unit tests + Neo4j smoke test（使用 Docker service）
- 不跑完整 GraphRAG 索引（token 成本问题），only dry run 验证管道可启动

---

## §7 资源预算

| 资源 | MVP | 说明 |
|---|---|---|
| LLM API 费用 | S$15-50 | 主要在 GraphRAG 索引阶段，之后增量更新费用低 |
| Neo4j 内存 | 2-4 GB | Docker 本地运行 |
| 磁盘 | 5-10 GB | 原始数据缓存 + Neo4j data |
| CPU | 4-8 cores | 数据处理和索引 |
| 时间 | 6-8 周 | 见 §9 时间线 |

**省钱策略：**
- 开发阶段用 Ollama 本地模型（qwen2.5:7b）做实体提取 → 零 API 费用
- 只有最终 demo 和报告截图时切换 OpenAI/DeepSeek API 提高质量
- Neo4j Community Edition 完全免费

---

## §8 安全与伦理

### 8.1 数据合规
- 本项目为学术研究 + 公开技术演示，不用于商业决策
- 所有数据来自新加坡政府公开数据（data.gov.sg 的 Open Data License）
- LTA DataMall 数据按 Terms of Use 使用
- 不采集、不存储个人身份信息
- 不爬取非公开页面

### 8.2 LLM 使用规范
- 所有 LLM 输出必须在 README 中声明为"AI 辅助生成"
- 不对 LLM 生成的内容宣称 100% 准确性
- 系统回答明确区分"来自知识图谱的结构化事实"和"LLM 的语言组织"

### 8.3 仓库安全
- `.env` 不提交
- API Key 不硬编码
- Neo4j 默认密码在文档中强调仅供本地开发
- Docker Compose 不在 `ports` 中暴露 Neo4j 到 `0.0.0.0`

---

## §9 8 周时间线

| 周次 | 主题 | 交付物 |
|---|---|---|
| 1 | 数据通路 + 实体模型设计 | 6 个 ingestion agent 可运行，实体类型 schema 定稿 |
| 2 | 数据处理 + 实体对齐 | Processing agent 完成，实体对齐规则通过测试 |
| 3 | GraphRAG 管道搭建 | Entity & relationship extraction agent 跑通（本地 LLM） |
| 4 | Community detection + summarization | 完整 GraphRAG 索引可执行，社区摘要质量达标 |
| 5 | Neo4j + 检索引擎 | 图写入 Neo4j，Local/Global/Cypher 三种检索可用 |
| 6 | Answer generation + Streamlit | 对话界面 + 图谱可视化联调，预设问题库完成 |
| 7 | 评估 + 迭代调优 | 50 题人工评估，prompt 调优，提高准确率和覆盖率 |
| 8 | 文档 + 演示 | README、demo 录屏、简历条目、技术报告 |

---

## §10 UrbanGraph-SG ↔ UrbanFlow-AU 对齐表

| 维度 | UrbanFlow-AU | UrbanGraph-SG |
|---|---|---|
| 项目类型 | 数值预测平台 | 知识问答系统 |
| 核心范式 | 监督学习 | 检索增强生成 |
| 数据形态 | 结构化表格 | 知识图谱 |
| 模型训练 | ✅ 必需 | ❌ 零样本（不训模型） |
| 回测 | ✅ rolling-origin | N/A |
| LLM 使用 | ❌ | ✅ 核心组件 |
| 数据库 | PostgreSQL | Neo4j |
| 实验追踪 | MLflow | 自定义 eval + 日志 |
| 监控 | Evidently | 人工评估 + 覆盖度统计 |
| Dashboard | Streamlit 运营仪表盘 | Streamlit 对话界面 + PyVis |
| 工程栈 | Prefect, Pandera, Docker Compose, pytest, ruff, GitHub Actions | **相同** → 复用工程经验 |
| 目标学校 | 澳洲 (Monash, Sydney, UTS) | 新加坡 (NUS, NTU, SMU, SUTD) |
