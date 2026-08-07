# UrbanGraph-SG-Report.md

## 项目章程、规则与验收标准

> 本文档是 UrbanGraph-SG 的**宪法**。当实现细节与本文档冲突时，以本文档为准。

---

## §A 项目定义

### A.1 一句话定义

**UrbanGraph-SG** is a GraphRAG-powered urban knowledge navigator for Singapore's Smart Nation. It ingests multi-source open data (LTA, NEA, SingStat, URA, OneMap, HDB), constructs a unified knowledge graph, and enables natural-language question answering with source-attributed, evidence-bound responses and interactive graph visualization.

### A.2 核心研究问题

| 层级 | 问题 |
|---|---|
| **知识表示层** | 如何将新加坡异构开放数据（交通、天气、人口、房价、空间规划）统一建模为可查询、可推理的城市知识图谱？ |
| **检索增强层** | GraphRAG 的 Local Search、Global Search 和 Cypher 查询三种检索范式，在回答城市领域问题时各自的适用场景和局限是什么？ |
| **可信度层** | 如何设计 prompt 和溯源机制，确保 LLM 生成的答案不会"编造"城市统计数据？ |

### A.3 目标用户画像

| 用户 | 典型问题 |
|---|---|
| 城市研究者 | "对新加坡 CBD 在暴雨天的交通模式进行系统分析" |
| 数据爱好者 | "哪些 MRT 站周边房价涨幅最高，与什么因素相关？" |
| 申请审核者（招生官） | "这个系统怎么工作的？能演示一个完整的问题-回答-溯源链路吗？" |
| 未来用户（你本人做 FYP 时） | "GraphRAG 在处理新加坡城市数据时有哪些 engineering challenges？" |

---

## §B 数据规则

### B.1 必须使用的数据源

| # | 来源 | API/格式 | 必须字段 |
|---|---|---|---|
| 1 | LTA DataMall | REST API | MRT 站点坐标、线路连接、巴士站点坐标、巴士线路 |
| 2 | data.gov.sg (NEA) | REST API | 日降雨量、温度、湿度、PSI |
| 3 | data.gov.sg (SingStat) | REST API / CSV | 规划区域人口、家庭收入 |
| 4 | OneMap API | REST API | POI 坐标+类别、规划区域多边形 |
| 5 | data.gov.sg (HDB) | REST API / CSV | 组屋转售价格、租金中位数（按城镇） |

### B.2 推荐增强数据源（MVP 之后）

- LTA DataMall: 巴士实时到站、出租车可用性、交通速度
- data.gov.sg: 学校假期日历
- URA: 用地规划数据

### B.3 数据范围

- MVP 使用最近 2 个完整日历年的数据（如 2025-2026）
- 所有数据按时间范围采样，不追求全历史
- 知识图谱实体数量：MVP 目标 3000-5000 个实体节点
- 关系数量：MVP 目标 8000-15000 条关系边

### B.4 数据质量红线

以下任一条件触发，GraphRAG 索引**不得执行**：

1. 任一必须数据源完全不可用（API 连续失败 3 次以上）
2. MRT 站坐标缺失率 > 5%
3. 气象数据缺失天数超过 30 天/年
4. 实体对齐后仍有 > 10% 的实体没有 `planning_area` 属性

### B.5 数据禁止事项

- 不得采集个人层面的数据（单个乘客、单个家庭）
- 不得使用需要付费的商业数据
- 不得将实时 API 数据用于任何商业目的
- 不得在 GitHub 公开 Neo4j 数据库文件（可公开 schema 和示例子图）

---

## §C 知识图谱规则

### C.1 实体类型定义

| 类型 | 标签 | 唯一标识 | 必有属性 | 可选属性 |
|---|---|---|---|---|
| MRT 站 | `TransportNode` | `lta-mrt-{station_code}` | name, lat, lon, planning_area | daily_ridership, opening_year, lines[] |
| 巴士站 | `TransportNode` | `lta-bus-{bus_stop_code}` | name, lat, lon, planning_area | services[] |
| NEA 气象站 | `WeatherStation` | `nea-station-{station_id}` | name, lat, lon | - |
| 天气事件（日） | `WeatherEvent` | `nea-event-{date}` | date, rainfall_mm, temperature_max, temperature_min | psi, humidity |
| 规划区域 | `PlanningArea` | `ura-area-{area_name}` | name, region, population | area_sqkm |
| POI | `POI` | `onemap-poi-{uuid}` | name, category, lat, lon | - |
| HDB 城镇 | `HDBTown` | `hdb-town-{town_name}` | name, region | avg_resale_price, rental_median, flat_types{} |
| 公共假期 | `Holiday` | `holiday-{date}` | date, name, type | - |
| 知识社区 | `EntityCommunity` | `community-{id}` | title, summary, level | member_count |

### C.2 关系类型定义

| 关系 | 起点 → 终点 | 生成方式 | 置信度要求 |
|---|---|---|---|
| `CONNECTS_TO` | TransportNode → TransportNode | 规则（LTA 线路数据） | 100% |
| `LOCATED_IN` | *Node → PlanningArea | 规则（空间 join） | 100% |
| `NEAR` | POI → TransportNode | 规则（距离 <500m） | 100% |
| `PART_OF` | 巴士站 → 巴士线路 | 规则（LTA 服务数据） | 100% |
| `RECORDS` | WeatherStation → WeatherEvent | 规则（NEA 站点映射） | 100% |
| `AFFECTS` | WeatherEvent → TransportNode | LLM 推断 | ≥80%（人工标注验证） |
| `CORRELATES_WITH` | WeatherEvent → HDBTown | LLM 推断 | ≥70%（人工标注验证） |
| `CONTAINS` | EntityCommunity → *Node | 算法（Leiden detection） | 取决于社区检测参数 |

**关键规则：**
- 确定性关系（CONNECTS_TO, LOCATED_IN, NEAR, PART_OF, RECORDS）**必须 100% 准确**——这些是可信度的基石
- 推断性关系（AFFECTS, CORRELATES_WITH）必须在 `properties` 中附带 `evidence` 字段
- 推断性关系在 Neo4j 中以虚线/灰边显示，在 UI 中明确标注"推断关系"

### C.3 实体对齐规则

同一物理实体在不同数据源中有不同名称时：

| 优先级 | 规则 | 示例 |
|---|---|---|
| 1 | 精确匹配 | "Orchard" == "Orchard" → 合并 |
| 2 | 标准化后匹配 | "Orchard MRT Station" → normalize → "orchard mrt" → 匹配 |
| 3 | Levenshtein < 3 | "Dhoby Ghaut" vs "Dhoby Ghaut Stn" → 距离=2 → 合并 |
| 4 | 空间邻近 + 名称相似 | 坐标差 < 100m + 名称相似度 > 0.85 → 人工审核后合并 |

**禁止的合并：**
- 仅凭坐标相近但名称完全不同（如 "Raffles Place" 和 "Telok Ayer"，距离 < 500m 但是不同的站）
- 类型不同的实体（不要把 MRT 站和巴士站合并，即使位置重合）

---

## §D GraphRAG 管道规则

### D.1 Entity Extraction Prompt 规则

Prompt 设计必须满足：

1. **领域适配**：必须包含 ≥5 个新加坡城市领域的 few-shot 示例
2. **类型约束**：强制输出预定义的实体类型，不允许 LLM 自创类型
3. **数值提取**：如果实体描述中包含数值（客流量、降雨量、房价），必须作为属性提取
4. **位置要求**：如果实体有地理位置，必须输出 `latitude` + `longitude` + `planning_area`
5. **去重指令**：prompt 中明确要求对明显重复的实体做聚合（如 "Orchard MRT" 和 "Orchard Station" 是同一实体）

### D.2 Relationship Extraction Prompt 规则

1. **确定性关系不调用 LLM**：`CONNECTS_TO`, `LOCATED_IN`, `NEAR`, `PART_OF`, `RECORDS` 用代码生成
2. **LLM 只推断** `AFFECTS` 和 `CORRELATES_WITH`
3. **每个推断关系必须有 `evidence` 字段**：引用原始文本/数据中的具体数字
4. **关系方向性必须正确**：不创造无向关系
5. **禁止创造不在实体列表中的实体**：所有关系的起点和终点必须存在于已提取的实体列表中

### D.3 Community Summarization 规则

1. 每个社区摘要 200-500 tokens
2. 必须包含社区主题（一句话）
3. 必须列出 ≤5 个代表性实体
4. 必须包含 ≥1 个数值亮点（如"该区域平均房价 S$550K"）

### D.4 GraphRAG 索引幂等性

- 相同数据和配置 → 执行两次 index → 实体和关系不应有结构性变化
- 允许 LLM 引入的非确定性文本差异（如摘要措辞不同），但不允许实体 ID 或关系结构变化
- 每次 index 记录 `config_hash`，用于追踪配置变更对应的图谱差异

---

## §E 检索与回答规则

### E.1 三种检索模式的使用规则

| 模式 | 适用问题类型 | 触发条件 | 不适用问题类型 |
|---|---|---|---|
| **Local Search** | 单实体/小范围的具体问题 | 问题中提到明确的实体名称 | 跨区域比较、宏观趋势 |
| **Global Search** | 宏观比较/总结性问题 | 问题涉及多个区域或抽象概念 | 具体数值查询 |
| **Cypher Query** | 精确结构化查询 | 问题可明确转为图查询模式 | 模糊语义理解 |

### E.2 Answer Generation 硬性规则

**必须遵守：**

1. **溯源要求**：每个事实性陈述必须有 [Source: ...] 标记
2. **不知道就说不知道**：Context 不足以回答时，必须说 "I don't have enough data"，不得编造
3. **数值必须有上下文**：不能只说"客流增加 30%"，必须说"2025 年 CBD 区域暴雨天 MRT 客流比晴天增加 25-35%（来源：LTA, NEA, 2025 年 1-12 月数据）"
4. **时间范围标注**：所有统计必须带时间范围
5. **不可以"可能"代替"不知道"**：不确定的信息不能说"可能 xxx"，应该说"根据现有数据，这个问题我们暂时无法给出可靠答案"

**禁止：**

1. 禁止编造不存在的 MRT 站、区域名、统计数字
2. 禁止在没有天气数据的情况下分析天气影响
3. 禁止用 LLM 的预训练知识"脑补"新加坡数据（除非该知识在 Context 中已确认）
4. 禁止回答与城市数据无关的闲聊问题（拒绝模板："I'm specialized in Singapore urban data. I can help you with questions about transport, weather, housing, and their interconnections."）

### E.3 回答置信度定义

| 置信度 | 条件 | UI 表现 |
|---|---|---|
| HIGH | 所有引用来自精确匹配的实体和确定性关系 | 绿色标记 |
| MEDIUM | 部分引用来自推断性关系 | 黄色标记 |
| LOW | Context 稀疏，回答依赖少量证据 | 橙色标记 |
| UNKNOWN | 无相关 Context | "I don't know" + 覆盖范围说明 |

---

## §F UI/UX 规则

### F.1 Streamlit 界面规则

1. **第一屏必须有预设问题**：用户可能不知道该问什么，左侧栏展示示例问题
2. **回答区和图谱区同步**：点击回答中的实体名 → 右侧图谱高亮对应节点
3. **图谱不要太大**：单次展示 ≤50 个节点，超过则按社区分组折叠
4. **来源可展开**：每个 [Source: ...] 标记可点击展开显示完整引用（数据集名、时间范围、原始值）
5. **不支持的功能明确告知**：不假装能回答所有问题

### F.2 预设问题设计规则

预设问题必须：
1. 覆盖交通、天气、房价三大领域
2. 包含具体实体名（可直接测试 Local Search）
3. 包含比较/总结性问题（测试 Global Search）
4. 包含结构化查询（测试 Cypher Query）
5. 每个问题附带一个"期待的回答特征"用于评估

---

## §G 验收标准

### G.1 MVP 验收标准

项目只有在**以下全部条件**同时满足时才算 MVP 完成：

- [ ] 6 个 ingestion agent 全部可独立运行，输出格式合格
- [ ] Processing agent 完成实体对齐：同一 MRT 站在不同源的名称被正确合并
- [ ] GraphRAG 索引管道在本地 LLM（Ollama）上可从头跑到尾，不报错
- [ ] 知识图谱写入 Neo4j 后，Cypher 查询 `MATCH (n) RETURN count(n)` 返回 >1000 个节点
- [ ] Local Search 对包含明确实体名的问题返回有来源引用的回答
- [ ] Global Search 对"新加坡哪些区域..."类问题返回基于社区摘要的回答
- [ ] Cypher Query Agent 能正确翻译 ≥5 种预设结构化查询
- [ ] Streamlit 三栏界面可展示：预设问题 → 回答 → 图谱联动
- [ ] Docker Compose 一键启动 Neo4j + Streamlit
- [ ] pytest 全部通过，ruff 无报错
- [ ] README 包含：架构图、截图（≥3 张）、demo 录屏链接、预设问题演示
- [ ] 知识图谱 schema 在 README 中图表化展示
- [ ] 仓库不包含 API Key、Neo4j 密码、大型数据文件

### G.2 质量目标（非必须但追求）

| 指标 | 目标值 |
|---|---|
| 预设问题可回答率 | ≥ 80% |
| 答案有至少 1 个来源引用 | 100% |
| "I don't know" 率（不可回答问题时） | 100%（不编造） |
| LLM 编造事实率（人工抽样 50 题） | < 10% |
| 端到端响应延迟 P95 | < 15s |
| 实体提取 prompt 在 5 次 run 中的实体数量标准差 | < 10% |

### G.3 不可接受的交付

- 知识图谱只有"玩具级"规模（<500 个实体）
- 回答质量完全依赖 LLM 预训练知识（即关了 Neo4j 系统还能给出类似质量的回答 → 说明 GraphRAG 没起作用）
- README 只有文字没有截图
- Docker Compose 启动失败

---

## §H 申请材料规则

### H.1 个人陈述叙事锚点

在申请 NUS/NTU/SMU 的 CS/AI 项目时，UrbanGraph-SG 应该承担以下叙事角色：

> UrbanGraph-SG demonstrates my ability to bridge knowledge graphs, large language models, and real-world urban data into a coherent AI system. Unlike my previous MLOps project focused on numerical forecasting, this project tackles the challenge of semantic understanding and evidence-grounded generation—skills central to modern AI research and directly aligned with my Final Year Project on KG+LLM Question Answering.

### H.2 简历条目

**中文**

> **UrbanGraph-SG：基于 GraphRAG 的新加坡城市知识图谱问答系统**
> 整合 LTA、NEA、SingStat、OneMap 等多源新加坡开放数据，构建覆盖交通、天气、房价领域、包含 3000+ 实体与 8000+ 关系的城市知识图谱；基于微软 GraphRAG 框架实现实体提取、社区检测与多层次摘要生成，并通过 Neo4j 图数据库支持 Local Search、Global Search 与 Cypher 查询三种检索范式。
> 设计带溯源的 LLM Answer Generation Pipeline 确保回答可信度，搭建 Streamlit 对话界面与 PyVis 知识图谱可视化，使用 Docker Compose 实现一键部署。

**English**

> **UrbanGraph-SG: GraphRAG-Powered Urban Knowledge Navigator for Singapore**
> Built a domain knowledge graph integrating LTA, NEA, SingStat, and OneMap open data, covering 3,000+ entities and 8,000+ relationships across transport, weather, and housing. Implemented GraphRAG indexing pipeline (entity/relationship extraction, community detection, hierarchical summarization) and deployed on Neo4j with three retrieval modes. Designed a provenance-grounded answer generation system with confidence labeling, and delivered an interactive Streamlit interface with real-time graph visualization.

### H.3 README 必须包含的视觉元素

1. 系统架构图（Mermaid 或手绘→数字化）
2. 知识图谱 schema 可视化（节点-关系图）
3. 问答 demo 截图（≥3 组 Q&A）
4. Streamlit 界面全屏截图
5. PyVis 图谱交互截图
6. Docker Compose 启动日志截图
7. （加分）1-2 分钟 demo 录屏 GIF 或 YouTube 链接

### H.4 与其他项目的叙事关联

在面试或 PS 中，应该自然串联：

> "从 UrbanFlow-AU 的数值预测能力，到 UrbanGraph-SG 的语义理解与知识推理能力，再到我的 Final Year Project 的学术深度，这三者构成了我从工程到研究、从数据分析到知识智能的完整技术成长路径。"

---

## §I 规则优先级与修订

### I.1 冲突解决顺序

当不同文件之间存在冲突时，优先级为：

1. **本文档（UrbanGraph-SG-report.md）** ← 最高优先级
2. `agent.md` ← Agent 规范
3. `README.md` ← 对外展示
4. 代码实现 ← 必须向上对齐

### I.2 修订流程

- 本文档的重大修改必须伴随一次 Git commit，commit message 中注明修改的 § 编号
- 不接受"先改代码，后补文档"——约束条件必须先更新

### I.3 文档生命周期

- 本文档在项目开始时创建，MVP 完成后锁定
- 锁定后只能追加附录（§J+），不得修改已有规则
- 如果发现规则不切实际需要修改，必须在 commit message 中解释为何原规则不可行

---

## §J 附录：预设问题库

以下是 MVP 必须内置的预设问题列表。每个问题附带评估标准。

| # | 问题 | 领域 | 检索模式 | 评估标准 |
|---|---|---|---|---|
| 1 | "What are all MRT stations in the CBD planning area?" | 交通 | Cypher | 返回完整列表 + 数量准确 |
| 2 | "How does heavy rainfall affect MRT ridership in Orchard area?" | 交通×天气 | Local | 有具体数值 + 有来源引用 |
| 3 | "Which planning area has the highest HDB resale price, and how is it connected to MRT accessibility?" | 房价×交通 | Global | 排名清晰 + 有因果推断 |
| 4 | "During which months does Singapore experience the most rainfall, and how does this correlate with bus ridership changes?" | 天气×交通 | Global | 有时间序列 + 有相关度 |
| 5 | "Find all POIs within 500m of Jurong East MRT station" | 空间 | Cypher | 返回完整列表 |
| 6 | "Compare the transport connectivity of Tampines versus Jurong East" | 交通 | Global | 有对比数据 |
| 7 | "On public holidays, do MRT stations near Orchard Road see higher or lower ridership?" | 交通×日历 | Local | 有方向性结论 |
| 8 | "What is the average HDB resale price in Punggol, and how has it changed?" | 房价 | Cypher | 有数值 + 有时间范围 |
| 9 | "Which MRT lines pass through Bishan station?" | 交通 | Cypher | 列出所有线路 |
| 10 | "Is there a relationship between high PSI (haze) days and taxi demand?" | 天气×交通 | Local | 有证据支持或无证据的诚实声明 |
| 11 | "List all bus stops along Orchard Road" | 交通 | Cypher | 返回完整列表 |
| 12 | "Which areas in Singapore have the lowest population but the highest MRT station density?" | 人口×交通 | Global | 有对比排名 |
| 13 | "How many MRT stations are there in total in the knowledge graph?" | 元数据 | Cypher | 精确数字 |
| 14 | "During school holidays, which MRT stations see the biggest change in passenger patterns?" | 交通×日历 | Global | 有具体站名 + 有变化幅度 |
| 15 | "If I want to live near an MRT station in a planning area with HDB resale prices below S$500K, which areas should I consider?" | 房价×交通 | Cypher | 返回符合条件的区域列表 |

**评估方法：**
- 每个问题人工检验回答质量和溯源完整性
- 评分：✅ 完全满足 / ⚠️ 部分满足 / ❌ 未满足
- MVP 目标：≥12/15 问题达到 ✅ 或 ⚠️
