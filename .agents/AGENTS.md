# 多Agent协作系统 - 项目架构

## 系统概述

本项目采用多Agent协作架构，由 **Orchestrator Agent** 统一协调多个专业Agent并行工作。

```
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestrator Agent (主控)                     │
│  - 项目整体规划与管理                                            │
│  - Agent任务分配与调度                                           │
│  - 进度跟踪与质量控制                                            │
│  - 关键决策与补丁修复                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Architect    │    │   Developer   │    │   Reviewer    │
│    Agent      │    │    Agent      │    │    Agent      │
│  (架构设计)    │    │  (代码实现)    │    │  (代码审查)    │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   Report Agent  │
                    │   (进度汇报)     │
                    └─────────────────┘
```

## Agent角色定义

### 1. Orchestrator Agent (主控Agent)

**职责**:
- 项目整体规划与里程碑管理
- 任务分解与Agent分配
- 进度跟踪与风险管理
- 跨Agent协调与冲突解决
- 关键代码审查与补丁修复

**工作模式**:
- 接收用户需求，进行任务拆解
- 为其他Agent生成详细Prompt
- 监控各Agent工作进度
- 整合各Agent输出，确保一致性

---

### 2. Architect Agent (架构Agent)

**职责**:
- 系统架构设计
- 模块接口定义
- 技术选型决策
- 设计文档编写

**输出**:
- `design_*.md` - 设计文档
- `interface_*.py` - 接口定义
- `architecture_report.md` - 架构报告

---

### 3. Developer Agent (开发Agent)

**职责**:
- 代码实现
- 单元测试编写
- 文档注释编写
- Bug修复

**输出**:
- `*.py` - 实现代码
- `test_*.py` - 测试代码
- `dev_report.md` - 开发进度报告

---

### 4. Reviewer Agent (审查Agent)

**职责**:
- 代码质量审查
- 性能分析
- 安全审计
- 最佳实践检查

**输出**:
- `review_*.md` - 审查报告
- `suggestions.md` - 改进建议

---

### 5. Report Agent (汇报Agent)

**职责**:
- 汇总各Agent进度
- 生成项目状态报告
- 识别阻塞问题
- 提出资源调配建议

**输出**:
- `daily_report.md` - 日报
- `milestone_report.md` - 里程碑报告

---

## 工作流程

### Phase 1: 任务初始化

1. **Orchestrator** 接收用户需求
2. **Orchestrator** 创建任务分解文档 `tasks/TASK-{id}.md`
3. **Orchestrator** 为各Agent生成专用Prompt

### Phase 2: 并行执行

1. **Architect** 进行系统设计
2. **Developer** 进行代码实现
3. **Reviewer** 进行代码审查
4. 各Agent定期向 **Report Agent** 提交进度

### Phase 3: 整合与交付

1. **Orchestrator** 审查各Agent输出
2. **Orchestrator** 解决冲突和不一致
3. **Orchestrator** 应用关键补丁
4. **Report Agent** 生成最终报告

---

## 文档规范

### Agent Prompt 文件

位置: `.agents/prompts/{agent_name}_prompt.md`

内容结构:
```markdown
# {Agent名称} Prompt

## 角色定义
...

## 能力要求
...

## 任务范围
...

## 工作流程
...

## 输出规范
...

## 汇报格式
...
```

### Agent 报告文件

位置: `.agents/reports/{agent_name}/{timestamp}_report.md`

内容结构:
```markdown
# {Agent名称} 工作报告

## 任务ID
...

## 执行摘要
...

## 详细工作
...

## 遇到的问题
...

## 需要协调的事项
...

## 下一步计划
...
```

### 任务追踪文件

位置: `.agents/tasks/TASK-{id}.md`

内容结构:
```markdown
# Task-{id}: {任务标题}

## 任务描述
...

## 分配Agent
...

## 状态
- [ ] 待开始
- [ ] 进行中
- [ ] 待审查
- [ ] 已完成

## 依赖任务
...

## 输出文件
...
```

---

## 沟通协议

### 1. Agent间通信

- 通过共享Markdown文件进行异步通信
- 使用特定格式标记阻塞/依赖

### 2. 向Orchestrator汇报

- 完成阶段性工作时提交报告
- 遇到阻塞问题时立即上报
- 使用标准报告模板

### 3. 文件命名约定

```
.agents/
├── prompts/
│   ├── orchestrator_prompt.md
│   ├── architect_prompt.md
│   ├── developer_prompt.md
│   ├── reviewer_prompt.md
│   └── reporter_prompt.md
├── reports/
│   ├── architect/
│   │   └── 20240320_143000_report.md
│   ├── developer/
│   │   └── 20240320_150000_report.md
│   └── reviewer/
│       └── 20240320_160000_report.md
├── tasks/
│   ├── TASK-001_building_simulator.md
│   └── TASK-002_hvac_env.md
└── status/
    ├── current_sprint.md
    └── backlog.md
```

---

## 质量门禁

### 代码提交前检查清单

- [ ] 代码符合PEP8规范
- [ ] 包含类型注解
- [ ] 包含Docstring
- [ ] 单元测试覆盖率>80%
- [ ] 通过静态类型检查 (mypy)
- [ ] 无安全漏洞 (bandit)

### 设计审查检查清单

- [ ] 接口定义清晰
- [ ] 模块职责单一
- [ ] 依赖关系合理
- [ ] 扩展性考虑
- [ ] 性能瓶颈分析

---

## 项目当前状态

### 已完成
- [x] 技术栈调研文档 (3个方向)
- [x] 基础设施复用分析
- [x] 方向一基础代码框架

### 进行中
- [ ] 方向一异常检测模块
- [ ] 方向一GraphRAG接口
- [ ] 方向二代码实现
- [ ] 方向三代码实现

### 待开始
- [ ] 系统集成测试
- [ ] 部署配置
- [ ] 文档完善

---

*最后更新: 2026年3月20日*
*版本: v1.0*
