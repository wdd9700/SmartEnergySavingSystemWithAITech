# Multi-Agent Collaboration Protocol
# 多Agent协作协议

## 1. 协作架构

```
                    ┌─────────────┐
                    │    Human    │
                    │   (User)    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Orchestrator│
                    │  (总指挥)   │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐       ┌────▼────┐       ┌────▼────┐
   │Architect│       │Developer│       │ Tester  │
   │ (架构师) │       │ (开发者) │       │ (测试员) │
   └────┬────┘       └────┬────┘       └────┬────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Reviewer   │
                    │  (审查员)   │
                    └─────────────┘
```

## 2. 通信协议

### 2.1 消息格式

所有Agent间通信使用标准Markdown格式：

```markdown
---
type: [REQUEST|RESPONSE|REPORT|NOTIFY]
from: [Agent名称]
to: [Agent名称或ALL]
timestamp: [ISO 8601格式]
priority: [URGENT|HIGH|NORMAL|LOW]
---

## 主题
[消息主题]

## 内容
[详细内容]

## 上下文
[相关背景信息]

## 期望响应
[期望的响应或行动]

## 附件
[相关文件或链接]
```

### 2.2 消息类型

| 类型 | 用途 | 响应要求 |
|------|------|----------|
| REQUEST | 请求协助或信息 | 必须响应 |
| RESPONSE | 对请求的响应 | 无需响应 |
| REPORT | 进度或状态汇报 | 无需响应 |
| NOTIFY | 通知信息 | 无需响应 |

## 3. 协作流程

### 3.1 任务生命周期

```
创建 → 分配 → 执行 → 审查 → 测试 → 完成
```

### 3.2 状态流转

```
┌─────────┐    ┌─────────┐    ┌─────────┐
│  TODO   │───→│ ASSIGNED│───→│   WIP   │
└─────────┘    └─────────┘    └────┬────┘
                                     │
┌─────────┐    ┌─────────┐    ┌────▼────┐
│  DONE   │←───│  TEST   │←───│ REVIEW  │
└─────────┘    └─────────┘    └─────────┘
```

## 4. Agent职责矩阵

| 职责 | Orchestrator | Architect | Developer | Tester | Reviewer |
|------|-------------|-----------|-----------|--------|----------|
| 任务分配 | ✅ | ❌ | ❌ | ❌ | ❌ |
| 架构设计 | ❌ | ✅ | ❌ | ❌ | 咨询 |
| 代码实现 | ❌ | ❌ | ✅ | ❌ | ❌ |
| 代码审查 | 监督 | 咨询 | 修复 | ❌ | ✅ |
| 测试执行 | 监督 | ❌ | 协助 | ✅ | ❌ |
| 进度跟踪 | ✅ | 汇报 | 汇报 | 汇报 | 汇报 |
| 质量控制 | ✅ | ❌ | ❌ | 协助 | ✅ |

## 5. 汇报机制

### 5.1 日报格式

```markdown
## Daily Report - [Agent名称]
**Date**: YYYY-MM-DD

### 今日完成
- [任务1]: [状态]
- [任务2]: [状态]

### 进行中
- [任务3]: [进度%] - [预计完成时间]

### 阻塞项
- [问题]: [需要的协助]

### 明日计划
- [计划1]
- [计划2]

### 风险预警
- [风险描述及影响]
```

### 5.2 紧急汇报触发条件

- 发现Critical级别Bug
- 进度延期超过20%
- 技术方案不可行
- 资源冲突无法解决

## 6. 冲突解决

### 6.1 技术分歧

1. 双方陈述理由
2. 引用最佳实践或标准
3. 寻求Architect意见
4. Orchestrator最终决策

### 6.2 资源冲突

1. 明确资源需求
2. 评估优先级
3. Orchestrator协调
4. 必要时向Human汇报

## 7. 文档规范

### 7.1 必须维护的文档

| 文档 | 负责人 | 更新时机 |
|------|--------|----------|
| 架构设计(ADR) | Architect | 设计变更 |
| API文档 | Developer | 接口变更 |
| 测试报告 | Tester | 测试完成 |
| 审查记录 | Reviewer | 审查完成 |
| 项目状态 | Orchestrator | 每日 |

### 7.2 文档存储位置

```
docs/
├── architecture/     # 架构文档 (Architect维护)
├── api/             # API文档 (Developer维护)
├── test-reports/    # 测试报告 (Tester维护)
├── reviews/         # 审查记录 (Reviewer维护)
└── status/          # 项目状态 (Orchestrator维护)
```

## 8. 质量保证

### 8.1 代码合并标准

- [ ] 通过Reviewer审查
- [ ] 通过Tester测试
- [ ] 代码覆盖率≥80%
- [ ] 无Critical/High级别问题
- [ ] 文档已更新

### 8.2 发布检查清单

- [ ] 所有P0/P1任务完成
- [ ] 集成测试通过
- [ ] 性能测试达标
- [ ] 文档完整
- [ ] 已知问题已记录

## 9. 工具链

### 9.1 协作工具

| 用途 | 工具 | 备注 |
|------|------|------|
| 任务管理 | Markdown文件 | 简单轻量 |
| 文档协作 | Git | 版本控制 |
| 代码审查 | PR Review | GitHub/GitLab |
| 沟通记录 | Agent Report | 标准格式 |

### 9.2 开发工具

| 用途 | 工具 |
|------|------|
| 代码风格 | black, flake8 |
| 类型检查 | mypy |
| 测试框架 | pytest |
| 覆盖率 | pytest-cov |
| 安全扫描 | bandit |

## 10. 附录

### 10.1 术语表

| 术语 | 含义 |
|------|------|
| WIP | Work In Progress (进行中) |
| ADR | Architecture Decision Record (架构决策记录) |
| PR | Pull Request (合并请求) |
| LGTM | Looks Good To Me (看起来不错) |

### 10.2 参考资源

- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [PEP 8 - Python代码风格](https://pep8.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

*版本: 1.0*
*创建日期: 2026-03-20*
*维护者: Orchestrator Agent*
