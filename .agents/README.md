# Multi-Agent协作系统

## 概述

这是一个为"节能减排AI系统"项目设计的多Agent协作架构，通过角色分工和标准化协作流程，提高开发效率和代码质量。

## Agent角色

### 1. Orchestrator (总指挥)
**文件**: `ORCHESTRATOR.md`

- **职责**: 项目管理、任务分配、进度跟踪、跨Agent协调
- **汇报对象**: 用户 (Human)
- **下属**: 所有其他Agent

### 2. Architect (架构师)
**文件**: `ARCHITECT.md`

- **职责**: 架构设计、技术选型、接口定义、代码结构规划
- **汇报对象**: Orchestrator
- **协作**: Developer (交付设计), Reviewer (审查架构)

### 3. Developer (开发工程师)
**文件**: `DEVELOPER.md`

- **职责**: 代码实现、单元测试、文档编写、Bug修复
- **汇报对象**: Orchestrator
- **协作**: Architect (获取设计), Reviewer (代码审查)

### 4. Tester (测试工程师)
**文件**: `TESTER.md`

- **职责**: 测试策略、测试用例设计、测试执行、缺陷管理
- **汇报对象**: Orchestrator
- **协作**: Developer (获取代码)

### 5. Reviewer (代码审查员)
**文件**: `REVIEWER.md`

- **职责**: 代码质量审查、架构合规检查、安全审计
- **汇报对象**: Orchestrator
- **协作**: Developer (审查代码)

## 协作协议

**文件**: `COLLABORATION_PROTOCOL.md`

包含：
- 通信协议标准
- 协作流程规范
- 汇报机制
- 冲突解决流程
- 质量保证标准

## 工作流程

```
用户需求
    ↓
Orchestrator 分析并分解任务
    ↓
┌──────────┬──────────┬──────────┐
↓          ↓          ↓          ↓
Architect Developer  Tester   Reviewer
(设计)    (实现)    (测试)   (审查)
    ↓          ↓          ↓          ↓
    └──────────┴──────────┴──────────┘
                    ↓
            Orchestrator 整合
                    ↓
                交付用户
```

## 快速开始

### 1. 启动项目

用户向Orchestrator提出需求，Orchestrator：
1. 分析需求
2. 创建任务卡片
3. 分配给相应Agent
4. 跟踪执行进度

### 2. Agent响应

各Agent按照`COLLABORATION_PROTOCOL.md`中的规范：
1. 接收任务
2. 执行任务
3. 定期汇报
4. 交付成果

### 3. 质量把控

Reviewer和Tester确保：
- 代码质量达标
- 测试覆盖充分
- 文档完整准确

## 文档结构

```
.agents/
├── README.md                    # 本文件
├── COLLABORATION_PROTOCOL.md    # 协作协议
├── ORCHESTRATOR.md             # 总指挥Agent
├── ARCHITECT.md                # 架构师Agent
├── DEVELOPER.md                # 开发工程师Agent
├── TESTER.md                   # 测试工程师Agent
├── REVIEWER.md                 # 代码审查员Agent
└── tasks/                      # 任务卡片存储
    ├── TODO/                   # 待办任务
    ├── IN_PROGRESS/            # 进行中
    ├── REVIEW/                 # 审查中
    └── DONE/                   # 已完成
```

## 使用示例

### 场景1: 新功能开发

```
用户: "实现HVAC温度控制功能"
    ↓
Orchestrator: 创建任务，分配给Architect
    ↓
Architect: 设计架构，输出ADR文档
    ↓
Orchestrator: 将设计交付Developer
    ↓
Developer: 实现代码，提交单元测试
    ↓
Reviewer: 审查代码，提出修改意见
    ↓
Developer: 修复问题，重新提交
    ↓
Tester: 执行测试，验证功能
    ↓
Orchestrator: 整合结果，向用户汇报
```

### 场景2: Bug修复

```
用户: "修复温度控制不准确的问题"
    ↓
Orchestrator: 创建Bug修复任务，分配给Developer
    ↓
Developer: 定位问题，实现修复
    ↓
Reviewer: 审查修复方案
    ↓
Tester: 验证修复，执行回归测试
    ↓
Orchestrator: 确认修复完成
```

## 贡献指南

如需修改Agent配置：
1. 编辑对应的 `.md` 文件
2. 更新版本号和日期
3. 记录变更日志
4. 通知Orchestrator

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0 | 2026-03-20 | 初始版本，包含5个Agent角色和协作协议 |

---

*维护者: Orchestrator Agent*
*最后更新: 2026-03-20*
