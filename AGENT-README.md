# Kimi Copilot Agent 配置系统

## 概述

这是一个增强的 GitHub Copilot Agent 配置系统，将 Kimi AI 接入 VS Code Copilot，提供专业的编程辅助能力。

## 文件结构

```
~/AppData/Roaming/Code/User/
├── .agent.md                    # Agent 核心配置 (增强提示词)
├── mcp.json                     # MCP 服务配置
├── copilot-instructions.md      # 项目指令模板
├── AGENT-README.md             # 本文件
└── .vscode/
    └── settings.json           # VS Code 设置
```

## 核心特性

### 1. 深度思考框架 (Think Deep)

每次编码任务前，系统会引导完成：
- **分析阶段**: 理解需求、现有代码、技术栈
- **设计阶段**: 方案选择、最小改动、测试策略
- **执行阶段**: 逐步实施、频繁验证、及时反馈

### 2. 常见陷阱防护

| 陷阱 | 防护机制 |
|------|---------|
| 上下文幻觉 | 强制先读取文件再修改 |
| 过度优化 | 先正确再优化原则 |
| 破坏兼容性 | 检查调用方，保持兼容 |
| 忽略边界 | 显式处理所有错误路径 |
| 测试遗漏 | 修改后必须验证 |
| 安全漏洞 | 输入验证 + 参数化查询 |

### 3. 四阶段开发流程

```
Explore (探索) → Plan (规划) → Execute (执行) → Review (审查)
```

### 4. MCP 服务集成

已配置 8 个自动启动的 MCP 服务：
- GitHub API 操作
- 浏览器自动化 (Playwright)
- 网页内容获取
- 文件系统操作
- Git 仓库操作
- 知识图谱记忆
- 代码上下文检索
- Markdown 转换

### 5. Skill 智能推荐

当检测到适合使用 Skill 的场景时，主动询问用户是否需要添加。

## 使用方法

### 启动 Agent

1. 打开 VS Code Copilot Chat (`Ctrl+Alt+I`)
2. 选择 **Agent** 模式
3. 开始对话，Agent 会自动加载配置

### 文件加载顺序

每次唤起时，Agent 按顺序读取：
1. `~/.agent.md` - 环境配置和编码规范
2. `~/.vscode/mcp.json` - MCP 服务配置
3. `.github/copilot-instructions.md` - 项目级指令
4. `.github/skills/**/SKILL.md` - 特定技能

### 项目定制

复制 `copilot-instructions.md` 到项目根目录，添加项目特定规则：

```yaml
---
name: my-project
description: "My project specific instructions"
---

# 项目规范

- 使用 TypeScript 严格模式
- 组件必须使用 Props 接口
- 测试覆盖率 > 90%
```

## 最佳实践

### 与 Agent 有效沟通

✅ **好的请求：**
- "帮我重构这个函数，提高可读性"
- "检查这个文件是否有内存泄漏"
- "为这个 API 添加错误处理"

❌ **避免：**
- "修复这个" (缺乏上下文)
- "优化代码" (过于笼统)
- "这样做对吗？" (没有具体代码)

### 利用 MCP 服务

Agent 会自动使用合适的 MCP 服务：
- Git 操作 → `modelcontextprotocol/git`
- 文件操作 → `modelcontextprotocol/filesystem`
- GitHub API → `github/github-mcp-server`

### 添加自定义 Skill

当 Agent 建议添加 Skill 时，可以：
- **接受**: Agent 会创建 Skill 文件
- **拒绝**: Agent 会继续当前方式工作
- **稍后**: Agent 会在适当时机再次提醒

## 故障排除

### MCP 服务无法启动

1. 检查 `npx` 和 `uvx` 是否可用：
   ```bash
   npx --version
   uvx --version
   ```

2. 重启 VS Code 以刷新环境变量

3. 检查 MCP 配置：
   ```bash
   Ctrl+Shift+P → MCP: List Servers
   ```

### Agent 行为不符合预期

1. 检查 `.agent.md` 是否被正确加载
2. 查看 VS Code 输出面板中的 MCP 日志
3. 尝试重启 Copilot Chat

## 更新和维护

### 更新 Agent 配置

编辑 `~/.agent.md`，修改：
- 编码规范
- MCP 服务配置
- Skill 触发条件

### 添加新的 MCP 服务

编辑 `~/.vscode/mcp.json`，添加新的服务器配置。

### 创建新 Skill

在 `.github/skills/<skill-name>/` 目录创建 `SKILL.md` 文件。

## 参考资源

- [GitHub Copilot 文档](https://docs.github.com/en/copilot)
- [MCP 协议文档](https://modelcontextprotocol.io/)
- [VS Code Agent 模式](https://code.visualstudio.com/docs/copilot/chat/chat-agent-mode)

---

**配置版本**: 1.0  
**最后更新**: 2026-03-19  
**维护者**: Kimi + User
