# QuantPilot 链路整改计划

日期：2026-04-17  
状态：计划中，等待今日 `14:50` 自动交易完成后执行  
范围：`collector -> nightly -> retry -> pretrade -> trade -> weekly_train -> report`

## 1. 目标

这次整改不以“继续打补丁”为目标，而以 4 件事为目标：

1. 把会导致误成功、误健康、误上线的链路级错误先清掉。
2. 把废弃代码、历史方案、文档漂移收敛掉，降低维护负担。
3. 判断当前架构是“局部修复足够”还是“需要明确重构边界”。
4. 明确 NAS -> Mac mini 数据同步的容量风险，并给出长期可执行的存储策略。

## 2. 当前实际链路

当前生产链路以 Mac mini 宿主机 `crontab` 为准，不再以旧 Docker scheduler 方案为准：

- 工作日 `10:00` / `13:30`：`run_pretrade_watchdog.sh`
- 工作日 `14:40`：`sync_data.sh`
- 工作日 `14:50`：`run_trade.sh`
- 工作日 `19:00`：`run_daily.sh`
- 周六 `10:00`：`run_weekly_train.sh`
- NAS 工作日 `18:00`：`collector/scheduler.py` 主采集

本机当前数据占用：

- `/Users/theo/quantpilot_data`：`5.1G`
- `/Users/theo/quantpilot_data/qlib_data`：`738M`
- `/Users/theo/quantpilot_data/models`：`83M`
- `/Users/theo/quantpilot_data/signals`：`18M`
- `/Users/theo/quantpilot_data/output`：`912K`

结论：

- 眼下还没有“马上打满盘”的风险。
- 但如果模型版本、回测输出、历史信号、失败补跑产物持续累积，Mac mini 的存储压力会逐步变成真实问题。

## 3. 已识别的核心风险

### P0：必须先修

1. Shell 包装层错误吞没

- `run_daily.sh`
- `run_trade.sh`
- `run_pretrade_watchdog.sh`

当前使用了 `if ! cmd; then RC=$?; fi` 这种写法，失败时可能拿到的是取反后的 `0`，导致外层任务误判成功。

2. 周训练无门槛直接替换线上模型

- 当前流程是：训练成功 -> 回测 -> 直接 deploy -> 直接 promote latest signal
- 缺少 promotion gate
- 这会把“能训练出来”和“适合上线”混为一件事

3. 健康检查不够“目标交易日感知”

- pretrade watchdog / healthcheck 更像“本地是否对齐”
- 不是“今天理论上应该推进到哪一天”
- 如果本地 snapshot 和 signal 一起停在旧日期，仍有误报 `ok` 的风险

4. 周训练邮件链路与日报链路分叉

- 日报已经通过 iCloud SMTP 跑通
- 周训练仍保留单独 SMTP 实现
- 两套邮件实现继续并存，会让下次失败通知再次变得不可靠

### P1：高优先级结构问题

5. `completed_a_share` 仍是 nightly 单点门闸

- collector 自己成功，不代表 completion metadata 一定推进
- nightly 又严格依赖 completion metadata
- 这会继续制造“collector 看似成功，nightly 卡死”的结构性问题

6. 训练 / 推理 / 回测 / 实盘 universe 不一致

- 训练配置仍是 `all`
- 推理实际是 `SH./SZ.`
- live trade 实际只做 `SH.`
- weekly backtest 也被收窄到 `SH.`

这会让训练目标、评估口径、实盘口径持续偏离。

7. live trade 与 backtest 仍未完全同口径

- live trade 只看前 `TOP_N * 3` 实时行情
- 信号日涨跌幅只读前 `50` 只
- backtest 是对全量候选逐只过滤

这会导致“回测可买，实盘不一定补位到相同集合”。

### P2：中期治理问题

8. Shell orchestration 过重，状态模型分散

- collector 用 metadata
- nightly 用 shell 判断
- watchdog 用另一套读取逻辑
- healthcheck 再重复一遍

状态源太多，难以保证一致性。

9. 测试覆盖偏文本断言，不是执行级验证

- 当前不少测试只验证“脚本里有这行字符串”
- 无法兜住 shell 退出码、锁行为、补跑行为这类真实 orchestration 缺陷

## 4. 废弃代码 / 历史代码清理计划

原则：代码是负债，不是资产。未被生产引用、只增加认知负担的代码应当删除，不应长期保留。

### 第一批候选（优先审计）

1. `scheduler/`

- 当前生产已改为宿主机 `crontab + scripts/`
- `scheduler/` 目录是历史 Docker 调度方案
- 文档里也已标注为“历史方案”
- 计划：确认没有实际容器依赖后整体删除

2. `dashboard/`

- 当前系统里已有 `observer/`
- `dashboard/` 仅通过 `strategy_cli.py` 间接引用
- 需确认是否仍有真实使用者

3. `strategy_cli.py`

- 现在生产入口已是 `main.py`、`scripts/*.sh`、`python -m ...`
- 若只是旧 CLI 包装层，应考虑删除

4. `Users/`

- 当前为空目录
- 明显属于无业务价值的残留
- 可直接删除

### 第二批候选（需结合引用图判断）

5. `docker-compose.mac.yml` 中仅服务于旧运行模式的段落
6. README / docs 中仍描述旧时间线、旧容器执行方式的内容
7. 兼容旧模型命名、旧信号命名、旧 fallback 分支中已经不再需要的兼容层

输出要求：

- 给出“可直接删 / 先迁移再删 / 暂时保留”三类清单
- 删除必须附引用依据，不靠主观感觉

## 5. 架构是否需要重构

结论先行：需要，但不是“大重写”，而是分层收拢。

### 不建议做的事

- 不做整仓库推倒重来
- 不做 collector / trader / inference 全部重写
- 不引入新的编排系统替代 cron

### 建议做的重构边界

1. 编排层收口

把当前散落在 shell、watchdog、healthcheck 里的日期判断、状态判断、失败重试，收敛成一套统一的 Python orchestration 模块。

2. 状态源收口

统一只认这几类状态：

- NAS completion metadata
- local completion metadata
- local latest instruments date
- latest signal date
- latest trade execution result

不允许每个脚本自己再发明一套“是否 ready”的判断。

3. 通知层收口

- 日报邮件
- health alert
- weekly train report

全部复用同一发送实现，不再保留第二套 SMTP 逻辑。

4. 模型上线层收口

把“训练成功”和“允许上线”拆开：

- train
- evaluate
- compare against baseline
- promote or reject

### 重构判断标准

若一项改动满足下面任一条件，就进重构范围，而不是继续补丁：

- 需要 3 个以上脚本同时复制相同状态判断
- 同一个失败场景在 collector / nightly / watchdog 中要修 2 次以上
- 同一业务规则在 backtest 和 live trade 出现两套实现

## 6. Mac mini 存储策略计划

当前本机 `qlib_data` 只有 `738M`，短期安全；但长期风险来自“持续累积的派生物”，不是主数据本体。

### 风险来源

1. 多版本模型持续堆积
2. 周训练输出目录持续堆积
3. 失败补跑产物、baseline 比较产物持续堆积
4. logs / health snapshots / retry logs 长期无清理
5. 若未来同步策略从原子覆盖变成保留多版本快照，容量会快速上升

### 计划中的治理动作

1. 明确“本机只存运行所需，不做长期归档”

- 长期归档以 NAS 为主
- Mac mini 只保留当前运行版本和有限回滚窗口

2. 加 retention policy

- models：保留最新正式版 + 最近 N 个 staging / rerun
- signals：保留最近 N 天
- output：保留最近 N 次训练/回测
- logs/health：按天滚动并保留固定窗口

3. 增加容量监控

- 在 pretrade 或 nightly 健康检查里加入本地磁盘占用阈值
- 比如使用率超过 `80%` 告警，超过 `90%` 升级为错误

4. 评估是否需要“本机最小化数据副本”

- 当前 Qlib 全量同步只有数百 MB，短期没必要折腾远程读
- 优先做 retention，比改成 NAS 远程直接读更稳

## 7. 执行计划

### Phase 0：冻结期

时间：今天自动交易完成前

动作：

- 不修改生产交易逻辑
- 只完成审计和整改计划确认

### Phase 1：正确性修复

目标：先把“误成功、误健康、误上线”清掉

任务：

1. 修复 shell 返回码吞没
2. 增加脚本级执行测试
3. 让 pretrade / healthcheck 引入目标交易日判断
4. 把 weekly train 邮件改为复用 reporter 发送链路
5. 给 weekly promotion 加 baseline gate

验收标准：

- 失败任务外层必定非零退出
- healthcheck 不再把 stale-but-aligned 误报为 `ok`
- weekly train 若评估不达标，不会替换线上模型

### Phase 2：口径统一

目标：让训练、回测、推理、实盘尽量同口径

任务：

1. 明确最终交易 universe
2. 对齐 config / train / backtest / live trade
3. 对齐候选过滤深度
4. 明确 limit-up / stop-loss / hold bonus 的唯一实现位置

验收标准：

- 同一条规则只有一份权威实现
- backtest 与 live trade 的差异被显式记录，而不是隐式存在

### Phase 3：代码负债清理

目标：删除历史方案和未使用代码

任务：

1. 产出“删除候选清单”
2. 逐项做引用核对
3. 删除 `Users/`
4. 删除确认无生产引用的历史调度/旧 dashboard/旧 CLI
5. 清理文档漂移

验收标准：

- 每个保留目录都能说清生产价值
- 每个删除目录都有引用核对依据

### Phase 4：架构收口

目标：降低后续维护成本，而不是继续靠补跑堆系统韧性

任务：

1. 抽离统一 readiness / completion / signal freshness 判定模块
2. 抽离统一 notification adapter
3. 抽离统一 model promotion gate
4. 评估 shell orchestration 收缩到 Python 的边界

验收标准：

- 新增业务规则不需要同时改 3 个脚本
- 失败场景只有一个权威判定入口

## 8. 开始执行前的确认项

正式动手前，先确认这 3 个策略：

1. 训练 / 回测 / 实盘最终是否统一为 `SH.`，还是回到全 A。
2. 周训练 promotion gate 采用什么硬门槛。
3. Mac mini 本地保留策略采用什么窗口。

## 9. 本次计划产出物

执行过程中会补齐这些文档/工件：

- 风险清单
- 删除候选清单
- 架构收口边界说明
- 存储 retention 策略
- 分阶段变更记录

---

当前建议：  
今天自动交易完成后，先执行 `Phase 1`，不要直接进入大规模重构或删库式清理。
