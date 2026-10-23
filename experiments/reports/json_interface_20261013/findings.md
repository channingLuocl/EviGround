# 兼容接口 JSON 输出检查

检查日期：2026-10-13（Asia/Shanghai）。接口和模型见原始报告。

## 结论

客户端把 `response_format` 放在实际 HTTP 请求 JSON 的顶层，没有漏传。
当前接口在以下测试中表现符合要求；没有复现此前批次的大量格式错误。
这证明当前端点的外部行为，不能证明网关内部逐层转发的内容，也不能保证所有请求和路由均正常。

## 实测结果

顺序执行 8 次请求，无重试。完整请求（不含鉴权信息、图片已脱敏）、响应、耗时在 `report.json`。

| 测试 | 结果 |
| --- | --- |
| 简单文本 JSON | HTTP 200，返回指定对象 |
| 纯文本对照，不传 response_format | HTTP 200，返回 `PROBE_COMPLETE` |
| 同一提示，传 json_object，要求纯文本以测试格式约束 | HTTP 200，返回 `{"response":"PROBE_COMPLETE"}` |
| 同一提示，传 strict json_schema，提示不含 schema 中的标记值 | HTTP 200，返回 schema 指定的 `{"marker":"seeground_json_probe","ok":true}` |
| 传非法 response_format.type | HTTP 400，invalid_request_error |
| 真实生产解析提示：The largest picture in the room. | HTTP 200，合法 JSON，必需字段类型检查通过 |
| 真实生产解析提示：the smaller chair | HTTP 200，合法 JSON，必需字段类型检查通过 |
| 红色图片 JSON | HTTP 200，返回指定字段并识别 red |

随后按并发 2 重复第一个生产解析请求 4 次，无重试，全部 HTTP 200，合法 JSON，必需字段类型检查通过。
耗时分别为 10.185、7.409、9.520、15.858 秒。完整记录在 `concurrency_report.json`。
本次生产解析检查只覆盖 JSON 及所列字段类型，不等同于完整下游语义验证或场景推理准确率验证。

## 与失败批次的对照

历史运行：`local_outputs/v6_1_local_20261013_001107`。
历史 manifest 确认 workers=2、api_concurrency=2。
`qwen_api.py`、`real_scene.py`、`diagnostic.py`、`query_syntax.py` 的当前 SHA-256 与历史 manifest 完全一致。

65 个推理失败样本的 console.log 最终异常：52 个非法 JSON、9 个字段协议错误、1 个缺少决策 reason、2 个超时、1 个 HTTP 502。
历史调用记录中有 54 次 HTTP 200 的 JSON 解析错误，包括最终复核调用；它们与 52 个因非法 JSON 终止的推理样本是不同统计口径。
历史记录中的普通文字和 `'''json` 包裹内容表明当时端点没有在这些响应上兑现 JSON 输出要求。
当前通过的少量请求不能否定这些历史错误，也不能定位历史故障发生在网关、上游或哪条路由。

## 协议含义与下一步

OpenAI 官方文档说明 `json_object` 保证 JSON 格式，不保证字段符合业务 schema；`json_schema` + strict 才约束字段。
当前 SeeGround 使用 json_object，因此字段缺失还需业务校验。
本次 strict schema 探测通过，但只验证了一个简单 schema；未把生产协议改成 strict schema。

文档：https://developers.openai.com/api/docs/guides/structured-outputs

若再次出现 HTTP 200 + 非 JSON，应使用 run.json 中的 request_id、时间、模型和请求摘要核对网关入站请求、上游出站请求、实际路由和返回内容。
当前没有网关管理权限或服务器日志，所以未确认内部转发过程，也没有证据把原因归于某个具体上游。
没有修改生产推理代码、运行脚本或 .env。

## 本地验证

7 个与文本/图片请求参数、解析入口、非法 JSON 拒绝及现有 probe 有关的定向测试通过。
最初执行的 13 个测试中，有 2 个子测试失败：原有测试期望 503/timeout 只调用一次，实际调用两次。
这涉及原有恢复策略，不是请求格式传递断言失败；本次检查未修改这部分代码。
`git diff --check` 通过。

## 复跑

在 SeeGround 目录加载原 .env 后，设置 `PYTHONPATH=local`，使用 `.venv/bin/python` 运行同目录 `probe.py`，再运行 `concurrency_probe.py`。
复跑会发出真实 API 请求，并覆盖本目录报告。
