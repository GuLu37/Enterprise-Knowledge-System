# 知识库文件列表与内容概括 Skill

## 用途

列出知识库中已上传（默认状态为 `ready`）的文档，并可选概括指定文档的内容。
适用场景：

- 用户问「知识库里有哪些文件 / 文档」
- 用户按文件名问某份文档讲了什么，如「帮我看看 04_产品与账号常见问题 这个文档」
- 用户问某文档包含哪些章节 / 主题

## 不适用场景

- 按文档**内容**检索某个具体答案（如「账号被锁定怎么办」）→ 应使用 `search_knowledge_base`
- 统计高频热词 → 应使用 `document_hot_word_statistics`

## 参数

- `file_name`：按文件名模糊筛选；用户提到具体文件名时填写。
- `document_ids`：只列指定文档 ID。
- `file_types`：按文件类型筛选，如 `['pdf','docx','md']`。
- `statuses`：按状态筛选，默认 `['ready']`（已入库）。
- `summarize`：是否概括内容，默认 `true`。
- `preview_chars`：单文档内容预览字符上限，默认 1500。
- `limit`：最多返回文档数，默认 20。

## 返回结构

每个文档返回：`document_id`、文件名、类型、状态、大小、`chunk_count`、时间，
以及（summarize=true 时）`outline`（章节大纲）、`content_preview`（内容预览）、`topic_keywords`（主题关键词）。

## 回答规范

- 模型拿到结果后，用自然语言汇总文件清单与概括，不要逐字复述 `content_preview`。
- 若用户问的文件名在列表里不存在，明确告知「该文档未在知识库中」，并建议确认是否已上传或完成索引。
- 若 `summary.document_count == 0`，直接说明未找到匹配文档。
