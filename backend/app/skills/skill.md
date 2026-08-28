# 技能索引

这是给 agent 先读的索引页，用于按需渐进式披露。

只在当前任务命中对应用途时，再打开下面的技能文件；不要一上来把所有 skill 都读完。

- `document_hot_word_statistics.md`，位置：`backend/app/skills/document_hot_word_statistics.md`。用途：按文件名或文档 ID 统计某个文档内容里的高频热词。
- `memory_query.md`，位置：`backend/app/skills/memory_query.md`。用途：查询当前登录用户的画像记忆和长期事件记忆。
- `delete_profile_memory.md`，位置：`backend/app/skills/delete_profile_memory.md`。用途：用户明确要求删除、遗忘、清除个性化记忆（画像记忆）时，按 memory_key 精确或按关键词模糊删除。


如果当前任务不匹配任何 Skill，不要读取其它技能文件。
