//! 对齐 Python `app.utils.chunking` 的对话/语义长期记忆切块。

use crate::ingest::split_text_chunks;

pub const CHAT_CHUNK_SIZE: usize = 500;
pub const CHAT_CHUNK_OVERLAP: usize = 80;
pub const LONG_TERM_MEMORY_TOKEN_LIMIT: usize = 700;
pub const TOPIC_CONTAINMENT_THRESHOLD: f64 = 0.50;

const ROLE_LABELS: &[(&str, &str)] = &[
    ("system", "系统"),
    ("user", "用户"),
    ("assistant", "助手"),
];

const TOPIC_CONTINUATION_CUES: &[&str] = &[
    "那", "这个", "它", "上面", "刚才", "继续", "还是", "其中", "他的", "她的", "该",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversationMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversationChunk {
    pub chunk_index: usize,
    pub text: String,
    pub turn_start: i64,
    pub turn_end: i64,
    pub message_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticMemoryChunk {
    pub chunk_index: usize,
    pub topic: String,
    pub summary: String,
    pub transcript: String,
    pub text: String,
    pub turn_start: i64,
    pub turn_end: i64,
    pub message_count: usize,
    pub trigger_reason: String,
}

fn char_len(text: &str) -> usize {
    text.chars().count()
}

fn role_label(role: &str) -> Option<&'static str> {
    ROLE_LABELS
        .iter()
        .find(|(key, _)| *key == role)
        .map(|(_, label)| *label)
}

pub fn normalize_conversation_message(
    role: &str,
    content: &str,
    include_system: bool,
) -> Option<ConversationMessage> {
    let role = role.trim().to_lowercase();
    let content = content.trim();
    if role.is_empty() || content.is_empty() {
        return None;
    }
    if role == "system" && !include_system {
        return None;
    }
    if role_label(&role).is_none() {
        return None;
    }
    Some(ConversationMessage {
        role,
        content: content.to_string(),
    })
}

fn message_to_block(message: &ConversationMessage) -> String {
    let label = role_label(&message.role).unwrap_or(message.role.as_str());
    format!("{}: {}", label, message.content.trim())
}

pub fn split_conversation_turns(
    messages: &[ConversationMessage],
    include_system: bool,
) -> Vec<Vec<ConversationMessage>> {
    let normalized: Vec<ConversationMessage> = messages
        .iter()
        .filter_map(|message| {
            normalize_conversation_message(&message.role, &message.content, include_system)
        })
        .collect();
    if normalized.is_empty() {
        return Vec::new();
    }
    let mut turns: Vec<Vec<ConversationMessage>> = Vec::new();
    let mut current: Vec<ConversationMessage> = Vec::new();
    for message in normalized {
        if message.role == "user" {
            if !current.is_empty() {
                turns.push(std::mem::take(&mut current));
            }
            current = vec![message];
            continue;
        }
        if current.is_empty() {
            current = vec![message];
        } else {
            current.push(message);
        }
    }
    if !current.is_empty() {
        turns.push(current);
    }
    turns
}

pub fn render_long_term_memory_transcript(turns: &[Vec<ConversationMessage>]) -> String {
    turns
        .iter()
        .map(|turn| {
            turn.iter()
                .map(message_to_block)
                .collect::<Vec<_>>()
                .join("\n")
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

pub fn format_long_term_memory_chunk(summary: &str, transcript: &str, topic: &str) -> String {
    let mut parts = Vec::new();
    if !topic.trim().is_empty() {
        parts.push(format!("【主题】{}", topic.trim()));
    }
    parts.push(format!("【摘要】{}", summary.trim()));
    parts.push("【原文】".into());
    parts.push(transcript.trim().to_string());
    parts.join("\n")
}

pub fn estimate_text_tokens(text: &str) -> usize {
    let normalized = text.trim();
    if normalized.is_empty() {
        return 0;
    }
    let mut count = 0usize;
    let mut ascii = String::new();
    let flush_ascii = |ascii: &mut String, count: &mut usize| {
        if !ascii.is_empty() {
            *count += 1;
            ascii.clear();
        }
    };
    for ch in normalized.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            ascii.push(ch);
        } else {
            flush_ascii(&mut ascii, &mut count);
            if ('\u{4e00}'..='\u{9fff}').contains(&ch) {
                count += 1;
            }
        }
    }
    flush_ascii(&mut ascii, &mut count);
    count.max(1)
}

pub fn topic_text_similarity(current_text: &str, next_text: &str) -> f64 {
    fn bigrams(text: &str) -> std::collections::HashSet<String> {
        let normalized: String = text.chars().filter(|ch| !ch.is_whitespace()).collect();
        let chars: Vec<char> = normalized.chars().collect();
        if chars.len() < 2 {
            return std::collections::HashSet::new();
        }
        (0..chars.len() - 1)
            .map(|index| chars[index..index + 2].iter().collect())
            .collect()
    }
    let current = bigrams(current_text);
    let incoming = bigrams(next_text);
    if current.is_empty() || incoming.is_empty() {
        return 0.0;
    }
    let hits = incoming.iter().filter(|gram| current.contains(*gram)).count();
    hits as f64 / incoming.len() as f64
}

/// 高置信延续则视为未换题，其余交给调用方（LLM）。
pub fn high_confidence_topic_continuation(current_text: &str, next_text: &str) -> bool {
    let stripped = next_text.trim();
    if stripped.is_empty() {
        return false;
    }
    if TOPIC_CONTINUATION_CUES
        .iter()
        .any(|cue| stripped.starts_with(cue))
    {
        return true;
    }
    topic_text_similarity(current_text, stripped) >= TOPIC_CONTAINMENT_THRESHOLD
}

fn split_message_into_blocks(
    message: &ConversationMessage,
    chunk_size: usize,
    chunk_overlap: usize,
) -> Vec<String> {
    let block = message_to_block(message);
    if char_len(&block) <= chunk_size {
        return vec![block];
    }
    let label = role_label(&message.role).unwrap_or(message.role.as_str());
    let content_chunk_size = chunk_size.saturating_sub(char_len(label) + 2).max(1);
    let content_chunk_overlap = chunk_overlap.saturating_sub(char_len(label) + 2);
    split_text_chunks(message.content.trim(), content_chunk_size, content_chunk_overlap)
        .into_iter()
        .filter(|piece| !piece.trim().is_empty())
        .map(|piece| format!("{}: {}", label, piece.trim()))
        .collect()
}

fn joined_length(blocks: &[(i64, String)]) -> usize {
    if blocks.is_empty() {
        return 0;
    }
    blocks.iter().map(|(_, text)| char_len(text)).sum::<usize>() + blocks.len().saturating_sub(1) * 2
}

fn build_overlap_tail(blocks: &[(i64, String)], chunk_overlap: usize) -> Vec<(i64, String)> {
    if blocks.is_empty() || chunk_overlap == 0 {
        return Vec::new();
    }
    let mut tail: Vec<(i64, String)> = Vec::new();
    let mut current_length = 0usize;
    for (turn_index, text) in blocks.iter().rev() {
        let item_length = char_len(text);
        let projected = if tail.is_empty() {
            item_length
        } else {
            current_length + 2 + item_length
        };
        if !tail.is_empty() && projected > chunk_overlap {
            break;
        }
        tail.push((*turn_index, text.clone()));
        current_length = projected;
        if current_length >= chunk_overlap {
            break;
        }
    }
    if tail.is_empty() {
        if let Some(last) = blocks.last() {
            tail.push(last.clone());
        }
    }
    tail.reverse();
    tail
}

fn build_chunk(blocks: &[(i64, String)], chunk_index: usize) -> ConversationChunk {
    let turn_indexes: Vec<i64> = blocks.iter().map(|(turn_index, _)| *turn_index).collect();
    let unique: std::collections::HashSet<i64> = turn_indexes.iter().copied().collect();
    ConversationChunk {
        chunk_index,
        text: blocks
            .iter()
            .map(|(_, text)| text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n"),
        turn_start: turn_indexes.iter().copied().min().unwrap_or(0),
        turn_end: turn_indexes.iter().copied().max().unwrap_or(0),
        message_count: unique.len(),
    }
}

pub fn build_conversation_chunks(
    messages: &[ConversationMessage],
    chunk_size: usize,
    chunk_overlap: usize,
    include_system: bool,
) -> Vec<ConversationChunk> {
    let turns = split_conversation_turns(messages, include_system);
    if turns.is_empty() {
        return Vec::new();
    }
    let chunk_size = chunk_size.max(1);
    let mut chunks: Vec<ConversationChunk> = Vec::new();
    let mut current_blocks: Vec<(i64, String)> = Vec::new();
    for (turn_index, turn) in turns.iter().enumerate() {
        let turn_index = turn_index as i64;
        for message in turn {
            for block in split_message_into_blocks(message, chunk_size, chunk_overlap) {
                let block_length = char_len(&block);
                let mut candidate = current_blocks.clone();
                candidate.push((turn_index, block.clone()));
                if !current_blocks.is_empty() && joined_length(&candidate) > chunk_size {
                    chunks.push(build_chunk(&current_blocks, chunks.len()));
                    current_blocks = build_overlap_tail(&current_blocks, chunk_overlap);
                    let mut overflow = current_blocks.clone();
                    overflow.push((turn_index, block.clone()));
                    if !current_blocks.is_empty() && joined_length(&overflow) > chunk_size {
                        current_blocks.clear();
                    }
                }
                if current_blocks.is_empty() && block_length > chunk_size {
                    chunks.push(ConversationChunk {
                        chunk_index: chunks.len(),
                        text: block,
                        turn_start: turn_index,
                        turn_end: turn_index,
                        message_count: 1,
                    });
                    continue;
                }
                current_blocks.push((turn_index, block));
            }
        }
    }
    if !current_blocks.is_empty() {
        chunks.push(build_chunk(&current_blocks, chunks.len()));
    }
    chunks
}

pub fn user_assistant_messages(query: &str, answer: &str) -> Vec<ConversationMessage> {
    [
        ("user", query),
        ("assistant", answer),
    ]
    .into_iter()
    .filter_map(|(role, content)| normalize_conversation_message(role, content, false))
    .collect()
}

pub fn build_semantic_memory_chunks<F>(
    messages: &[ConversationMessage],
    mut should_split_topic: F,
    mut summarize_chunk: impl FnMut(&str) -> (String, String),
    token_limit: usize,
    include_system: bool,
) -> Vec<SemanticMemoryChunk>
where
    F: FnMut(&str, &str) -> bool,
{
    let turns = split_conversation_turns(messages, include_system);
    if turns.is_empty() {
        return Vec::new();
    }
    let token_limit = token_limit.max(1);
    let mut chunks: Vec<SemanticMemoryChunk> = Vec::new();
    let mut buffer_turns: Vec<Vec<ConversationMessage>> = Vec::new();
    let mut buffer_start = 0i64;

    let mut flush = |reason: &str,
                     buffer_turns: &mut Vec<Vec<ConversationMessage>>,
                     buffer_start: i64,
                     chunks: &mut Vec<SemanticMemoryChunk>,
                     summarize_chunk: &mut dyn FnMut(&str) -> (String, String)| {
        if buffer_turns.is_empty() {
            return;
        }
        let transcript = render_long_term_memory_transcript(buffer_turns);
        let (mut topic, mut summary) = summarize_chunk(&transcript);
        if summary.trim().is_empty() {
            summary = transcript.chars().take(120).collect::<String>().trim().to_string();
        }
        if topic.trim().is_empty() {
            topic = summary.clone();
        }
        chunks.push(SemanticMemoryChunk {
            chunk_index: chunks.len(),
            text: format_long_term_memory_chunk(&summary, &transcript, &topic),
            topic,
            summary,
            transcript,
            turn_start: buffer_start,
            turn_end: buffer_start + buffer_turns.len() as i64 - 1,
            message_count: buffer_turns.len(),
            trigger_reason: reason.to_string(),
        });
        buffer_turns.clear();
    };

    let mut push_single_turn = |turn: &[ConversationMessage],
                                turn_index: i64,
                                reason: &str,
                                chunks: &mut Vec<SemanticMemoryChunk>,
                                summarize_chunk: &mut dyn FnMut(&str) -> (String, String)| {
        let transcript = render_long_term_memory_transcript(&[turn.to_vec()]);
        let (mut topic, mut summary) = summarize_chunk(&transcript);
        if summary.trim().is_empty() {
            summary = transcript.chars().take(120).collect::<String>().trim().to_string();
        }
        if topic.trim().is_empty() {
            topic = summary.clone();
        }
        chunks.push(SemanticMemoryChunk {
            chunk_index: chunks.len(),
            text: format_long_term_memory_chunk(&summary, &transcript, &topic),
            topic,
            summary,
            transcript,
            turn_start: turn_index,
            turn_end: turn_index,
            message_count: 1,
            trigger_reason: reason.to_string(),
        });
    };

    for (turn_index, turn) in turns.iter().enumerate() {
        let turn_index = turn_index as i64;
        let turn_text = render_long_term_memory_transcript(&[turn.clone()]);
        if turn_text.trim().is_empty() {
            continue;
        }
        if buffer_turns.is_empty() {
            if estimate_text_tokens(&turn_text) > token_limit {
                push_single_turn(turn, turn_index, "token_limit", &mut chunks, &mut summarize_chunk);
                continue;
            }
            buffer_turns = vec![turn.clone()];
            buffer_start = turn_index;
            continue;
        }
        let current_text = render_long_term_memory_transcript(&buffer_turns);
        let candidate_text = format!("{current_text}\n\n{turn_text}").trim().to_string();
        if should_split_topic(&current_text, &turn_text) {
            flush(
                "topic_change",
                &mut buffer_turns,
                buffer_start,
                &mut chunks,
                &mut summarize_chunk,
            );
            if estimate_text_tokens(&turn_text) > token_limit {
                push_single_turn(turn, turn_index, "topic_change", &mut chunks, &mut summarize_chunk);
                continue;
            }
            buffer_turns = vec![turn.clone()];
            buffer_start = turn_index;
            continue;
        }
        if estimate_text_tokens(&candidate_text) > token_limit {
            flush(
                "token_limit",
                &mut buffer_turns,
                buffer_start,
                &mut chunks,
                &mut summarize_chunk,
            );
            if estimate_text_tokens(&turn_text) > token_limit {
                push_single_turn(turn, turn_index, "token_limit", &mut chunks, &mut summarize_chunk);
                continue;
            }
            buffer_turns = vec![turn.clone()];
            buffer_start = turn_index;
            continue;
        }
        buffer_turns.push(turn.clone());
    }
    flush(
        "flush",
        &mut buffer_turns,
        buffer_start,
        &mut chunks,
        &mut summarize_chunk,
    );
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversation_chunks_keep_short_turn() {
        let messages = user_assistant_messages("报销时限是什么", "三个工作日");
        let chunks = build_conversation_chunks(&messages, CHAT_CHUNK_SIZE, CHAT_CHUNK_OVERLAP, false);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].turn_start, 0);
        assert_eq!(chunks[0].turn_end, 0);
        assert!(chunks[0].text.contains("用户: 报销时限是什么"));
        assert!(chunks[0].text.contains("助手: 三个工作日"));
        assert!(chunks[0].text.contains("\n\n"));
    }

    #[test]
    fn conversation_chunks_split_long_assistant() {
        let answer = "甲".repeat(600);
        let messages = user_assistant_messages("请详细说明", &answer);
        let chunks = build_conversation_chunks(&messages, CHAT_CHUNK_SIZE, CHAT_CHUNK_OVERLAP, false);
        assert!(chunks.len() >= 2, "长回答应切成多块, got {}", chunks.len());
        assert!(chunks.iter().all(|chunk| chunk.turn_start == 0 && chunk.turn_end == 0));
        assert!(chunks.iter().any(|chunk| chunk.text.contains("用户: 请详细说明")));
    }

    #[test]
    fn token_estimate_counts_cjk_and_words() {
        assert_eq!(estimate_text_tokens("hello_world 报销"), 3);
        assert_eq!(estimate_text_tokens(""), 0);
    }

    #[test]
    fn continuation_cues_skip_topic_split() {
        assert!(high_confidence_topic_continuation("报销标准", "那住宿费呢"));
        assert!(high_confidence_topic_continuation(
            "差旅报销需要发票和行程单",
            "发票和行程单必须原件"
        ));
        assert!(!high_confidence_topic_continuation("差旅报销", "下周产品发布计划"));
    }

    #[test]
    fn semantic_chunks_split_on_token_limit() {
        let mut messages = Vec::new();
        for index in 0..8 {
            messages.push(ConversationMessage {
                role: "user".into(),
                content: format!("问题{index}"),
            });
            messages.push(ConversationMessage {
                role: "assistant".into(),
                content: format!("答复{index}"),
            });
        }
        let chunks = build_semantic_memory_chunks(
            &messages,
            |_current, _next| false,
            |transcript| ("主题".into(), transcript.chars().take(20).collect()),
            40,
            false,
        );
        assert!(chunks.len() >= 2, "token 上限应切成多块, got {}", chunks.len());
        assert!(chunks.iter().any(|chunk| chunk.trigger_reason == "token_limit"));
        assert_eq!(chunks.last().unwrap().trigger_reason, "flush");
    }

    #[test]
    fn semantic_chunks_split_on_topic_change() {
        let messages = vec![
            ConversationMessage {
                role: "user".into(),
                content: "差旅报销怎么走".into(),
            },
            ConversationMessage {
                role: "assistant".into(),
                content: "先提交发票".into(),
            },
            ConversationMessage {
                role: "user".into(),
                content: "服务器什么时候上线".into(),
            },
            ConversationMessage {
                role: "assistant".into(),
                content: "周五发布".into(),
            },
        ];
        let chunks = build_semantic_memory_chunks(
            &messages,
            |current, next| !high_confidence_topic_continuation(current, next),
            |transcript| ("t".into(), transcript.chars().take(12).collect()),
            700,
            false,
        );
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].trigger_reason, "topic_change");
        assert_eq!(chunks[0].turn_start, 0);
        assert_eq!(chunks[0].turn_end, 0);
        assert_eq!(chunks[1].turn_start, 1);
        assert_eq!(chunks[1].turn_end, 1);
    }
}
