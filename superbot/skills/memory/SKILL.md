---
name: memory
description: Vector-based memory system with semantic recall.
always: true
---

# Memory

## How It Works

superbot uses a vector-based memory system that automatically stores and retrieves relevant context:

- **Automatic Storage**: Conversations are stored in real-time to vector database
- **Semantic Retrieval**: Relevant memories are retrieved based on meaning, not just keywords
- **No Manual Management**: The system handles storage and retrieval automatically

## Context Provided

When you respond to user messages, the system automatically retrieves:
- **Relevant Facts**: Key facts from past conversations
- **Conversation Context**: Recent related dialogue
- **Context Understanding**: LLM-powered understanding of user intent

## Best Practices

1. **Don't write to files** for memory - the system handles this automatically
2. **Focus on the conversation** - relevant context will be provided
3. **Important facts** will be extracted and stored automatically

## No Action Required

You don't need to manage memory manually. Just focus on helping the user - the memory system works in the background.
