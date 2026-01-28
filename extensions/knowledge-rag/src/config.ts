import { Type, type Static } from "@sinclair/typebox";

export const knowledgeRagConfigSchema = Type.Object({
  // Document RAG settings
  documents: Type.Optional(
    Type.Object({
      enabled: Type.Optional(Type.Boolean({ default: true })),
      storagePath: Type.Optional(
        Type.String({
          description: "Path to store documents (default: ~/.sigil/knowledge-rag/documents)",
        })
      ),
      chunkSize: Type.Optional(
        Type.Number({
          default: 500,
          description: "Number of words per chunk",
        })
      ),
      chunkOverlap: Type.Optional(
        Type.Number({
          default: 100,
          description: "Number of overlapping words between chunks",
        })
      ),
      hybridWeights: Type.Optional(
        Type.Object({
          keyword: Type.Optional(Type.Number({ default: 0.4 })),
          semantic: Type.Optional(Type.Number({ default: 0.6 })),
        })
      ),
    })
  ),

  // Pinecone knowledge base settings
  pinecone: Type.Optional(
    Type.Object({
      enabled: Type.Optional(Type.Boolean({ default: true })),
      apiKey: Type.Optional(
        Type.String({
          description: "Pinecone API key (or set PINECONE_API_KEY env var)",
        })
      ),
      indexName: Type.String({
        description: "Pinecone index name to query",
      }),
      namespace: Type.Optional(
        Type.String({
          description: "Optional namespace within the index",
        })
      ),
      topK: Type.Optional(
        Type.Number({
          default: 5,
          description: "Number of results to return",
        })
      ),
      minScore: Type.Optional(
        Type.Number({
          default: 0.3,
          description: "Minimum similarity score threshold",
        })
      ),
      autoRecall: Type.Optional(
        Type.Boolean({
          default: true,
          description: "Automatically inject relevant KB context before agent turns",
        })
      ),
    })
  ),

  // Embedding settings (for both document RAG and Pinecone queries)
  embedding: Type.Optional(
    Type.Object({
      provider: Type.Optional(
        Type.Union([Type.Literal("local"), Type.Literal("openai")], {
          default: "local",
          description: "Use local (Xenova) or OpenAI embeddings",
        })
      ),
      openaiApiKey: Type.Optional(
        Type.String({
          description: "OpenAI API key for embeddings (if provider is 'openai')",
        })
      ),
      openaiModel: Type.Optional(
        Type.String({
          default: "text-embedding-ada-002",
          description: "OpenAI embedding model",
        })
      ),
      localModel: Type.Optional(
        Type.String({
          default: "Xenova/all-MiniLM-L6-v2",
          description: "Local embedding model (transformers.js)",
        })
      ),
    })
  ),
});

export type KnowledgeRagConfig = Static<typeof knowledgeRagConfigSchema>;

export const DEFAULT_CONFIG: Partial<KnowledgeRagConfig> = {
  documents: {
    enabled: true,
    chunkSize: 500,
    chunkOverlap: 100,
    hybridWeights: {
      keyword: 0.4,
      semantic: 0.6,
    },
  },
  pinecone: {
    enabled: true,
    topK: 5,
    minScore: 0.3,
    autoRecall: true,
  },
  embedding: {
    provider: "local",
    localModel: "Xenova/all-MiniLM-L6-v2",
    openaiModel: "text-embedding-ada-002",
  },
};
