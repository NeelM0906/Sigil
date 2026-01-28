/**
 * Sigil Knowledge RAG Plugin
 *
 * Provides:
 * 1. Document RAG - Upload, store, and search documents with hybrid search
 * 2. Knowledge Base - Always-available Pinecone knowledge base integration
 */

import { Type } from "@sinclair/typebox";
import path from "node:path";
import os from "node:os";

import type { SigilPluginApi } from "sigil/plugin-sdk";

import { knowledgeRagConfigSchema, DEFAULT_CONFIG, type KnowledgeRagConfig } from "./src/config.js";
import { createEmbeddingService, type EmbeddingService } from "./src/embeddings.js";
import { extractText, isSupportedFormat, getFileMetadata } from "./src/document-rag/extractor.js";
import { chunkText, countWords } from "./src/document-rag/chunker.js";
import { DocumentStore, type ChunkWithEmbedding } from "./src/document-rag/store.js";
import { HybridSearcher, type SearchResult } from "./src/document-rag/search.js";
import { PineconeKnowledgeBase, formatKnowledgeContext, type KnowledgeResult } from "./src/pinecone/client.js";

// ============================================================================
// Plugin State
// ============================================================================

let embeddingService: EmbeddingService | null = null;
let documentStore: DocumentStore | null = null;
let hybridSearcher: HybridSearcher | null = null;
let pineconeKB: PineconeKnowledgeBase | null = null;

// ============================================================================
// Plugin Definition
// ============================================================================

const knowledgeRagPlugin = {
  id: "knowledge-rag",
  name: "Knowledge RAG",
  description: "Hybrid RAG document retrieval and Pinecone knowledge base integration",
  kind: "memory" as const,
  configSchema: knowledgeRagConfigSchema,

  async register(api: SigilPluginApi) {
    const rawConfig = api.pluginConfig ?? {};
    const config: KnowledgeRagConfig = {
      ...DEFAULT_CONFIG,
      ...rawConfig,
      documents: { ...DEFAULT_CONFIG.documents, ...(rawConfig.documents ?? {}) },
      pinecone: { ...DEFAULT_CONFIG.pinecone, ...(rawConfig.pinecone ?? {}) },
      embedding: { ...DEFAULT_CONFIG.embedding, ...(rawConfig.embedding ?? {}) },
    };

    api.logger.info("[knowledge-rag] Initializing plugin...");

    // Initialize embedding service
    try {
      embeddingService = await createEmbeddingService(config.embedding);
      api.logger.info(`[knowledge-rag] Embedding service ready (provider: ${config.embedding?.provider ?? "local"})`);
    } catch (error) {
      api.logger.error(`[knowledge-rag] Failed to initialize embedding service: ${error}`);
      return;
    }

    // Initialize document store and searcher
    if (config.documents?.enabled !== false) {
      const storagePath = config.documents?.storagePath
        ? api.resolvePath(config.documents.storagePath)
        : path.join(os.homedir(), ".sigil", "knowledge-rag", "documents");

      documentStore = new DocumentStore(storagePath);
      hybridSearcher = new HybridSearcher(documentStore, embeddingService);
      hybridSearcher.rebuildIndex();

      api.logger.info(`[knowledge-rag] Document store ready (${documentStore.getDocumentCount()} documents)`);

      // Register document tools
      registerDocumentTools(api, config);
    }

    // Initialize Pinecone knowledge base
    if (config.pinecone?.enabled !== false && config.pinecone?.indexName) {
      const apiKey = config.pinecone.apiKey ?? process.env.PINECONE_API_KEY;

      if (apiKey) {
        pineconeKB = new PineconeKnowledgeBase(
          {
            apiKey,
            indexName: config.pinecone.indexName,
            namespace: config.pinecone.namespace,
            topK: config.pinecone.topK,
            minScore: config.pinecone.minScore,
          },
          embeddingService
        );

        api.logger.info(`[knowledge-rag] Pinecone KB configured (index: ${config.pinecone.indexName})`);

        // Register knowledge base tools
        registerKnowledgeTools(api, config);

        // Register auto-recall lifecycle hook
        if (config.pinecone.autoRecall !== false) {
          registerAutoRecall(api, config);
        }
      } else {
        api.logger.warn("[knowledge-rag] Pinecone enabled but no API key provided");
      }
    }

    api.logger.info("[knowledge-rag] Plugin registered successfully");
  },
};

// ============================================================================
// Document Tools
// ============================================================================

function registerDocumentTools(api: SigilPluginApi, config: KnowledgeRagConfig) {
  // Tool: Upload/store document
  api.registerTool(
    {
      name: "document_upload",
      label: "Upload Document",
      description:
        "Store an uploaded document for later retrieval. Extracts text, generates embeddings, and enables hybrid search. Supports PDF, DOCX, TXT, MD files.",
      parameters: Type.Object({
        file_path: Type.String({ description: "Path to the uploaded file" }),
        user_id: Type.String({ description: "User ID for document isolation" }),
      }),
      async execute(_toolCallId, params) {
        const { file_path, user_id } = params as { file_path: string; user_id: string };

        if (!documentStore || !hybridSearcher || !embeddingService) {
          return { success: false, error: "Document store not initialized" };
        }

        if (!isSupportedFormat(file_path)) {
          return {
            success: false,
            error: "Unsupported file format. Supported: PDF, DOCX, TXT, MD",
          };
        }

        try {
          // Extract text
          const text = await extractText(file_path);
          const metadata = getFileMetadata(file_path);

          // Chunk text
          const chunks = chunkText(text, {
            chunkSize: config.documents?.chunkSize,
            overlap: config.documents?.chunkOverlap,
          });

          // Generate embeddings
          const chunksWithEmbeddings: ChunkWithEmbedding[] = [];
          for (const chunk of chunks) {
            const embedding = await embeddingService.embed(chunk.text);
            chunksWithEmbeddings.push({ chunk, embedding });
          }

          // Store document
          const doc = await documentStore.storeDocument({
            userId: user_id,
            filename: metadata.filename,
            format: metadata.format ?? "unknown",
            text,
            chunks: chunksWithEmbeddings,
            sizeBytes: metadata.sizeBytes,
          });

          // Update search index
          hybridSearcher.addDocumentToIndex(doc, chunksWithEmbeddings);

          return {
            success: true,
            document: {
              id: doc.id,
              filename: doc.filename,
              wordCount: doc.wordCount,
              chunkCount: doc.chunkCount,
            },
            message: `Document "${doc.filename}" stored with ${doc.chunkCount} searchable chunks.`,
          };
        } catch (error) {
          return { success: false, error: String(error) };
        }
      },
    },
    { names: ["document_upload"] }
  );

  // Tool: Search documents
  api.registerTool(
    {
      name: "document_search",
      label: "Search Documents",
      description:
        "Search through stored documents using hybrid search (keyword + semantic). Use when user asks about their uploaded documents or wants information from files they've shared.",
      parameters: Type.Object({
        query: Type.String({ description: "Search query" }),
        user_id: Type.String({ description: "User ID" }),
        top_k: Type.Optional(Type.Number({ description: "Number of results (default: 5)" })),
      }),
      async execute(_toolCallId, params) {
        const { query, user_id, top_k = 5 } = params as {
          query: string;
          user_id: string;
          top_k?: number;
        };

        if (!hybridSearcher) {
          return { found: false, error: "Document search not initialized" };
        }

        try {
          const results = await hybridSearcher.search(query, user_id, {
            topK: top_k,
            keywordWeight: config.documents?.hybridWeights?.keyword,
            semanticWeight: config.documents?.hybridWeights?.semantic,
          });

          if (results.length === 0) {
            return {
              found: false,
              message: "No relevant information found in your documents.",
            };
          }

          return {
            found: true,
            count: results.length,
            results: results.map((r) => ({
              filename: r.filename,
              excerpt: r.text.slice(0, 500) + (r.text.length > 500 ? "..." : ""),
              relevance: r.hybridScore.toFixed(3),
              methods: r.methods.join(" + "),
            })),
          };
        } catch (error) {
          return { found: false, error: String(error) };
        }
      },
    },
    { names: ["document_search"] }
  );

  // Tool: List documents
  api.registerTool(
    {
      name: "document_list",
      label: "List Documents",
      description: "List all stored documents for a user.",
      parameters: Type.Object({
        user_id: Type.String({ description: "User ID" }),
      }),
      async execute(_toolCallId, params) {
        const { user_id } = params as { user_id: string };

        if (!documentStore) {
          return { count: 0, documents: [] };
        }

        const docs = documentStore.getUserDocuments(user_id);
        return {
          count: docs.length,
          documents: docs.map((d) => ({
            id: d.id,
            filename: d.filename,
            format: d.format,
            uploadedAt: d.uploadedAt,
            wordCount: d.wordCount,
            chunkCount: d.chunkCount,
          })),
        };
      },
    },
    { names: ["document_list"] }
  );
}

// ============================================================================
// Knowledge Base Tools
// ============================================================================

function registerKnowledgeTools(api: SigilPluginApi, config: KnowledgeRagConfig) {
  // Tool: Query knowledge base
  api.registerTool(
    {
      name: "knowledge_search",
      label: "Search Knowledge Base",
      description:
        "Search the external knowledge base (Pinecone). Use when the query relates to domain-specific knowledge that may be in the knowledge base. This is always available alongside other knowledge sources.",
      parameters: Type.Object({
        query: Type.String({ description: "Search query" }),
        top_k: Type.Optional(Type.Number({ description: "Number of results (default: 5)" })),
      }),
      async execute(_toolCallId, params) {
        const { query, top_k } = params as { query: string; top_k?: number };

        if (!pineconeKB) {
          return { found: false, error: "Knowledge base not configured" };
        }

        try {
          const results = await pineconeKB.query(query, {
            topK: top_k ?? config.pinecone?.topK,
            minScore: config.pinecone?.minScore,
          });

          if (results.length === 0) {
            return {
              found: false,
              message: "No relevant information found in the knowledge base.",
            };
          }

          const avgScore = results.reduce((sum, r) => sum + r.score, 0) / results.length;

          return {
            found: true,
            count: results.length,
            averageRelevance: avgScore.toFixed(3),
            results: results.map((r) => ({
              text: r.text.slice(0, 500) + (r.text.length > 500 ? "..." : ""),
              relevance: r.score.toFixed(3),
            })),
            lowRelevanceWarning:
              avgScore < 0.3
                ? "Results have low relevance and may not directly answer the question."
                : undefined,
          };
        } catch (error) {
          return { found: false, error: String(error) };
        }
      },
    },
    { names: ["knowledge_search"] }
  );

  // Tool: Check knowledge base status
  api.registerTool(
    {
      name: "knowledge_status",
      label: "Knowledge Base Status",
      description: "Check if the knowledge base is available and get statistics.",
      parameters: Type.Object({}),
      async execute() {
        if (!pineconeKB) {
          return { available: false, error: "Knowledge base not configured" };
        }

        const available = await pineconeKB.isAvailable();
        const stats = await pineconeKB.getStats();

        return {
          available,
          indexName: config.pinecone?.indexName,
          stats: stats ?? undefined,
        };
      },
    },
    { names: ["knowledge_status"] }
  );
}

// ============================================================================
// Auto-Recall Lifecycle Hook
// ============================================================================

function registerAutoRecall(api: SigilPluginApi, config: KnowledgeRagConfig) {
  api.registerLifecycleHook("beforeAgentTurn", async (ctx) => {
    if (!pineconeKB) return;

    // Get the user's message
    const lastUserMessage = ctx.messages
      .slice()
      .reverse()
      .find((m) => m.role === "user");

    if (!lastUserMessage || typeof lastUserMessage.content !== "string") {
      return;
    }

    const query = lastUserMessage.content;

    // Skip very short queries or commands
    if (query.length < 10 || query.startsWith("/")) {
      return;
    }

    try {
      // Query knowledge base
      const results = await pineconeKB.query(query, {
        topK: 3,
        minScore: config.pinecone?.minScore ?? 0.35,
      });

      if (results.length === 0) {
        return;
      }

      // Format and inject context
      const context = formatKnowledgeContext(results, 1500);

      if (context) {
        // Inject as system message
        ctx.messages.push({
          role: "system",
          content: `Relevant knowledge base context for this query:\n${context}`,
        });

        api.logger.debug(
          `[knowledge-rag] Auto-recalled ${results.length} KB results for query`
        );
      }
    } catch (error) {
      api.logger.warn(`[knowledge-rag] Auto-recall failed: ${error}`);
    }
  });
}

export default knowledgeRagPlugin;
