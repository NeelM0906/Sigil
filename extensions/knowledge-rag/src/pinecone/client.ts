/**
 * Pinecone knowledge base client.
 * Provides always-available access to external knowledge.
 */

import type { EmbeddingService } from "../embeddings.js";

// ============================================================================
// Types
// ============================================================================

export interface PineconeConfig {
  apiKey: string;
  indexName: string;
  namespace?: string;
  topK?: number;
  minScore?: number;
}

export interface KnowledgeResult {
  id: string;
  text: string;
  score: number;
  metadata?: Record<string, unknown>;
}

// ============================================================================
// Pinecone Client
// ============================================================================

export class PineconeKnowledgeBase {
  private config: PineconeConfig;
  private embeddings: EmbeddingService;
  private pinecone: Awaited<ReturnType<typeof this.initPinecone>> | null = null;
  private initPromise: Promise<void> | null = null;

  constructor(config: PineconeConfig, embeddings: EmbeddingService) {
    this.config = config;
    this.embeddings = embeddings;
  }

  private async initPinecone() {
    const { Pinecone } = await import("@pinecone-database/pinecone");
    return new Pinecone({ apiKey: this.config.apiKey });
  }

  private async ensureInitialized(): Promise<void> {
    if (this.pinecone) return;
    if (this.initPromise) return this.initPromise;

    this.initPromise = (async () => {
      this.pinecone = await this.initPinecone();
      console.log(`[knowledge-rag] Pinecone client initialized for index: ${this.config.indexName}`);
    })();

    return this.initPromise;
  }

  /**
   * Query the knowledge base
   */
  async query(queryText: string, options?: { topK?: number; minScore?: number }): Promise<KnowledgeResult[]> {
    await this.ensureInitialized();

    const topK = options?.topK ?? this.config.topK ?? 5;
    const minScore = options?.minScore ?? this.config.minScore ?? 0.3;

    // Generate embedding for query
    const queryEmbedding = await this.embeddings.embed(queryText);

    // Query Pinecone
    const index = this.pinecone!.index(this.config.indexName);
    const queryOptions: {
      vector: number[];
      topK: number;
      includeMetadata: boolean;
      namespace?: string;
    } = {
      vector: queryEmbedding,
      topK,
      includeMetadata: true,
    };

    if (this.config.namespace) {
      queryOptions.namespace = this.config.namespace;
    }

    const results = await index.query(queryOptions);

    if (!results.matches || results.matches.length === 0) {
      return [];
    }

    // Filter by minimum score and format results
    return results.matches
      .filter((match) => (match.score ?? 0) >= minScore)
      .map((match) => ({
        id: match.id,
        text: (match.metadata?.text as string) ?? "",
        score: match.score ?? 0,
        metadata: match.metadata as Record<string, unknown> | undefined,
      }));
  }

  /**
   * Check if the knowledge base is available
   */
  async isAvailable(): Promise<boolean> {
    try {
      await this.ensureInitialized();
      const index = this.pinecone!.index(this.config.indexName);
      await index.describeIndexStats();
      return true;
    } catch (error) {
      console.warn(`[knowledge-rag] Pinecone not available: ${error}`);
      return false;
    }
  }

  /**
   * Get index statistics
   */
  async getStats(): Promise<{ vectorCount: number; dimension: number } | null> {
    try {
      await this.ensureInitialized();
      const index = this.pinecone!.index(this.config.indexName);
      const stats = await index.describeIndexStats();
      return {
        vectorCount: stats.totalRecordCount ?? 0,
        dimension: stats.dimension ?? 0,
      };
    } catch {
      return null;
    }
  }
}

/**
 * Format knowledge results for injection into context
 */
export function formatKnowledgeContext(results: KnowledgeResult[], maxLength = 2000): string {
  if (results.length === 0) {
    return "";
  }

  let context = "<knowledge-base-context>\n";
  let currentLength = context.length;

  for (const result of results) {
    const entry = `[Relevance: ${result.score.toFixed(2)}] ${result.text}\n\n`;

    if (currentLength + entry.length > maxLength) {
      // Truncate if necessary
      const remaining = maxLength - currentLength - 50;
      if (remaining > 100) {
        context += entry.slice(0, remaining) + "...\n";
      }
      break;
    }

    context += entry;
    currentLength += entry.length;
  }

  context += "</knowledge-base-context>";
  return context;
}
