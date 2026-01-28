/**
 * Embedding service supporting both local (Xenova/transformers.js) and OpenAI embeddings.
 */

import type { KnowledgeRagConfig } from "./config.js";

// Lazy-loaded embedding model for local embeddings
let localModel: Awaited<ReturnType<typeof import("@xenova/transformers").pipeline>> | null = null;
let localModelLoading: Promise<typeof localModel> | null = null;

export interface EmbeddingService {
  embed(text: string): Promise<number[]>;
  embedBatch(texts: string[]): Promise<number[][]>;
  dimension: number;
}

/**
 * Create a local embedding service using Xenova/transformers.js
 */
async function createLocalEmbeddingService(modelName: string): Promise<EmbeddingService> {
  if (!localModel) {
    if (!localModelLoading) {
      localModelLoading = (async () => {
        const { pipeline } = await import("@xenova/transformers");
        console.log(`[knowledge-rag] Loading local embedding model: ${modelName}...`);
        localModel = await pipeline("feature-extraction", modelName);
        console.log(`[knowledge-rag] Local embedding model loaded.`);
        return localModel;
      })();
    }
    await localModelLoading;
  }

  return {
    dimension: 384, // MiniLM-L6-v2 dimension

    async embed(text: string): Promise<number[]> {
      const output = await localModel!(text, { pooling: "mean", normalize: true });
      return Array.from(output.data as Float32Array);
    },

    async embedBatch(texts: string[]): Promise<number[][]> {
      const results: number[][] = [];
      for (const text of texts) {
        const output = await localModel!(text, { pooling: "mean", normalize: true });
        results.push(Array.from(output.data as Float32Array));
      }
      return results;
    },
  };
}

/**
 * Create an OpenAI embedding service
 */
function createOpenAIEmbeddingService(apiKey: string, model: string): EmbeddingService {
  // Lazy import to avoid requiring openai if not used
  let OpenAI: typeof import("openai").default;

  const getClient = async () => {
    if (!OpenAI) {
      const mod = await import("openai");
      OpenAI = mod.default;
    }
    return new OpenAI({ apiKey });
  };

  // Dimension varies by model
  const dimensions: Record<string, number> = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
  };

  return {
    dimension: dimensions[model] ?? 1536,

    async embed(text: string): Promise<number[]> {
      const client = await getClient();
      const response = await client.embeddings.create({
        model,
        input: text,
      });
      return response.data[0].embedding;
    },

    async embedBatch(texts: string[]): Promise<number[][]> {
      const client = await getClient();
      const response = await client.embeddings.create({
        model,
        input: texts,
      });
      return response.data.map((d) => d.embedding);
    },
  };
}

/**
 * Create an embedding service based on config
 */
export async function createEmbeddingService(
  config: KnowledgeRagConfig["embedding"]
): Promise<EmbeddingService> {
  const provider = config?.provider ?? "local";

  if (provider === "openai") {
    const apiKey = config?.openaiApiKey ?? process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error(
        "[knowledge-rag] OpenAI embedding provider selected but no API key provided. " +
          "Set embedding.openaiApiKey in config or OPENAI_API_KEY env var."
      );
    }
    const model = config?.openaiModel ?? "text-embedding-ada-002";
    return createOpenAIEmbeddingService(apiKey, model);
  }

  // Default to local
  const modelName = config?.localModel ?? "Xenova/all-MiniLM-L6-v2";
  return createLocalEmbeddingService(modelName);
}

/**
 * Compute cosine similarity between two vectors
 */
export function cosineSimilarity(vecA: number[], vecB: number[]): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  return denominator === 0 ? 0 : dotProduct / denominator;
}
