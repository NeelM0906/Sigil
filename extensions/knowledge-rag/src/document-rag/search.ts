/**
 * Hybrid search combining TF-IDF (keyword) and semantic (embedding) search.
 */

import type { EmbeddingService } from "../embeddings.js";
import { cosineSimilarity } from "../embeddings.js";
import type { DocumentStore, StoredDocument, ChunkWithEmbedding } from "./store.js";

// ============================================================================
// Types
// ============================================================================

export interface SearchResult {
  docId: string;
  filename: string;
  chunkIndex: number;
  text: string;
  keywordScore: number;
  semanticScore: number;
  hybridScore: number;
  methods: ("keyword" | "semantic")[];
}

export interface SearchOptions {
  topK?: number;
  minScore?: number;
  keywordWeight?: number;
  semanticWeight?: number;
}

// ============================================================================
// TF-IDF Implementation
// ============================================================================

interface TfIdfIndex {
  documents: Map<string, Map<string, number>>; // docId -> term -> tf-idf
  idf: Map<string, number>; // term -> idf
  docCount: number;
}

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2);
}

function computeTf(tokens: string[]): Map<string, number> {
  const tf = new Map<string, number>();
  for (const token of tokens) {
    tf.set(token, (tf.get(token) ?? 0) + 1);
  }
  // Normalize by document length
  const maxFreq = Math.max(...tf.values());
  for (const [term, freq] of tf) {
    tf.set(term, freq / maxFreq);
  }
  return tf;
}

function buildTfIdfIndex(documents: { id: string; text: string }[]): TfIdfIndex {
  const docCount = documents.length;
  const termDocFreq = new Map<string, number>();
  const docTerms = new Map<string, Map<string, number>>();

  // First pass: compute term frequencies and document frequencies
  for (const doc of documents) {
    const tokens = tokenize(doc.text);
    const tf = computeTf(tokens);
    docTerms.set(doc.id, tf);

    const uniqueTerms = new Set(tokens);
    for (const term of uniqueTerms) {
      termDocFreq.set(term, (termDocFreq.get(term) ?? 0) + 1);
    }
  }

  // Compute IDF
  const idf = new Map<string, number>();
  for (const [term, df] of termDocFreq) {
    idf.set(term, Math.log((docCount + 1) / (df + 1)) + 1);
  }

  // Compute TF-IDF
  const tfidfDocs = new Map<string, Map<string, number>>();
  for (const [docId, tf] of docTerms) {
    const tfidf = new Map<string, number>();
    for (const [term, tfVal] of tf) {
      const idfVal = idf.get(term) ?? 0;
      tfidf.set(term, tfVal * idfVal);
    }
    tfidfDocs.set(docId, tfidf);
  }

  return {
    documents: tfidfDocs,
    idf,
    docCount,
  };
}

function queryTfIdf(
  index: TfIdfIndex,
  query: string,
  topK: number
): { docId: string; score: number }[] {
  const queryTokens = tokenize(query);
  const queryTf = computeTf(queryTokens);

  // Compute query TF-IDF
  const queryTfIdf = new Map<string, number>();
  for (const [term, tf] of queryTf) {
    const idfVal = index.idf.get(term) ?? 0;
    queryTfIdf.set(term, tf * idfVal);
  }

  // Score documents using cosine similarity
  const scores: { docId: string; score: number }[] = [];

  for (const [docId, docTfIdf] of index.documents) {
    let dotProduct = 0;
    let queryNorm = 0;
    let docNorm = 0;

    for (const [term, qVal] of queryTfIdf) {
      const dVal = docTfIdf.get(term) ?? 0;
      dotProduct += qVal * dVal;
      queryNorm += qVal * qVal;
    }

    for (const dVal of docTfIdf.values()) {
      docNorm += dVal * dVal;
    }

    const denominator = Math.sqrt(queryNorm) * Math.sqrt(docNorm);
    const score = denominator > 0 ? dotProduct / denominator : 0;

    if (score > 0) {
      scores.push({ docId, score });
    }
  }

  return scores.sort((a, b) => b.score - a.score).slice(0, topK);
}

// ============================================================================
// Hybrid Search
// ============================================================================

export class HybridSearcher {
  private store: DocumentStore;
  private embeddings: EmbeddingService;
  private tfidfIndex: TfIdfIndex | null = null;
  private chunkCache: Map<string, ChunkWithEmbedding[]> = new Map();

  constructor(store: DocumentStore, embeddings: EmbeddingService) {
    this.store = store;
    this.embeddings = embeddings;
  }

  /**
   * Rebuild the TF-IDF index from all documents
   */
  rebuildIndex(): void {
    const documents = this.store.getAllDocuments();
    const indexDocs: { id: string; text: string }[] = [];

    for (const doc of documents) {
      const chunks = this.store.getDocumentEmbeddings(doc.id);
      if (chunks) {
        this.chunkCache.set(doc.id, chunks);
        // Index each chunk separately for finer-grained retrieval
        for (const chunk of chunks) {
          indexDocs.push({
            id: `${doc.id}:${chunk.chunk.index}`,
            text: chunk.chunk.text,
          });
        }
      }
    }

    this.tfidfIndex = buildTfIdfIndex(indexDocs);
    console.log(`[knowledge-rag] TF-IDF index built with ${indexDocs.length} chunks`);
  }

  /**
   * Perform hybrid search combining keyword and semantic search
   */
  async search(
    query: string,
    userId: string,
    options: SearchOptions = {}
  ): Promise<SearchResult[]> {
    const {
      topK = 5,
      minScore = 0.1,
      keywordWeight = 0.4,
      semanticWeight = 0.6,
    } = options;

    // Ensure index is built
    if (!this.tfidfIndex) {
      this.rebuildIndex();
    }

    const userDocs = this.store.getUserDocuments(userId);
    if (userDocs.length === 0) {
      return [];
    }

    const userDocIds = new Set(userDocs.map((d) => d.id));
    const results = new Map<string, SearchResult>();

    // 1. Keyword search (TF-IDF)
    if (this.tfidfIndex) {
      const keywordResults = queryTfIdf(this.tfidfIndex, query, topK * 2);

      for (const { docId: chunkId, score } of keywordResults) {
        const [docId, chunkIndexStr] = chunkId.split(":");
        if (!userDocIds.has(docId)) continue;

        const chunkIndex = parseInt(chunkIndexStr, 10);
        const doc = this.store.getDocument(docId);
        const chunks = this.chunkCache.get(docId);

        if (doc && chunks && chunks[chunkIndex]) {
          const key = chunkId;
          const existing = results.get(key);

          if (existing) {
            existing.keywordScore = score;
            existing.methods.push("keyword");
          } else {
            results.set(key, {
              docId,
              filename: doc.filename,
              chunkIndex,
              text: chunks[chunkIndex].chunk.text,
              keywordScore: score,
              semanticScore: 0,
              hybridScore: 0,
              methods: ["keyword"],
            });
          }
        }
      }
    }

    // 2. Semantic search (embeddings)
    const queryEmbedding = await this.embeddings.embed(query);

    for (const doc of userDocs) {
      const chunks = this.chunkCache.get(doc.id) ?? this.store.getDocumentEmbeddings(doc.id);
      if (!chunks) continue;

      for (const chunk of chunks) {
        const similarity = cosineSimilarity(queryEmbedding, chunk.embedding);

        if (similarity >= minScore) {
          const key = `${doc.id}:${chunk.chunk.index}`;
          const existing = results.get(key);

          if (existing) {
            existing.semanticScore = similarity;
            if (!existing.methods.includes("semantic")) {
              existing.methods.push("semantic");
            }
          } else {
            results.set(key, {
              docId: doc.id,
              filename: doc.filename,
              chunkIndex: chunk.chunk.index,
              text: chunk.chunk.text,
              keywordScore: 0,
              semanticScore: similarity,
              hybridScore: 0,
              methods: ["semantic"],
            });
          }
        }
      }
    }

    // 3. Compute hybrid scores
    for (const result of results.values()) {
      result.hybridScore =
        result.keywordScore * keywordWeight + result.semanticScore * semanticWeight;
    }

    // 4. Sort by hybrid score and return top K
    return Array.from(results.values())
      .filter((r) => r.hybridScore >= minScore)
      .sort((a, b) => b.hybridScore - a.hybridScore)
      .slice(0, topK);
  }

  /**
   * Add a document to the index
   */
  addDocumentToIndex(doc: StoredDocument, chunks: ChunkWithEmbedding[]): void {
    this.chunkCache.set(doc.id, chunks);
    // Rebuild index to include new document
    this.rebuildIndex();
  }

  /**
   * Remove a document from the index
   */
  removeDocumentFromIndex(docId: string): void {
    this.chunkCache.delete(docId);
    this.rebuildIndex();
  }
}
