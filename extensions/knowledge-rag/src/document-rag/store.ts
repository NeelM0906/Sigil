/**
 * Document storage with filesystem-based persistence.
 * Stores documents, chunks, and embeddings with per-user isolation.
 */

import fs from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";

import type { Chunk } from "./chunker.js";

// ============================================================================
// Types
// ============================================================================

export interface StoredDocument {
  id: string;
  userId: string;
  filename: string;
  format: string;
  uploadedAt: string;
  textPath: string;
  embeddingsPath: string;
  wordCount: number;
  chunkCount: number;
  sizeBytes: number;
}

export interface ChunkWithEmbedding {
  chunk: Chunk;
  embedding: number[];
}

export interface DocumentIndex {
  version: number;
  documents: StoredDocument[];
}

// ============================================================================
// Document Store
// ============================================================================

export class DocumentStore {
  private readonly storagePath: string;
  private readonly documentsDir: string;
  private readonly embeddingsDir: string;
  private readonly indexPath: string;
  private index: DocumentIndex;

  constructor(storagePath: string) {
    this.storagePath = storagePath;
    this.documentsDir = path.join(storagePath, "documents");
    this.embeddingsDir = path.join(storagePath, "embeddings");
    this.indexPath = path.join(storagePath, "index.json");

    // Ensure directories exist
    this.ensureDirectories();

    // Load or create index
    this.index = this.loadIndex();
  }

  private ensureDirectories(): void {
    for (const dir of [this.storagePath, this.documentsDir, this.embeddingsDir]) {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
    }
  }

  private loadIndex(): DocumentIndex {
    if (fs.existsSync(this.indexPath)) {
      try {
        const data = fs.readFileSync(this.indexPath, "utf-8");
        return JSON.parse(data) as DocumentIndex;
      } catch {
        console.warn("[knowledge-rag] Failed to load index, creating new one");
      }
    }
    return { version: 1, documents: [] };
  }

  private saveIndex(): void {
    fs.writeFileSync(this.indexPath, JSON.stringify(this.index, null, 2));
  }

  /**
   * Store a new document with its text and embeddings
   */
  async storeDocument(params: {
    userId: string;
    filename: string;
    format: string;
    text: string;
    chunks: ChunkWithEmbedding[];
    sizeBytes: number;
  }): Promise<StoredDocument> {
    const { userId, filename, format, text, chunks, sizeBytes } = params;

    const docId = `${userId}_${Date.now()}_${randomUUID().slice(0, 8)}`;
    const safeFilename = filename.replace(/[^a-zA-Z0-9._-]/g, "_");

    const textPath = path.join(this.documentsDir, `${docId}_${safeFilename}.txt`);
    const embeddingsPath = path.join(this.embeddingsDir, `${docId}.json`);

    // Save text
    fs.writeFileSync(textPath, text, "utf-8");

    // Save embeddings
    const embeddingsData = chunks.map((c) => ({
      index: c.chunk.index,
      text: c.chunk.text,
      startWord: c.chunk.startWord,
      endWord: c.chunk.endWord,
      embedding: c.embedding,
    }));
    fs.writeFileSync(embeddingsPath, JSON.stringify(embeddingsData));

    // Create document record
    const doc: StoredDocument = {
      id: docId,
      userId,
      filename,
      format,
      uploadedAt: new Date().toISOString(),
      textPath,
      embeddingsPath,
      wordCount: text.split(/\s+/).filter((w) => w.length > 0).length,
      chunkCount: chunks.length,
      sizeBytes,
    };

    // Add to index
    this.index.documents.push(doc);
    this.saveIndex();

    return doc;
  }

  /**
   * Get all documents for a user
   */
  getUserDocuments(userId: string): StoredDocument[] {
    return this.index.documents.filter((d) => d.userId === userId);
  }

  /**
   * Get a specific document by ID
   */
  getDocument(docId: string): StoredDocument | null {
    return this.index.documents.find((d) => d.id === docId) ?? null;
  }

  /**
   * Get document text content
   */
  getDocumentText(docId: string): string | null {
    const doc = this.getDocument(docId);
    if (!doc || !fs.existsSync(doc.textPath)) {
      return null;
    }
    return fs.readFileSync(doc.textPath, "utf-8");
  }

  /**
   * Get document chunks with embeddings
   */
  getDocumentEmbeddings(docId: string): ChunkWithEmbedding[] | null {
    const doc = this.getDocument(docId);
    if (!doc || !fs.existsSync(doc.embeddingsPath)) {
      return null;
    }

    const data = JSON.parse(fs.readFileSync(doc.embeddingsPath, "utf-8"));
    return data.map(
      (item: { index: number; text: string; startWord: number; endWord: number; embedding: number[] }) => ({
        chunk: {
          index: item.index,
          text: item.text,
          startWord: item.startWord,
          endWord: item.endWord,
        },
        embedding: item.embedding,
      })
    );
  }

  /**
   * Delete a document
   */
  deleteDocument(docId: string): boolean {
    const doc = this.getDocument(docId);
    if (!doc) {
      return false;
    }

    // Remove files
    if (fs.existsSync(doc.textPath)) {
      fs.unlinkSync(doc.textPath);
    }
    if (fs.existsSync(doc.embeddingsPath)) {
      fs.unlinkSync(doc.embeddingsPath);
    }

    // Remove from index
    this.index.documents = this.index.documents.filter((d) => d.id !== docId);
    this.saveIndex();

    return true;
  }

  /**
   * Get all documents (for search indexing)
   */
  getAllDocuments(): StoredDocument[] {
    return [...this.index.documents];
  }

  /**
   * Get total document count
   */
  getDocumentCount(): number {
    return this.index.documents.length;
  }

  /**
   * Get user document count
   */
  getUserDocumentCount(userId: string): number {
    return this.index.documents.filter((d) => d.userId === userId).length;
  }
}
