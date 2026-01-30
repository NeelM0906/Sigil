const fs = require('fs');
const path = require('path');
const pdfParse = require('pdf-parse');
const mammoth = require('mammoth');
const natural = require('natural');
const { pipeline } = require('@xenova/transformers');

// Storage directory
const STORAGE_DIR = path.join(process.cwd(), 'document_storage');
const EMBEDDINGS_DIR = path.join(STORAGE_DIR, 'embeddings');

// Ensure directories exist
[STORAGE_DIR, EMBEDDINGS_DIR].forEach(dir => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
});

// In-memory stores
const documents = new Map();
const embeddings = new Map();
let embeddingModel = null;
let tfidf = null;

// Initialize embedding model (lazy loading)
async function getEmbeddingModel() {
  if (!embeddingModel) {
    console.log('Loading embedding model (first time only)...');
    embeddingModel = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    console.log('Embedding model loaded!');
  }
  return embeddingModel;
}

// Load existing documents on startup
function loadDocuments() {
  const indexPath = path.join(STORAGE_DIR, 'index.json');
  if (fs.existsSync(indexPath)) {
    const data = JSON.parse(fs.readFileSync(indexPath, 'utf-8'));
    data.forEach(doc => documents.set(doc.id, doc));
    console.log(`Loaded ${documents.size} documents from storage`);

    // Rebuild TF-IDF index
    rebuildTFIDF();

    // Load embeddings
    loadEmbeddings();
  }
}

// Save documents index
function saveIndex() {
  const indexPath = path.join(STORAGE_DIR, 'index.json');
  const data = Array.from(documents.values());
  fs.writeFileSync(indexPath, JSON.stringify(data, null, 2));
}

// Load embeddings
function loadEmbeddings() {
  for (const [docId, doc] of documents.entries()) {
    const embPath = path.join(EMBEDDINGS_DIR, docId + '.json');
    if (fs.existsSync(embPath)) {
      const data = JSON.parse(fs.readFileSync(embPath, 'utf-8'));
      embeddings.set(docId, data);
    }
  }
  console.log(`Loaded ${embeddings.size} document embeddings`);
}

// Rebuild TF-IDF index
function rebuildTFIDF() {
  tfidf = new natural.TfIdf();

  for (const doc of documents.values()) {
    const text = fs.readFileSync(doc.storagePath, 'utf-8');
    tfidf.addDocument(text);
  }

  console.log('TF-IDF index rebuilt');
}

// Extract text from file
async function extractText(filePath) {
  const ext = path.extname(filePath).toLowerCase();

  try {
    if (ext === '.pdf') {
      const dataBuffer = fs.readFileSync(filePath);
      const data = await pdfParse(dataBuffer);
      return data.text;
    } else if (ext === '.docx') {
      const result = await mammoth.extractRawText({ path: filePath });
      return result.value;
    } else if (ext === '.txt' || ext === '.md') {
      return fs.readFileSync(filePath, 'utf-8');
    } else {
      throw new Error('Unsupported file type: ' + ext);
    }
  } catch (error) {
    throw new Error(`Failed to extract text: ${error.message}`);
  }
}

// Chunk text into smaller pieces
function chunkText(text, chunkSize = 500, overlap = 100) {
  const words = text.split(/\s+/);
  const chunks = [];

  for (let i = 0; i < words.length; i += chunkSize - overlap) {
    const chunk = words.slice(i, i + chunkSize).join(' ');
    if (chunk.trim().length > 0) {
      chunks.push(chunk);
    }
  }

  return chunks;
}

// Generate embeddings for text
async function generateEmbedding(text) {
  const model = await getEmbeddingModel();
  const output = await model(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

// Cosine similarity
function cosineSimilarity(vecA, vecB) {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Store document with embeddings
async function storeDocument(filePath, userId) {
  const filename = path.basename(filePath);
  const text = await extractText(filePath);

  const docId = `${userId}_${Date.now()}_${filename}`;
  const storagePath = path.join(STORAGE_DIR, docId + '.txt');

  // Save text to storage
  fs.writeFileSync(storagePath, text);

  // Generate chunks and embeddings
  console.log(`Generating embeddings for ${filename}...`);
  const chunks = chunkText(text);
  const chunkEmbeddings = [];

  for (let i = 0; i < chunks.length; i++) {
    const embedding = await generateEmbedding(chunks[i]);
    chunkEmbeddings.push({
      text: chunks[i],
      embedding: embedding
    });
  }

  // Save embeddings
  const embPath = path.join(EMBEDDINGS_DIR, docId + '.json');
  fs.writeFileSync(embPath, JSON.stringify(chunkEmbeddings));
  embeddings.set(docId, chunkEmbeddings);

  // Store metadata
  const doc = {
    id: docId,
    userId: userId,
    filename: filename,
    storagePath: storagePath,
    uploadedAt: new Date().toISOString(),
    size: text.length,
    numChunks: chunks.length
  };

  documents.set(docId, doc);
  saveIndex();

  // Rebuild TF-IDF with new document
  rebuildTFIDF();

  console.log(`Stored ${filename} with ${chunks.length} chunks`);

  return doc;
}

// Keyword search using TF-IDF
function keywordSearch(query, userId, topK = 5) {
  if (!tfidf) return [];

  const results = [];
  const userDocs = Array.from(documents.values()).filter(
    doc => doc.userId === userId
  );

  tfidf.tfidfs(query, (i, measure) => {
    if (i < userDocs.length && measure > 0) {
      const doc = userDocs[i];
      results.push({
        docId: doc.id,
        filename: doc.filename,
        score: measure,
        type: 'keyword'
      });
    }
  });

  return results.sort((a, b) => b.score - a.score).slice(0, topK);
}

// Semantic search using embeddings
async function semanticSearch(query, userId, topK = 5) {
  const queryEmbedding = await generateEmbedding(query);
  const results = [];

  for (const [docId, doc] of documents.entries()) {
    if (doc.userId !== userId) continue;

    const docEmbeddings = embeddings.get(docId);
    if (!docEmbeddings) continue;

    // Find best matching chunk
    let bestScore = -1;
    let bestChunk = null;

    for (const chunk of docEmbeddings) {
      const similarity = cosineSimilarity(queryEmbedding, chunk.embedding);
      if (similarity > bestScore) {
        bestScore = similarity;
        bestChunk = chunk;
      }
    }

    if (bestScore > 0.3) { // Threshold
      results.push({
        docId: doc.id,
        filename: doc.filename,
        score: bestScore,
        text: bestChunk.text,
        type: 'semantic'
      });
    }
  }

  return results.sort((a, b) => b.score - a.score).slice(0, topK);
}

// Hybrid search: Combine keyword + semantic
async function hybridSearch(query, userId, topK = 5) {
  // Run both searches in parallel
  const [keywordResults, semanticResults] = await Promise.all([
    Promise.resolve(keywordSearch(query, userId, topK)),
    semanticSearch(query, userId, topK)
  ]);

  // Combine and deduplicate results
  const combined = new Map();

  // Add keyword results (weight: 0.4)
  for (const result of keywordResults) {
    combined.set(result.docId, {
      ...result,
      hybridScore: result.score * 0.4,
      methods: ['keyword']
    });
  }

  // Add semantic results (weight: 0.6)
  for (const result of semanticResults) {
    if (combined.has(result.docId)) {
      const existing = combined.get(result.docId);
      existing.hybridScore += result.score * 0.6;
      existing.methods.push('semantic');
      existing.text = result.text; // Use semantic chunk text
    } else {
      combined.set(result.docId, {
        ...result,
        hybridScore: result.score * 0.6,
        methods: ['semantic']
      });
    }
  }

  // Sort by hybrid score
  const finalResults = Array.from(combined.values())
    .sort((a, b) => b.hybridScore - a.hybridScore)
    .slice(0, topK);

  // Get excerpts for results
  for (const result of finalResults) {
    const doc = documents.get(result.docId);
    if (!result.text) {
      // Get text excerpt if not from semantic search
      const fullText = fs.readFileSync(doc.storagePath, 'utf-8');
      result.text = fullText.substring(0, 300) + '...';
    }
  }

  return finalResults;
}

// Get all user documents
function getUserDocuments(userId) {
  return Array.from(documents.values())
    .filter(doc => doc.userId === userId)
    .map(doc => ({
      filename: doc.filename,
      uploadedAt: doc.uploadedAt,
      size: doc.size,
      numChunks: doc.numChunks || 0
    }));
}

// Get document content
function getDocumentContent(userId, filename) {
  const doc = Array.from(documents.values()).find(
    d => d.userId === userId && d.filename === filename
  );

  if (!doc) return null;

  return fs.readFileSync(doc.storagePath, 'utf-8');
}

// Load documents on startup
loadDocuments();

// Export skill
module.exports = {
  name: 'document-store',
  description: 'Store and retrieve documents with hybrid search (keyword + semantic)',

  functions: {
    // Store a document
    store_document: {
      description: 'Store an uploaded document permanently with embeddings. Call this when user uploads a file.',
      parameters: {
        type: 'object',
        properties: {
          file_path: {
            type: 'string',
            description: 'Path to the uploaded file'
          },
          user_id: {
            type: 'string',
            description: 'User ID'
          }
        },
        required: ['file_path', 'user_id']
      },
      handler: async ({ file_path, user_id }) => {
        try {
          const doc = await storeDocument(file_path, user_id);
          return {
            success: true,
            message: `Stored document: ${doc.filename} with ${doc.numChunks} chunks`,
            filename: doc.filename,
            chunks: doc.numChunks
          };
        } catch (error) {
          return {
            success: false,
            error: error.message
          };
        }
      }
    },

    // Hybrid search
    search_documents: {
      description: 'Search stored documents using hybrid search. ONLY call when user explicitly requests document search with phrases like "search my documents", "what do my documents say", "based on my uploads", "use my files", "check documents for", or clearly references uploaded content. Do NOT use for general questions.',
      parameters: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'Search query or question'
          },
          user_id: {
            type: 'string',
            description: 'User ID'
          },
          top_k: {
            type: 'number',
            description: 'Number of results to return (default: 3)'
          }
        },
        required: ['query', 'user_id']
      },
      handler: async ({ query, user_id, top_k = 3 }) => {
        try {
          const results = await hybridSearch(query, user_id, top_k);

          if (results.length === 0) {
            return {
              found: false,
              message: 'No relevant information found in your documents.'
            };
          }

          return {
            found: true,
            count: results.length,
            results: results.map(r => ({
              filename: r.filename,
              excerpt: r.text,
              relevance: r.hybridScore.toFixed(3),
              methods: r.methods.join(' + ')
            }))
          };
        } catch (error) {
          return {
            found: false,
            error: error.message
          };
        }
      }
    },

    // List documents
    list_documents: {
      description: 'List all stored documents for a user',
      parameters: {
        type: 'object',
        properties: {
          user_id: {
            type: 'string',
            description: 'User ID'
          }
        },
        required: ['user_id']
      },
      handler: async ({ user_id }) => {
        const docs = getUserDocuments(user_id);
        return {
          count: docs.length,
          documents: docs
        };
      }
    },

    // Get full document
    get_document: {
      description: 'Get full content of a specific document',
      parameters: {
        type: 'object',
        properties: {
          user_id: {
            type: 'string',
            description: 'User ID'
          },
          filename: {
            type: 'string',
            description: 'Document filename'
          }
        },
        required: ['user_id', 'filename']
      },
      handler: async ({ user_id, filename }) => {
        const content = getDocumentContent(user_id, filename);

        if (!content) {
          return {
            found: false,
            message: 'Document not found'
          };
        }

        return {
          found: true,
          filename: filename,
          content: content
        };
      }
    }
  }
};