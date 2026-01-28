/**
 * Text chunking for document processing.
 * Uses fixed-window chunking with overlap for context preservation.
 */

export interface Chunk {
  text: string;
  index: number;
  startWord: number;
  endWord: number;
}

export interface ChunkOptions {
  /** Number of words per chunk (default: 500) */
  chunkSize?: number;
  /** Number of overlapping words between chunks (default: 100) */
  overlap?: number;
  /** Minimum chunk size to keep (default: 50) */
  minChunkSize?: number;
}

const DEFAULT_CHUNK_SIZE = 500;
const DEFAULT_OVERLAP = 100;
const DEFAULT_MIN_CHUNK_SIZE = 50;

/**
 * Split text into chunks with overlap for context preservation.
 * Uses word-based splitting to avoid cutting words.
 */
export function chunkText(text: string, options: ChunkOptions = {}): Chunk[] {
  const chunkSize = options.chunkSize ?? DEFAULT_CHUNK_SIZE;
  const overlap = options.overlap ?? DEFAULT_OVERLAP;
  const minChunkSize = options.minChunkSize ?? DEFAULT_MIN_CHUNK_SIZE;

  // Normalize whitespace and split into words
  const words = text.split(/\s+/).filter((w) => w.length > 0);

  if (words.length === 0) {
    return [];
  }

  // If text is smaller than chunk size, return as single chunk
  if (words.length <= chunkSize) {
    return [
      {
        text: words.join(" "),
        index: 0,
        startWord: 0,
        endWord: words.length - 1,
      },
    ];
  }

  const chunks: Chunk[] = [];
  const step = chunkSize - overlap;
  let chunkIndex = 0;

  for (let i = 0; i < words.length; i += step) {
    const chunkWords = words.slice(i, i + chunkSize);

    // Skip chunks that are too small (except the last one)
    if (chunkWords.length < minChunkSize && i + step < words.length) {
      continue;
    }

    const chunkText = chunkWords.join(" ");

    if (chunkText.trim().length > 0) {
      chunks.push({
        text: chunkText,
        index: chunkIndex,
        startWord: i,
        endWord: Math.min(i + chunkSize - 1, words.length - 1),
      });
      chunkIndex++;
    }

    // Stop if we've processed all words
    if (i + chunkSize >= words.length) {
      break;
    }
  }

  return chunks;
}

/**
 * Estimate the number of chunks for a given text length
 */
export function estimateChunkCount(
  wordCount: number,
  options: ChunkOptions = {}
): number {
  const chunkSize = options.chunkSize ?? DEFAULT_CHUNK_SIZE;
  const overlap = options.overlap ?? DEFAULT_OVERLAP;
  const step = chunkSize - overlap;

  if (wordCount <= chunkSize) {
    return 1;
  }

  return Math.ceil((wordCount - overlap) / step);
}

/**
 * Count words in text
 */
export function countWords(text: string): number {
  return text.split(/\s+/).filter((w) => w.length > 0).length;
}
