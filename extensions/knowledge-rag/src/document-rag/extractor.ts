/**
 * Text extraction from various document formats.
 * Supports: PDF, DOCX, TXT, MD
 */

import fs from "node:fs";
import path from "node:path";

export type SupportedFormat = "pdf" | "docx" | "txt" | "md";

const SUPPORTED_EXTENSIONS: Record<string, SupportedFormat> = {
  ".pdf": "pdf",
  ".docx": "docx",
  ".txt": "txt",
  ".md": "md",
};

/**
 * Check if a file format is supported
 */
export function isSupportedFormat(filePath: string): boolean {
  const ext = path.extname(filePath).toLowerCase();
  return ext in SUPPORTED_EXTENSIONS;
}

/**
 * Get the format of a file
 */
export function getFormat(filePath: string): SupportedFormat | null {
  const ext = path.extname(filePath).toLowerCase();
  return SUPPORTED_EXTENSIONS[ext] ?? null;
}

/**
 * Extract text from a PDF file
 */
async function extractFromPdf(filePath: string): Promise<string> {
  const pdfParse = (await import("pdf-parse")).default;
  const dataBuffer = fs.readFileSync(filePath);
  const data = await pdfParse(dataBuffer);
  return data.text;
}

/**
 * Extract text from a DOCX file
 */
async function extractFromDocx(filePath: string): Promise<string> {
  const mammoth = await import("mammoth");
  const result = await mammoth.extractRawText({ path: filePath });
  return result.value;
}

/**
 * Extract text from a plain text file
 */
async function extractFromText(filePath: string): Promise<string> {
  return fs.readFileSync(filePath, "utf-8");
}

/**
 * Extract text from any supported document format
 */
export async function extractText(filePath: string): Promise<string> {
  const format = getFormat(filePath);

  if (!format) {
    const ext = path.extname(filePath);
    throw new Error(
      `Unsupported file format: ${ext}. Supported formats: ${Object.keys(SUPPORTED_EXTENSIONS).join(", ")}`
    );
  }

  switch (format) {
    case "pdf":
      return extractFromPdf(filePath);
    case "docx":
      return extractFromDocx(filePath);
    case "txt":
    case "md":
      return extractFromText(filePath);
    default:
      throw new Error(`Unhandled format: ${format}`);
  }
}

/**
 * Get file metadata
 */
export function getFileMetadata(filePath: string): {
  filename: string;
  format: SupportedFormat | null;
  sizeBytes: number;
} {
  const stats = fs.statSync(filePath);
  return {
    filename: path.basename(filePath),
    format: getFormat(filePath),
    sizeBytes: stats.size,
  };
}
