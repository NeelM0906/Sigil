import { exec } from 'node:child_process';
import { promisify } from 'node:util';
import path from 'node:path';
import fs from 'node:fs/promises';

const execAsync = promisify(exec);

/**
 * Extract the actual user message from commandBody
 * Removes system metadata, timestamps, and message IDs
 */
function extractUserMessage(commandBody: string): string {
  // Pattern 1: [WhatsApp ... EST] MESSAGE [message_id: ...]
  const whatsappPattern = /\[WhatsApp[^\]]+\]\s*(.+?)\s*\[message_id:/s;
  const match1 = commandBody.match(whatsappPattern);
  if (match1) {
    return match1[1].trim();
  }

  // Pattern 2: Look for content after last ] bracket
  const lines = commandBody.split('\n');
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i].trim();
    if (line && !line.startsWith('System:') && !line.startsWith('[') && !line.includes('message_id')) {
      return line;
    }
  }

  // Fallback: return as-is
  return commandBody.trim();
}

/**
 * Check if user is asking about Kumar/Genesis content
 */
function isKumarQuery(message: string): boolean {
  const kumarKeywords = [
    'kumar', 'genesis', 'patient intake', 'medical history',
    'clinical', 'allergy', 'asthma', 'medical', 'intake protocol'
  ];

  const lowerMsg = message.toLowerCase();
  return kumarKeywords.some(keyword => lowerMsg.includes(keyword));
}

/**
 * Get Kumar's answer if relevant
 */
export async function getKumarAnswer(userMessage: string): Promise<string | null> {
  console.log('[Kumar] ===== getKumarAnswer CALLED =====');
  console.log('[Kumar] Raw message length:', userMessage.length);

  // Extract clean message
  const cleanMessage = extractUserMessage(userMessage);
  console.log('[Kumar] Clean message:', cleanMessage);

  // Only check Kumar for relevant queries
  if (!isKumarQuery(cleanMessage)) {
    console.log('[Kumar] Not a Kumar query, skipping');
    return null;
  }

  console.log('[Kumar] IS a Kumar query, checking knowledge base...');

  try {
    const kumarPath = path.join(process.cwd(), 'skills', 'kumar');
    const knowledgeFile = path.join(kumarPath, 'kumar_clean.json');

    console.log('[Kumar] Knowledge file path:', knowledgeFile);

    // Check if knowledge base exists
    try {
      await fs.access(knowledgeFile);
      console.log('[Kumar] Knowledge file exists');
    } catch {
      console.log('[Kumar] Knowledge file NOT FOUND');
      return null;
    }

    console.log('[Kumar] Running Python script with clean message...');

    // Use clean message for Python script
    const escapedMessage = cleanMessage.replace(/"/g, '\\"').replace(/\n/g, ' ');

    const { stdout, stderr } = await execAsync(
      `python get_answer.py "${escapedMessage}"`,
      {
        cwd: kumarPath,
        timeout: 5000,
        env: { ...process.env, PYTHONIOENCODING: 'utf-8' }
      }
    );

    console.log('[Kumar] Python stdout:', stdout || '(empty)');
    console.log('[Kumar] Python stderr:', stderr || '(none)');

    console.log('[Kumar] Python stdout length:', stdout.length);
    if (stderr) console.log('[Kumar] Python stderr:', stderr);

    const answer = stdout.trim();

    if (answer && answer.length > 100) {
      console.log('[Kumar] ✓ GOT ANSWER FROM CSV');
      console.log('[Kumar] Answer preview:', answer.substring(0, 100));
      return answer;
    } else {
      console.log('[Kumar] Answer too short or empty');
      return null;
    }

  } catch (err: any) {
    console.error('[Kumar] ERROR:', err.message);
    return null;
  }
}

/**
 * Save conversation (simplified)
 */
export async function saveKumarConversation(
  userMessage: string,
  botResponse: string
) {
  try {
    const kumarPath = path.join(process.cwd(), 'skills', 'kumar');

    // Extract clean message
    const cleanMessage = extractUserMessage(userMessage);

    const escapedUser = cleanMessage.replace(/"/g, '\\"').replace(/\n/g, ' ');
    const escapedBot = botResponse.replace(/"/g, '\\"').replace(/\n/g, ' ');

    execAsync(
      `python learn_from_conversation.py "${escapedUser}" "${escapedBot}"`,
      { cwd: kumarPath, timeout: 5000 }
    ).catch(() => {});

  } catch {
    // Silent fail
  }
}