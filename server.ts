import 'dotenv/config';
import express, { Request, Response } from 'express';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = 3000;
const ALLOWED_IMAGE_MIME_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp']);
const MAX_IMAGE_BASE64_BYTES = 18 * 1024 * 1024; // ~18MB base64 string cap per page

function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') {
    throw new Error('MISSING_API_KEY');
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

function isTransientError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err);
  return (
    msg.includes('503') ||
    msg.includes('UNAVAILABLE') ||
    msg.includes('high demand') ||
    msg.includes('429') ||
    msg.includes('RESOURCE_EXHAUSTED') ||
    msg.includes('ECONNRESET') ||
    msg.includes('ETIMEDOUT')
  );
}

async function withRetry<T>(fn: () => Promise<T>, maxRetries = 3): Promise<T> {
  let lastErr: unknown;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (err: unknown) {
      lastErr = err;
      if (!isTransientError(err) || attempt === maxRetries) {
        throw err;
      }
      const jitter = Math.floor(Math.random() * 400);
      const delayMs = 1000 * Math.pow(2, attempt) + jitter;
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  }
  throw lastErr;
}

async function generateContentResilient(
  ai: GoogleGenAI,
  params: {
    contents: Parameters<GoogleGenAI['models']['generateContent']>[0]['contents'];
    primaryConfig: Parameters<GoogleGenAI['models']['generateContent']>[0]['config'];
    fallbackConfig?: Parameters<GoogleGenAI['models']['generateContent']>[0]['config'];
  }
) {
  try {
    return await withRetry(
      () =>
        ai.models.generateContent({
          model: 'gemini-4-hydrogen',
          contents: params.contents,
          config: params.primaryConfig,
        }),
      2
    );
  } catch (primaryErr) {
    if (!isTransientError(primaryErr)) {
      throw primaryErr;
    }
    console.warn('gemini-4-hydrogen tạm thời quá tải (503/429), chuyển sang gemini-3.1-flash-lite...');
    return await withRetry(
      () =>
        ai.models.generateContent({
          model: 'gemini-3.1-flash-lite',
          contents: params.contents,
          config: params.fallbackConfig ?? params.primaryConfig,
        }),
      2
    );
  }
}

function extractReadableErrorMessage(err: unknown): string {
  const raw = err instanceof Error ? err.message : String(err);
  try {
    const parsed = JSON.parse(raw);
    if (parsed?.error?.message) {
      return String(parsed.error.message);
    }
  } catch {
    // Raw message is not pure JSON
  }
  return raw;
}

function handleGeminiError(res: Response, err: unknown) {
  const raw = err instanceof Error ? err.message : 'Lỗi không xác định khi gọi AI.';
  const readable = extractReadableErrorMessage(err);

  if (
    raw === 'MISSING_API_KEY' ||
    raw.includes('API_KEY_INVALID') ||
    raw.includes('PERMISSION_DENIED') ||
    raw.includes('403')
  ) {
    return res.status(401).json({
      error: 'Khóa API Gemini chưa được cấu hình hoặc không hợp lệ. Vui lòng kiểm tra trong mục Settings > Secrets.',
    });
  }
  if (raw.includes('503') || raw.includes('UNAVAILABLE') || raw.includes('high demand')) {
    return res.status(503).json({
      error:
        'Máy chủ AI đang có lượng truy cập cao tạm thời (lỗi 503). Hệ thống đã tự động thử lại nhưng chưa kết nối được, vui lòng bấm "Thử lại" sau vài giây.',
    });
  }
  if (raw.includes('RESOURCE_EXHAUSTED') || raw.includes('429')) {
    return res.status(429).json({
      error: 'Hệ thống đang vượt hạn mức gọi API tạm thời. Vui lòng chờ 10–15 giây rồi bấm thử lại.',
    });
  }
  console.error('Gemini API error:', err);
  return res.status(500).json({
    error: `Không thể phân tích tài liệu: ${readable}`,
  });
}

function parseDataUrl(dataUrl: string): { mimeType: string; data: string } | null {
  if (typeof dataUrl !== 'string' || dataUrl.length > MAX_IMAGE_BASE64_BYTES) {
    return null;
  }
  const match = dataUrl.match(/^data:([a-zA-Z0-9/+.-]+);base64,([A-Za-z0-9+/=]+)$/);
  if (!match) {
    return null;
  }
  const mimeType = match[1].toLowerCase();
  if (!ALLOWED_IMAGE_MIME_TYPES.has(mimeType)) {
    return null;
  }
  return { mimeType, data: match[2] };
}

function clampBox(rawBox: unknown): [number, number, number, number] {
  if (!Array.isArray(rawBox) || rawBox.length !== 4) {
    return [50, 50, 950, 950];
  }
  const nums = rawBox.map((v) => {
    const n = Number(v);
    if (!Number.isFinite(n)) return 0;
    return Math.max(0, Math.min(1000, Math.round(n)));
  });
  const ymin = Math.min(nums[0], nums[2]);
  const xmin = Math.min(nums[1], nums[3]);
  const ymax = Math.max(nums[0], nums[2]);
  const xmax = Math.max(nums[1], nums[3]);
  if (ymax - ymin < 25 || xmax - xmin < 25) {
    return [
      Math.max(0, ymin - 40),
      Math.max(0, xmin - 40),
      Math.min(1000, ymax + 40),
      Math.min(1000, xmax + 40),
    ];
  }
  return [ymin, xmin, ymax, xmax];
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '45mb' }));

  // Serve pdfjs-dist CMaps and standard fonts locally for crisp Japanese CID font rendering
  const pdfjsDir = path.resolve(__dirname, 'node_modules/pdfjs-dist');
  if (fs.existsSync(path.join(pdfjsDir, 'cmaps'))) {
    app.use('/pdfjs/cmaps', express.static(path.join(pdfjsDir, 'cmaps')));
  }
  if (fs.existsSync(path.join(pdfjsDir, 'standard_fonts'))) {
    app.use('/pdfjs/standard_fonts', express.static(path.join(pdfjsDir, 'standard_fonts')));
  }

  app.get('/api/health', (_req: Request, res: Response) => {
    res.json({ ok: true });
  });

  app.post('/api/analyze-page', async (req: Request, res: Response) => {
    try {
      const { pageNumber, imageDataUrl, extractedText, fileName } = req.body ?? {};
      const pageNum = Number(pageNumber);
      if (!Number.isInteger(pageNum) || pageNum < 1 || pageNum > 200) {
        return res.status(400).json({ error: 'Số trang không hợp lệ.' });
      }
      const parsedImage = parseDataUrl(imageDataUrl);
      if (!parsedImage) {
        return res.status(400).json({ error: 'Dữ liệu hình ảnh trang PDF không hợp lệ hoặc vượt quá giới hạn dung lượng.' });
      }

      const safeTextContext = typeof extractedText === 'string' ? extractedText.slice(0, 12000) : '';
      const safeFileName = typeof fileName === 'string' ? fileName.slice(0, 180) : 'document.pdf';

      const ai = getGeminiClient();

      const systemInstruction = `Bạn là Kỹ sư Điều phối Thiết kế & Thi công Nội thất cao cấp (設計監理・内装ディレクター) thông thạo tiếng Nhật và tiếng Việt, chuyên xử lý hồ sơ bản vẽ, chỉ thị chỉnh sửa (赤チェック / redline / 修正指示), bảng vật liệu (仕上表 / FF&E) và chi tiết đồ gỗ (造作家具図) từ khách hàng Nhật Bản.

Nhiệm vụ của bạn khi phân tích hình ảnh 1 trang tài liệu PDF nội thất:
1. Đọc kỹ toàn bộ chữ in, chữ viết tay, mũi tên chỉ dẫn, đám mây sửa đổi (cloud mark), kích thước (mm), cao độ (FL, GL, CH, H), và mã vật liệu (例: メラミン化粧板, 突板, 人工大理石, クロス, 塩ビタイル, 巾木, 見切り材, 納まり, ダイノックシート, 塗装).
2. Dịch chuẩn xác toàn bộ các khối nội dung tiếng Nhật sang tiếng Việt chuyên ngành nội thất (giữ mã hiệu vật liệu, kích thước gốc).
3. Trích xuất ĐẦY ĐỦ tất cả các CHỈ THỊ / YÊU CẦU KHÁCH MUỐN LÀM HOẶC CHỈNH SỬA (directives):
   - Phân loại rõ:
     - "edit": Chỉnh sửa bản vẽ, thay đổi kích thước, dời vị trí, xóa bỏ, sửa chi tiết.
     - "create": Làm mới, vẽ bổ sung, thêm hạng mục.
     - "material": Thay đổi hoặc chỉ định vật liệu, mã màu, hoàn thiện bề mặt, phụ kiện (bản lề, ray, tay nắm, đèn LED...).
     - "dimension": Lưu ý kích thước, cao độ, khe hở kỹ thuật, tiêu chuẩn lắp đặt.
     - "verify": Điểm khách ghi chú chưa rõ, thiếu mã, hoặc mâu thuẫn cần hỏi xác nhận lại khách Nhật.
   - Đánh dấu mức độ ưu tiên: "urgent" (sửa gấp / redline nổi bật / từ ngữ nhấn mạnh như 至急, 要修正, 変更, NG), "important" (quan trọng), "normal" (thông thường).
   - SỐ THỨ TỰ CHỈ THỊ TRÊN PDF (pdfMark): Hãy đọc chính xác số thứ tự hoặc ký hiệu đánh số của chỉ thị trên bản vẽ (VD: "①", "②", "1", "2", "(3)", "A-1", "No.1"...). Liệt kê các chỉ thị theo đúng thứ tự số trên PDF (hoặc từ trên xuống dưới, trái sang phải nếu không đánh số). Nếu trên bản vẽ không có số hiệu riêng thì để chuỗi rỗng "".
   - QUAN TRỌNG NHẤT VỀ CHỤP VÙNG HÌNH ẢNH (box_2d): Với mỗi chỉ thị, hãy trả về tọa độ bounding box chuẩn xác [ymin, xmin, ymax, xmax] theo thang 0–1000 bao trọn VÙNG HÌNH VẼ ĐỒ NỘI THẤT / CHI TIẾT KỸ THUẬT CÙNG MŨI TÊN VÀ CHÚ THÍCH TIẾNG NHẬT của chỉ thị đó trên trang, để hệ thống tự động chụp cắt ra bức ảnh trực quan nhất cho anh em họa viên / thợ xem và làm theo mà không bị cắt mất ngữ cảnh.
4. Trích xuất danh sách thuật ngữ chuyên ngành nội thất Nhật–Việt xuất hiện trong trang.`;

      const promptText = `Tệp: "${safeFileName}" - Trang ${pageNum}.
${safeTextContext ? `\nVăn bản trích xuất thô từ lớp text PDF (tham khảo bổ sung, có thể thiếu chữ viết tay/ảnh):\n${safeTextContext}\n` : ''}
Hãy phân tích hình ảnh trang PDF đính kèm và trả về JSON đúng schema yêu cầu bằng tiếng Việt (giữ nguyên trích dẫn tiếng Nhật ở các trường tương ứng).`;

      const pageSchema = {
        type: Type.OBJECT,
        properties: {
          documentTypeVi: {
            type: Type.STRING,
            description: 'Loại tài liệu của trang này bằng tiếng Việt (VD: Mặt bằng bố trí, Chi tiết tủ bếp, Chỉ thị sửa đổi Redline, Bảng hoàn thiện vật liệu)',
          },
          pageTitleVi: {
            type: Type.STRING,
            description: 'Tiêu đề ngắn gọn của trang bằng tiếng Việt',
          },
          pageSummaryVi: {
            type: Type.STRING,
            description: 'Tóm tắt 2-4 câu về nội dung trang và những yêu cầu chính khách muốn làm trên trang này',
          },
          translationBlocks: {
            type: Type.ARRAY,
            description: 'Các khối văn bản/ghi chú quan trọng đã dịch từ tiếng Nhật sang tiếng Việt',
            items: {
              type: Type.OBJECT,
              properties: {
                sectionLabel: { type: Type.STRING, description: 'Tên khu vực hoặc tiêu đề mục' },
                japaneseOriginal: { type: Type.STRING, description: 'Nguyên văn tiếng Nhật' },
                vietnameseTranslation: { type: Type.STRING, description: 'Bản dịch tiếng Việt chuẩn chuyên ngành nội thất' },
                contextNote: { type: Type.STRING, description: 'Ghi chú ngữ cảnh thi công / bản vẽ nếu có' },
              },
              required: ['sectionLabel', 'japaneseOriginal', 'vietnameseTranslation', 'contextNote'],
            },
          },
          directives: {
            type: Type.ARRAY,
            description: 'Danh sách từng chỉ thị, hạng mục cần làm hoặc cần chỉnh sửa của khách hàng Nhật',
            items: {
              type: Type.OBJECT,
              properties: {
                pdfMark: {
                  type: Type.STRING,
                  description: 'Số thứ tự hoặc ký hiệu chỉ thị ghi trên bản vẽ PDF (VD: ①, ②, 1, 2, A1). Để "" nếu không ghi số.',
                },
                category: {
                  type: Type.STRING,
                  description: 'Chỉ chọn một trong: edit, create, material, dimension, verify',
                },
                priority: {
                  type: Type.STRING,
                  description: 'Chỉ chọn một trong: urgent, important, normal',
                },
                titleVi: {
                  type: Type.STRING,
                  description: 'Tiêu đề hành động ngắn gọn, rõ ràng (VD: Đổi độ sâu tủ giày thành D380mm)',
                },
                actionRequiredVi: {
                  type: Type.STRING,
                  description: 'Hướng dẫn chi tiết cần làm gì / chỉnh sửa gì cho đội ngũ thiết kế hoặc sản xuất',
                },
                japaneseQuote: {
                  type: Type.STRING,
                  description: 'Nguyên văn chỉ thị tiếng Nhật trên tài liệu',
                },
                locationLabel: {
                  type: Type.STRING,
                  description: 'Vị trí phòng hoặc món đồ nội thất (VD: Phòng khách - Vách TV)',
                },
                specs: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                  description: 'Các thông số kích thước, mã vật liệu, phụ kiện đi kèm',
                },
                box_2d: {
                  type: Type.ARRAY,
                  items: { type: Type.INTEGER },
                  description: 'Tọa độ vùng hình ảnh cần chụp cắt [ymin, xmin, ymax, xmax] từ 0 đến 1000, bao gồm cả hình vẽ chi tiết và ghi chú mũi tên liên quan',
                },
              },
              required: [
                'pdfMark',
                'category',
                'priority',
                'titleVi',
                'actionRequiredVi',
                'japaneseQuote',
                'locationLabel',
                'specs',
                'box_2d',
              ],
            },
          },
          glossaryTerms: {
            type: Type.ARRAY,
            description: 'Thuật ngữ nội thất tiếng Nhật xuất hiện trong trang',
            items: {
              type: Type.OBJECT,
              properties: {
                japanese: { type: Type.STRING },
                reading: { type: Type.STRING, description: 'Phiên âm Hiragana/Romaji' },
                vietnamese: { type: Type.STRING },
                note: { type: Type.STRING, description: 'Giải thích ngắn trong thực tế nội thất Nhật' },
              },
              required: ['japanese', 'reading', 'vietnamese', 'note'],
            },
          },
        },
        required: [
          'documentTypeVi',
          'pageTitleVi',
          'pageSummaryVi',
          'translationBlocks',
          'directives',
          'glossaryTerms',
        ],
      };

      const response = await generateContentResilient(ai, {
        contents: {
          parts: [
            {
              inlineData: {
                mimeType: parsedImage.mimeType,
                data: parsedImage.data,
              },
            },
            { text: promptText },
          ],
        },
        primaryConfig: {
          systemInstruction,
          thinkingConfig: { thinkingLevel: ThinkingLevel.MEDIUM },
          responseMimeType: 'application/json',
          responseSchema: pageSchema,
        },
        fallbackConfig: {
          systemInstruction,
          responseMimeType: 'application/json',
          responseSchema: pageSchema,
        },
      });

      const raw = (response.text ?? '{}').trim();
      const parsed = JSON.parse(raw);

      const validCategories = new Set(['edit', 'create', 'material', 'dimension', 'verify']);
      const validPriorities = new Set(['urgent', 'important', 'normal']);

      const directives = Array.isArray(parsed.directives)
        ? parsed.directives.map((d: Record<string, unknown>, idx: number) => ({
            id: `p${pageNum}-d${idx + 1}-${Date.now().toString(36)}`,
            pageNumber: pageNum,
            pdfSequence: idx + 1,
            pdfMark: typeof d.pdfMark === 'string' ? d.pdfMark.trim() : '',
            category: validCategories.has(String(d.category)) ? String(d.category) : 'edit',
            priority: validPriorities.has(String(d.priority)) ? String(d.priority) : 'important',
            status: 'todo',
            titleVi: String(d.titleVi ?? 'Chỉ thị chưa đặt tên'),
            actionRequiredVi: String(d.actionRequiredVi ?? ''),
            japaneseQuote: String(d.japaneseQuote ?? ''),
            locationLabel: String(d.locationLabel ?? `Trang ${pageNum}`),
            specs: Array.isArray(d.specs) ? d.specs.map((s) => String(s)) : [],
            box_2d: clampBox(d.box_2d),
          }))
        : [];

      const translationBlocks = Array.isArray(parsed.translationBlocks)
        ? parsed.translationBlocks.map((b: Record<string, unknown>, idx: number) => ({
            id: `p${pageNum}-t${idx + 1}`,
            sectionLabel: String(b.sectionLabel ?? `Mục ${idx + 1}`),
            japaneseOriginal: String(b.japaneseOriginal ?? ''),
            vietnameseTranslation: String(b.vietnameseTranslation ?? ''),
            contextNote: String(b.contextNote ?? ''),
          }))
        : [];

      const glossaryTerms = Array.isArray(parsed.glossaryTerms)
        ? parsed.glossaryTerms.map((g: Record<string, unknown>) => ({
            japanese: String(g.japanese ?? ''),
            reading: String(g.reading ?? ''),
            vietnamese: String(g.vietnamese ?? ''),
            note: String(g.note ?? ''),
          }))
        : [];

      return res.json({
        pageNumber: pageNum,
        documentTypeVi: String(parsed.documentTypeVi ?? 'Tài liệu nội thất'),
        pageTitleVi: String(parsed.pageTitleVi ?? `Trang ${pageNum}`),
        pageSummaryVi: String(parsed.pageSummaryVi ?? ''),
        translationBlocks,
        directives,
        glossaryTerms,
      });
    } catch (err) {
      return handleGeminiError(res, err);
    }
  });

  app.post('/api/synthesize-summary', async (req: Request, res: Response) => {
    try {
      const { fileName, pages } = req.body ?? {};
      if (!Array.isArray(pages) || pages.length === 0) {
        return res.status(400).json({ error: 'Chưa có dữ liệu trang để tổng kết.' });
      }

      const compactInput = pages.slice(0, 40).map((p: Record<string, unknown>) => ({
        pageNumber: p.pageNumber,
        pageTitleVi: p.pageTitleVi,
        pageSummaryVi: p.pageSummaryVi,
        directives: Array.isArray(p.directives)
          ? p.directives.map((d: Record<string, unknown>) => ({
              category: d.category,
              priority: d.priority,
              titleVi: d.titleVi,
              actionRequiredVi: d.actionRequiredVi,
              locationLabel: d.locationLabel,
            }))
          : [],
      }));

      const ai = getGeminiClient();

      const summarySchema = {
        type: Type.OBJECT,
        properties: {
          projectTitleVi: { type: Type.STRING },
          clientContextVi: { type: Type.STRING },
          topPrioritiesVi: { type: Type.ARRAY, items: { type: Type.STRING } },
          revisionsSummaryVi: { type: Type.ARRAY, items: { type: Type.STRING } },
          newWorkSummaryVi: { type: Type.ARRAY, items: { type: Type.STRING } },
          clarificationsNeededVi: { type: Type.ARRAY, items: { type: Type.STRING } },
        },
        required: [
          'projectTitleVi',
          'clientContextVi',
          'topPrioritiesVi',
          'revisionsSummaryVi',
          'newWorkSummaryVi',
          'clarificationsNeededVi',
        ],
      };

      const summaryInstruction =
        'Bạn là Quản lý dự án nội thất Nhật–Việt. Hãy tổng kết toàn bộ tài liệu PDF thành các danh mục rõ ràng: tên dự án, bối cảnh khách Nhật, việc ưu tiên hàng đầu, những điểm cần chỉnh sửa, những hạng mục cần làm mới/bổ sung, và những điểm cần hỏi xác nhận lại với khách hàng Nhật.';

      const response = await generateContentResilient(ai, {
        contents: `Tài liệu: ${String(fileName ?? 'Hồ sơ nội thất')}\n\nDữ liệu phân tích các trang:\n${JSON.stringify(compactInput, null, 2)}\n\nHãy tổng hợp thành bản báo cáo tổng kết ngắn gọn, súc tích, dễ hành động cho nhóm thiết kế & thi công nội thất Việt Nam.`,
        primaryConfig: {
          systemInstruction: summaryInstruction,
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          responseMimeType: 'application/json',
          responseSchema: summarySchema,
        },
        fallbackConfig: {
          systemInstruction: summaryInstruction,
          responseMimeType: 'application/json',
          responseSchema: summarySchema,
        },
      });

      const parsed = JSON.parse((response.text ?? '{}').trim());
      return res.json(parsed);
    } catch (err) {
      return handleGeminiError(res, err);
    }
  });

  app.post('/api/ask-document', async (req: Request, res: Response) => {
    try {
      const { question, pageContext, imageDataUrl } = req.body ?? {};
      if (typeof question !== 'string' || !question.trim()) {
        return res.status(400).json({ error: 'Vui lòng nhập câu hỏi.' });
      }

      const ai = getGeminiClient();
      const parts: Array<{ inlineData: { mimeType: string; data: string } } | { text: string }> = [];

      if (imageDataUrl) {
        const parsedImage = parseDataUrl(imageDataUrl);
        if (parsedImage) {
          parts.push({ inlineData: { mimeType: parsedImage.mimeType, data: parsedImage.data } });
        }
      }

      const contextSnippet = typeof pageContext === 'string' ? pageContext.slice(0, 8000) : '';
      parts.push({
        text: `Ngữ cảnh tài liệu:\n${contextSnippet}\n\nCâu hỏi của người dùng:\n${question.trim().slice(0, 2000)}`,
      });

      const askInstruction =
        'Bạn là chuyên gia kỹ thuật & điều phối dự án nội thất Nhật–Việt. Hãy trả lời chính xác, thực tế, dễ áp dụng cho họa viên CAD và thợ thi công. Nếu người dùng nhờ viết email/tin nhắn hỏi lại khách Nhật, hãy cung cấp cả bản tiếng Nhật lịch sự (Keigo/Business Japanese) kèm bản dịch tiếng Việt.';

      const response = await generateContentResilient(ai, {
        contents: { parts },
        primaryConfig: {
          systemInstruction: askInstruction,
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
        },
        fallbackConfig: {
          systemInstruction: askInstruction,
        },
      });

      return res.json({ answer: response.text ?? 'Không có phản hồi.' });
    } catch (err) {
      return handleGeminiError(res, err);
    }
  });

  // Vite middleware in development, static dist in production
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req: Request, res: Response) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Sekkei Brief server running at http://localhost:${PORT}`);
  });
}

startServer();
