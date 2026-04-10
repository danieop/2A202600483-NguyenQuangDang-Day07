# Báo cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Quang Đăng  
**Nhóm:** X100
**Ngày:** 2026-04-10

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**  
High cosine similarity nghĩa là hai vector embedding có hướng rất giống nhau, tức là hai đoạn văn bản có ý nghĩa gần nhau về ngữ nghĩa. Điểm càng gần 1.0 thì mức độ tương đồng ngữ nghĩa càng cao.

**Ví dụ HIGH similarity:**
- Sentence A: VF3 warranty covers repair for manufacturing defects.
- Sentence B: Vehicle warranty includes repairs for manufacturer defects.
- Tại sao tương đồng: Cả hai đều nói về cùng một nội dung bảo hành do lỗi sản xuất.

**Ví dụ LOW similarity:**
- Sentence A: Audit committee oversees compliance and governance.
- Sentence B: How to charge the VF3 battery at home safely.
- Tại sao khác: Một câu về governance/compliance, một câu về hướng dẫn sử dụng xe.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**  
Cosine similarity tập trung vào hướng vector (ngữ nghĩa) thay vì độ lớn vector. Với text embedding, hướng ngữ nghĩa quan trọng hơn magnitude nên cosine ổn định và hợp lý hơn.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**  
num_chunks = ceil((10000 - 50) / (500 - 50))  
= ceil(9950 / 450)  
= ceil(22.11) = 23  
**Đáp án:** 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**  
num_chunks = ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = 25.  
Chunk count tăng vì step nhỏ hơn. Overlap lớn hơn giúp giữ ngữ cảnh qua ranh giới chunk.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** VinFast vehicle docs (spec + warranty + first responder)

**Tại sao nhóm chọn domain này?**  
Bộ tài liệu gồm nhiều loại nội dung (spec sheet, warranty policy, first responder) và song ngữ Việt/Anh, phù hợp để test chunking và metadata filter trong bài toán RAG.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | VF3_spec.md | data/vinfast_markdown_clean/VF3_spec.md | 4867 | source, extension=.md, model=VF3, doc_type=spec, language=en, region=global |
| 2 | vf3_vn_warranty.md | data/vinfast_markdown_clean/vf3_vn_warranty.md | 17814 | source, extension=.md, model=VF3, doc_type=warranty, language=vi, region=VN |
| 3 | 20230927_VF6_VN_VN_1_1706781000_Warranty.md | data/vinfast_markdown_clean/20230927_VF6_VN_VN_1_1706781000_Warranty.md | 28491 | source, extension=.md, model=VF6, doc_type=warranty, language=vi, region=VN |
| 4 | VF9 US Vehicle Warranty Booklet.md | data/vinfast_markdown_clean/VF9 US Vehicle Warranty Booklet.md | 27490 | source, extension=.md, model=VF9, doc_type=warranty, language=en, region=US |
| 5 | VINFAST_VF8_First_Responder_Guide.md | data/vinfast_markdown_clean/VINFAST_VF8_First_Responder_Guide.md | 17402 | source, extension=.md, model=VF8, doc_type=first_responder, language=en, region=global |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| source | str | data\vinfast_markdown_clean\vf3_vn_warranty.md | Filter theo tài liệu gốc |
| extension | str | .md | Kiểm soát ingestion theo định dạng |
| model | str | VF3 / VF6 / VF8 / VF9 | Lọc đúng mẫu xe trong câu hỏi |
| doc_type | str | spec / warranty / first_responder | Lọc đúng loại tài liệu |
| language | str | vi / en | Lọc theo ngôn ngữ query |
| region | str | VN / US / global | Tách quy định theo thị trường |
| doc_id | str | vf3_vn_warranty | Gom nhóm các chunk cùng document |
| chunk_index | int | 24 | Truy vết vị trí chunk |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Kết quả `ChunkingStrategyComparator().compare(..., chunk_size=500)` trên 3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| VF3_spec | FixedSizeChunker (`fixed_size`) | 11 | 487.91 | Medium |
| VF3_spec | SentenceChunker (`by_sentences`) | 3 | 1620.67 | Low |
| VF3_spec | RecursiveChunker (`recursive`) | 14 | 345.86 | High |
| vf3_vn_warranty | FixedSizeChunker (`fixed_size`) | 40 | 494.10 | Medium |
| vf3_vn_warranty | SentenceChunker (`by_sentences`) | 27 | 658.48 | High |
| vf3_vn_warranty | RecursiveChunker (`recursive`) | 49 | 361.59 | High |
| 20230927_VF6_VN_VN_1_1706781000_Warranty | FixedSizeChunker (`fixed_size`) | 64 | 494.39 | Medium |
| 20230927_VF6_VN_VN_1_1706781000_Warranty | SentenceChunker (`by_sentences`) | 81 | 350.46 | Medium |
| 20230927_VF6_VN_VN_1_1706781000_Warranty | RecursiveChunker (`recursive`) | 83 | 341.29 | High |

### Strategy Của Tôi

**Loại:** RecursiveChunker + metadata-aware retrieval

**Mô tả cách hoạt động:**  
Tôi dùng recursive chunking để cắt theo separator priority (`\n\n`, `\n`, `. `, ` `, fallback) và tiến hành tách đến khi đạt ngưỡng kích thước. Khi ingest, mỗi chunk được bổ sung metadata `doc_id`, `chunk_index`, `source`. Tôi ưu tiên bỏ `vinfast_markdown_clean` để giảm noise.

**Tại sao tôi chọn strategy này cho domain nhóm?**  
Tài liệu spec + warranty có cấu trúc khác nhau, recursive giúp giữ context ở bảng thông số và bullet policy tốt hơn fixed-size. Metadata filter hỗ trợ query cross-document (nhất là query về pin theo model).

**Code snippet (nếu custom):**
```python
# Không viết custom class mới.
# Sử dụng RecursiveChunker + metadata doc_id/chunk_index/source.
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| Golden benchmark 5 queries | best baseline (by_sentences) | N/A (không benchmark lại trong lần chạy này) | N/A | N/A |
| Golden benchmark 5 queries | **của tôi (recursive + metadata)** | 114 chunks (store 5 docs) | N/A | 6.0/10 (Top-3 relevant: 3/5) |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/5) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Nguyễn Minh Hiếu | FixedSizeChunker (500/50) | 4/5 | Top-1 đúng ở 4/5 query, ổn định trên cả bảng spec lẫn warranty VI | Query 5 (cross-doc, cần filter) fail — mọi strategy đều fail vì top-3 bị VF9 US warranty chiếm |
| Nguyễn Quang Đăng | Recursive + metadata filter | 3/5 | Tot cho query warranty/filter theo model | Chua on dinh voi query spec va battery-warranty cross-doc |
| Nguyễn Việt Long | Fixed-size + metadata filter | 4/5 | Tot hon o query spec va cac query warranty ro keyword | Query battery warranty cross-doc van kho |
| Hà Huy Hoàng | Semantic Chunker + Hybrid Search (Vector + BM25) | 4/5 | Khắc phục được phần lớn lỗi ở Query 5 (cross-doc) nhờ cụm từ khóa (BM25) và ngữ nghĩa (Vector) bổ trợ nhau. Tránh được nhiễu từ tài liệu VF9 US. | Thời gian indexing chậm và tốn tài nguyên tính toán hơn. Chunk size động đôi khi làm trượt Top-1 ở các query hỏi về thông số spec quá ngắn gọn |
| Tống Tiến Mạnh| RecursiveChunker tùy chỉnh (chunk_size=500, overlap=100) | 4/5 | Giữ cấu trúc section markdown, chunk bao trọn điều khoản | Chunk đôi khi vẫn dài nếu section liên tục >500 ký tự |

**Strategy nào tốt nhất cho domain này? Tại sao?**  
Recursive + metadata filter vẫn là hướng đúng, nhưng cần bổ sung metadata chi tiết hơn và tuning chunk-size cho query spec để cải thiện độ phủ top-3.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk`** — approach:  
Split theo regex ranh giới câu `(?<=[.!?])\s+`, giữ dấu câu, gom theo `max_sentences_per_chunk`, bỏ whitespace dư.

**`RecursiveChunker.chunk` / `_split`** — approach:  
Đệ quy theo separator priority. Base case là đoạn <= `chunk_size`. Nếu separator hiện tại không hiệu quả thì fallback separator tiếp theo; cuối cùng cắt theo ký tự.

### EmbeddingStore

**`add_documents` + `search`** — approach:  
Store hỗ trợ in-memory và fallback chroma. Similarity ranking dùng dot product. Bổ sung `embed_many` để batch embedding, giảm request API và thời gian runtime.

**`search_with_filter` + `delete_document`** — approach:  
Filter metadata trước, ranking sau. Delete xóa tất cả chunk có `metadata['doc_id'] == doc_id`.

### KnowledgeBaseAgent

**`answer`** — approach:  
Retrieve top-k context, tạo prompt RAG, gọi LLM. Bổ sung mode summary query để đa dạng hóa nguồn context (dedupe theo source), giảm trường hợp trả lời "insufficient context".

### Test Results

```
======================================== 42 passed in 0.06s ========================================
```

**Số tests pass:** 42 / 42

**OpenAI smoke test:** Đã chạy thành công trên 2 file markdown clean (`vf3_vn_warranty.md`, `VFSC_Code_of_Conduct.md`), lưu store, search top-3 và sinh answer bằng `gpt-4o-mini`.

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | VF3 warranty covers repair for manufacturing defects. | Vehicle warranty includes repairs for manufacturer defects. | high | 0.6595 | Yes |
| 2 | Code of conduct requires ethical behavior. | Employees must follow integrity and ethics guidelines. | high | 0.6544 | Yes |
| 3 | The first responder guide explains emergency handling. | Sustainability report discusses carbon emissions. | low | 0.1696 | Yes |
| 4 | Bảo hành pin có điều kiện áp dụng riêng. | Chính sách bảo hành xe điện có phạm vi cụ thể. | high | 0.5451 (medium) | Partly |
| 5 | Audit committee oversees compliance and governance. | How to charge the VF3 battery at home safely. | low | 0.0266 | Yes |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**  
Pair 4 bất ngờ nhất: chủ đề giống nhau nhưng score ở mức medium. Điều này cho thấy embedding vẫn phân biệt mạnh theo cách diễn đạt chi tiết và context.

---

## 6. Results — Cá nhân (10 điểm)

Chạy lại benchmark theo bộ golden queries nhóm thống nhất trên 5 tài liệu markdown clean tương ứng.

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | What is the battery capacity and range of the VinFast VF3? | Battery type LFP, capacity 18.64 kWh, range ~210 km (NEDC). |
| 2 | Thời hạn bảo hành chung của xe VinFast VF3 là bao lâu? | 7 năm/160.000 km (thường), 3 năm/100.000 km (thương mại), tính từ ngày kích hoạt bảo hành. |
| 3 | Những hư hỏng nào không được VinFast bảo hành? | Hư hỏng do sửa chữa trái phép, lạm dụng, thiên tai/tai nạn, hao mòn tự nhiên, phụ tùng không chính hãng. |
| 4 | How should first responders handle a VinFast VF8 high-voltage battery fire? | Assume HV live, wear PPE, use large water volume, monitor re-ignition risk. |
| 5 | What is the battery warranty period for VinFast vehicles? | VF3 (VN) battery warranty key point: 8 years or 160,000 km (non-commercial); cần filter theo model/doc_type. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | VF3 battery capacity/range | Top-1 lại rơi vào VF9 US warranty period, không chứa thông số 18.64 kWh và 210 km. | 0.6261 | No | Agent báo context chưa đủ để trả lời thông số VF3. |
| 2 | Thời hạn bảo hành chung VF3 | Chunk top-1 từ vf3_vn_warranty nói về thời hạn bảo hành và điều kiện. | 0.7082 | Yes | Agent trả lời đúng 7 năm/160.000 km và mối liên hệ với ngày kích hoạt bảo hành. |
| 3 | Hư hỏng không được bảo hành | Chunk top-1 từ vf3_vn_warranty có danh sách loại trừ (thiên tai, tai nạn, hao mòn...). | 0.7111 | Yes | Agent liệt kê đúng các nhóm hư hỏng không thuộc phạm vi bảo hành. |
| 4 | VF8 HV battery fire handling | Chunk top-1 từ VF8 first responder guide nói về danger, thermal runaway, xử lý cháy. | 0.7197 | Yes | Agent nêu đúng PPE, dùng nhiều nước, và theo dõi nguy cơ tái bốc cháy. |
| 5 | Battery warranty period (cross-doc) | Top-1 sau filter (model=VF3, doc_type=warranty) vẫn chưa đủ thông tin "8 năm" rõ ràng trong top-3. | 0.6318 | No | Agent trả lời có cảnh báo thiếu thông tin pin bảo hành cụ thể trong context đã retrieve. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 3 / 5

**Ghi chú Query 5 (filter test):** Không filter thì top-1 đến từ `VF9 US Vehicle Warranty Booklet` (score 0.7773); có filter `model=VF3, doc_type=warranty` thì top-1 chuyển về `vf3_vn_warranty` (score 0.6318), nhưng vẫn chưa đạt tiêu chí keyword trong top-3.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**  
Tăng overlap cho fixed-size có thể giúp giữ context ở query cần thông tin sát biên chunk.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**  
Thêm metadata business-oriented (document_type, effective_date, department) tăng hiệu quả filter rõ rệt.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**  
Tôi sẽ bổ sung clean pipeline bắt buộc cho OCR markdown ngay từ đầu, bắt buộc pre-chunk trước embedding, và thêm cache/re-ranker để tối ưu top-1 precision.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 14 / 15 |
| My approach | Cá nhân | 9 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **86 / 100** |
