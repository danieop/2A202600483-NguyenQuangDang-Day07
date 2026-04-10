# Bao Cao Lab 7: Embedding & Vector Store

**Ho ten:** Nguyen Quang Dang  
**Nhom:** VinFast-RAG  
**Ngay:** 2026-04-10

---

## 1. Warm-up (5 diem)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghia la gi?**  
High cosine similarity nghia la hai vector embedding co huong rat giong nhau, tuc la hai doan van ban co y nghia gan nhau ve ngu nghia. Diem cang gan 1.0 thi muc do tuong dong ngu nghia cang cao.

**Vi du HIGH similarity:**
- Sentence A: VF3 warranty covers repair for manufacturing defects.
- Sentence B: Vehicle warranty includes repairs for manufacturer defects.
- Tai sao tuong dong: Ca hai deu noi ve cung mot noi dung bao hanh do loi san xuat.

**Vi du LOW similarity:**
- Sentence A: Audit committee oversees compliance and governance.
- Sentence B: How to charge the VF3 battery at home safely.
- Tai sao khac: Mot cau ve governance/compliance, mot cau ve huong dan su dung xe.

**Tai sao cosine similarity duoc uu tien hon Euclidean distance cho text embeddings?**  
Cosine similarity tap trung vao huong vector (ngu nghia) thay vi do lon vector. Voi text embedding, huong ngu nghia quan trong hon magnitude nen cosine on dinh va hop ly hon.

### Chunking Math (Ex 1.2)

**Document 10,000 ky tu, chunk_size=500, overlap=50. Bao nhieu chunks?**  
num_chunks = ceil((10000 - 50) / (500 - 50))  
= ceil(9950 / 450)  
= ceil(22.11) = 23  
**Dap an:** 23 chunks.

**Neu overlap tang len 100, chunk count thay doi the nao? Tai sao muon overlap nhieu hon?**  
num_chunks = ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = 25.  
Chunk count tang vi step nho hon. Overlap lon hon giup giu ngu canh qua ranh gioi chunk.

---

## 2. Document Selection — Nhom (10 diem)

### Domain & Ly Do Chon

**Domain:** VinFast vehicle docs (spec + warranty + first responder)

**Tai sao nhom chon domain nay?**  
Bo tai lieu gom nhieu loai noi dung (spec sheet, warranty policy, first responder) va song ngu Viet/Anh, phu hop de test chunking va metadata filter trong bai toan RAG.

### Data Inventory

| # | Ten tai lieu | Nguon | So ky tu | Metadata da gan |
|---|--------------|-------|----------|-----------------|
| 1 | VF3_spec.md | data/vinfast_markdown_clean/VF3_spec.md | 4867 | source, extension=.md, model=VF3, doc_type=spec, language=en, region=global |
| 2 | vf3_vn_warranty.md | data/vinfast_markdown_clean/vf3_vn_warranty.md | 17814 | source, extension=.md, model=VF3, doc_type=warranty, language=vi, region=VN |
| 3 | 20230927_VF6_VN_VN_1_1706781000_Warranty.md | data/vinfast_markdown_clean/20230927_VF6_VN_VN_1_1706781000_Warranty.md | 28491 | source, extension=.md, model=VF6, doc_type=warranty, language=vi, region=VN |
| 4 | VF9 US Vehicle Warranty Booklet.md | data/vinfast_markdown_clean/VF9 US Vehicle Warranty Booklet.md | 27490 | source, extension=.md, model=VF9, doc_type=warranty, language=en, region=US |
| 5 | VINFAST_VF8_First_Responder_Guide.md | data/vinfast_markdown_clean/VINFAST_VF8_First_Responder_Guide.md | 17402 | source, extension=.md, model=VF8, doc_type=first_responder, language=en, region=global |

### Metadata Schema

| Truong metadata | Kieu | Vi du gia tri | Tai sao huu ich cho retrieval? |
|----------------|------|---------------|-------------------------------|
| source | str | data\\vinfast_markdown_clean\\vf3_vn_warranty.md | Filter theo tai lieu goc |
| extension | str | .md | Kiem soat ingestion theo dinh dang |
| model | str | VF3 / VF6 / VF8 / VF9 | Loc dung mau xe trong cau hoi |
| doc_type | str | spec / warranty / first_responder | Loc dung loai tai lieu |
| language | str | vi / en | Loc theo ngon ngu query |
| region | str | VN / US / global | Tach quy dinh theo thi truong |
| doc_id | str | vf3_vn_warranty | Gom nhom cac chunk cung document |
| chunk_index | int | 24 | Truy vet vi tri chunk |

---

## 3. Chunking Strategy — Ca nhan chon, nhom so sanh (15 diem)

### Baseline Analysis

Ket qua `ChunkingStrategyComparator().compare(..., chunk_size=500)` tren 3 tai lieu:

| Tai lieu | Strategy | Chunk Count | Avg Length | Preserves Context? |
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

### Strategy Cua Toi

**Loai:** RecursiveChunker + metadata-aware retrieval

**Mo ta cach hoat dong:**  
Toi dung recursive chunking de cat theo separator priority (`\n\n`, `\n`, `. `, ` `, fallback) va tien hanh tach den khi dat nguong kich thuoc. Khi ingest, moi chunk duoc bo sung metadata `doc_id`, `chunk_index`, `source`. Toi uu tien bo `vinfast_markdown_clean` de giam noise.

**Tai sao toi chon strategy nay cho domain nhom?**  
Tai lieu spec + warranty co cau truc khac nhau, recursive giup giu context o bang thong so va bullet policy tot hon fixed-size. Metadata filter ho tro query cross-document (nhat la query ve pin theo model).

**Code snippet (neu custom):**
```python
# Khong viet custom class moi.
# Su dung RecursiveChunker + metadata doc_id/chunk_index/source.
```

### So Sanh: Strategy cua toi vs Baseline

| Tai lieu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| Golden benchmark 5 queries | best baseline (by_sentences) | N/A (khong benchmark lai trong lan chay nay) | N/A | N/A |
| Golden benchmark 5 queries | **cua toi (recursive + metadata)** | 114 chunks (store 5 docs) | N/A | 6.0/10 (Top-3 relevant: 3/5) |

### So Sanh Voi Thanh Vien Khac

| Thanh vien | Strategy | Retrieval Score (/10) | Diem manh | Diem yeu |
|-----------|----------|----------------------|-----------|----------|
| Toi | Recursive + metadata filter | 6.0 | Tot cho query warranty/filter theo model | Chua on dinh voi query spec va battery-warranty cross-doc |
| Thanh vien A | Fixed size 500 + overlap 50 | 7.2 | Don gian, de implement | Mat context o ranh gioi chunk |
| Thanh vien B | Sentence chunking | 8.1 | Readability tot | Kem on dinh voi list/table dai |

**Strategy nao tot nhat cho domain nay? Tai sao?**  
Recursive + metadata filter van la huong dung, nhung can bo sung metadata chi tiet hon va tuning chunk-size cho query spec de cai thien do phu top-3.

---

## 4. My Approach — Ca nhan (10 diem)

### Chunking Functions

**`SentenceChunker.chunk`** — approach:  
Split theo regex ranh gioi cau `(?<=[.!?])\s+`, giu dau cau, gom theo `max_sentences_per_chunk`, bo whitespace du.

**`RecursiveChunker.chunk` / `_split`** — approach:  
De quy theo separator priority. Base case la doan <= `chunk_size`. Neu separator hien tai khong hieu qua thi fallback separator tiep theo; cuoi cung cat theo ky tu.

### EmbeddingStore

**`add_documents` + `search`** — approach:  
Store ho tro in-memory va fallback chroma. Similarity ranking dung dot product. Bo sung `embed_many` de batch embedding, giam request API va thoi gian runtime.

**`search_with_filter` + `delete_document`** — approach:  
Filter metadata truoc, ranking sau. Delete xoa tat ca chunk co `metadata['doc_id'] == doc_id`.

### KnowledgeBaseAgent

**`answer`** — approach:  
Retrieve top-k context, tao prompt RAG, goi LLM. Bo sung mode summary query de da dang hoa nguon context (dedupe theo source), giam truong hop tra loi "insufficient context".

### Test Results

```
======================================== 42 passed in 0.06s ========================================
```

**So tests pass:** 42 / 42

**OpenAI smoke test:** Da chay thanh cong tren 2 file markdown clean (`vf3_vn_warranty.md`, `VFSC_Code_of_Conduct.md`), luu store, search top-3 va sinh answer bang `gpt-4o-mini`.

---

## 5. Similarity Predictions — Ca nhan (5 diem)

| Pair | Sentence A | Sentence B | Du doan | Actual Score | Dung? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | VF3 warranty covers repair for manufacturing defects. | Vehicle warranty includes repairs for manufacturer defects. | high | 0.6595 | Yes |
| 2 | Code of conduct requires ethical behavior. | Employees must follow integrity and ethics guidelines. | high | 0.6544 | Yes |
| 3 | The first responder guide explains emergency handling. | Sustainability report discusses carbon emissions. | low | 0.1696 | Yes |
| 4 | Bao hanh pin co dieu kien ap dung rieng. | Chinh sach bao hanh xe dien co pham vi cu the. | high | 0.5451 (medium) | Partly |
| 5 | Audit committee oversees compliance and governance. | How to charge the VF3 battery at home safely. | low | 0.0266 | Yes |

**Ket qua nao bat ngo nhat? Dieu nay noi gi ve cach embeddings bieu dien nghia?**  
Pair 4 bat ngo nhat: chu de giong nhau nhung score o muc medium. Dieu nay cho thay embedding van phan biet manh theo cach dien dat chi tiet va context.

---

## 6. Results — Ca nhan (10 diem)

Chay lai benchmark theo bo golden queries nhom thong nhat tren 5 tai lieu markdown clean tuong ung.

### Benchmark Queries & Gold Answers (nhom thong nhat)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | What is the battery capacity and range of the VinFast VF3? | Battery type LFP, capacity 18.64 kWh, range ~210 km (NEDC). |
| 2 | Thời hạn bảo hành chung của xe VinFast VF3 là bao lâu? | 7 năm/160.000 km (thường), 3 năm/100.000 km (thương mại), tính từ ngày kích hoạt bảo hành. |
| 3 | Những hư hỏng nào không được VinFast bảo hành? | Hư hỏng do sửa chữa trái phép, lạm dụng, thiên tai/tai nạn, hao mòn tự nhiên, phụ tùng không chính hãng. |
| 4 | How should first responders handle a VinFast VF8 high-voltage battery fire? | Assume HV live, wear PPE, use large water volume, monitor re-ignition risk. |
| 5 | What is the battery warranty period for VinFast vehicles? | VF3 (VN) battery warranty key point: 8 years or 160,000 km (non-commercial); cần filter theo model/doc_type. |

### Ket Qua Cua Toi

| # | Query | Top-1 Retrieved Chunk (tom tat) | Score | Relevant? | Agent Answer (tom tat) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | VF3 battery capacity/range | Top-1 lai roi vao VF9 US warranty period, khong chua thong so 18.64 kWh va 210 km. | 0.6261 | No | Agent bao context chua du de tra loi thong so VF3. |
| 2 | Thoi han bao hanh chung VF3 | Chunk top-1 tu vf3_vn_warranty noi ve thoi han bao hanh va dieu kien. | 0.7082 | Yes | Agent tra loi dung 7 nam/160.000 km va moi lien he voi ngay kich hoat bao hanh. |
| 3 | Hu hong khong duoc bao hanh | Chunk top-1 tu vf3_vn_warranty co danh sach loai tru (thien tai, tai nan, hao mon...). | 0.7111 | Yes | Agent liet ke dung cac nhom hư hong khong thuoc pham vi bao hanh. |
| 4 | VF8 HV battery fire handling | Chunk top-1 tu VF8 first responder guide noi ve danger, thermal runaway, xu ly chay. | 0.7197 | Yes | Agent neu dung PPE, dung nhieu nuoc, va theo doi nguy co tai boc chay. |
| 5 | Battery warranty period (cross-doc) | Top-1 sau filter (model=VF3, doc_type=warranty) van chua du thong tin "8 nam" ro rang trong top-3. | 0.6318 | No | Agent tra loi co canh bao thieu thong tin pin bao hanh cu the trong context da retrieve. |

**Bao nhieu queries tra ve chunk relevant trong top-3?** 3 / 5

**Ghi chu Query 5 (filter test):** Khong filter thi top-1 den tu `VF9 US Vehicle Warranty Booklet` (score 0.7773); co filter `model=VF3, doc_type=warranty` thi top-1 chuyen ve `vf3_vn_warranty` (score 0.6318), nhung van chua dat tieu chi keyword trong top-3.

---

## 7. What I Learned (5 diem — Demo)

**Dieu hay nhat toi hoc duoc tu thanh vien khac trong nhom:**  
Tang overlap cho fixed-size co the giup giu context o query can thong tin sat bien chunk.

**Dieu hay nhat toi hoc duoc tu nhom khac (qua demo):**  
Them metadata business-oriented (document_type, effective_date, department) tang hieu qua filter ro ret.

**Neu lam lai, toi se thay doi gi trong data strategy?**  
Toi se bo sung clean pipeline bat buoc cho OCR markdown ngay tu dau, bat buoc pre-chunk truoc embedding, va them cache/re-ranker de toi uu top-1 precision.

---

## Tu Danh Gia

| Tieu chi | Loai | Diem tu danh gia |
|----------|------|-------------------|
| Warm-up | Ca nhan | 5 / 5 |
| Document selection | Nhom | 9 / 10 |
| Chunking strategy | Nhom | 14 / 15 |
| My approach | Ca nhan | 9 / 10 |
| Similarity predictions | Ca nhan | 5 / 5 |
| Results | Ca nhan | 6 / 10 |
| Core implementation (tests) | Ca nhan | 30 / 30 |
| Demo | Nhom | 4 / 5 |
| **Tong** | | **82 / 100** |
