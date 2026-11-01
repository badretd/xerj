"""Size and CPU of the merge-path zstd candidates on the two big stored columns of a semantic autoindex record:
the `text` column (Wikipedia passages) and the `text_vector` companion (lexical feature-hashing embedding, packed as
`xerj:f32:` + hex, the format xerj_common::packed_f32 writes). Columns are serialized as the stored codec does
(serde_json::to_vec of the column's values = a JSON array). Source of the offline table in NEWCORE_MAP.md 7.3.

usage: python3 column_codec_offline.py CORPUS.jsonl [N=20000]   (needs the `zstandard` package; one core)

The vectors come from a Python port of the lexical embedder's scheme (FNV-1a feature hashing of lower-cased
alphanumeric tokens and their boundary-marked character trigrams, 384 dims, L2-normalized) over each whole text, without
the 512-character chunking: good for sizing the codec on realistic packed vectors, not asserted bit-identical to
`xerj_ai::local::local_embed`."""
import json, struct, sys, time, math, unicodedata
import zstandard as zstd

N = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
texts = []
with open(sys.argv[1], encoding="utf-8") as f:
    for line in f:
        texts.append(json.loads(line)["text"])
        if len(texts) >= N:
            break

def fnv1a64(b):
    h = 0xcbf29ce484222325
    for x in b:
        h ^= x
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h

def local_embed(text, dims=384):
    acc = [0.0] * dims
    lower = text.lower()
    tok = []
    cur = []
    for ch in lower:
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur: tok.append("".join(cur)); cur = []
    if cur: tok.append("".join(cur))
    def add(b, w):
        h = fnv1a64(b)
        acc[h % dims] += w * (1.0 if (h >> 63) & 1 == 0 else -1.0)
    for t in tok:
        add(t.encode(), 1.0)
        p = "#" + t + "#"
        for i in range(len(p) - 2):
            add(p[i:i+3].encode(), 0.35)
    n = math.sqrt(sum(x * x for x in acc))
    if n > 0:
        acc = [x / n for x in acc]
    return acc

def pack(v):
    return "xerj:f32:" + b"".join(struct.pack("<f", x) for x in v).hex()

t0 = time.time()
vecs = [pack(local_embed(t)) for t in texts]
print(f"embedded {len(texts)} in python {time.time()-t0:.1f}s", file=sys.stderr)
cols = {"text": json.dumps(texts, ensure_ascii=False).encode(), "text_vector": json.dumps(vecs).encode()}
raw = b"".join(bytes.fromhex(v[len("xerj:f32:"):]) for v in vecs)
d = len(raw) // len(vecs)
shuf = bytes(raw[i] for k in range(4) for i in range(k, len(raw), 4))
cols["vector_f32_binary"] = raw
cols["vector_f32_byteshuffled"] = shuf

def long_effort(data):
    wl = min(24, max(10, (len(data) - 1).bit_length()))
    p = zstd.ZstdCompressionParameters.from_level(19, window_log=wl, enable_ldm=True)
    return zstd.ZstdCompressor(compression_params=p).compress(data)

out = {}
for name, data in cols.items():
    row = {"raw_bytes": len(data)}
    for label, fn in [("zstd3", lambda d: zstd.ZstdCompressor(level=3).compress(d)),
                      ("zstd9", lambda d: zstd.ZstdCompressor(level=9).compress(d)),
                      ("zstd3_ldm24", lambda d: zstd.ZstdCompressor(compression_params=zstd.ZstdCompressionParameters.from_level(3, window_log=min(24, max(10, (len(d)-1).bit_length())), enable_ldm=True)).compress(d)),
                      ("zstd6_ldm24", lambda d: zstd.ZstdCompressor(compression_params=zstd.ZstdCompressionParameters.from_level(6, window_log=min(24, max(10, (len(d)-1).bit_length())), enable_ldm=True)).compress(d)),
                      ("zstd9_ldm24", lambda d: zstd.ZstdCompressor(compression_params=zstd.ZstdCompressionParameters.from_level(9, window_log=min(24, max(10, (len(d)-1).bit_length())), enable_ldm=True)).compress(d)),
                      ("zstd12_ldm24", lambda d: zstd.ZstdCompressor(compression_params=zstd.ZstdCompressionParameters.from_level(12, window_log=min(24, max(10, (len(d)-1).bit_length())), enable_ldm=True)).compress(d)),
                      ("zstd15_ldm24", lambda d: zstd.ZstdCompressor(compression_params=zstd.ZstdCompressionParameters.from_level(15, window_log=min(24, max(10, (len(d)-1).bit_length())), enable_ldm=True)).compress(d)),
                      ("zstd19_ldm24", long_effort)]:
        t = time.process_time()
        c = fn(data)
        dt = time.process_time() - t
        row[label] = {"bytes": len(c), "ratio": round(len(data) / len(c), 3), "cpu_s": round(dt, 3), "MB_per_s": round(len(data) / 1e6 / max(dt, 1e-9), 1)}
    out[name] = row
print(json.dumps(out, indent=1))
