import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup, logging
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import sys
import io

# ───────────────────────────────────────────────
# 0. 설정
# ───────────────────────────────────────────────
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.set_verbosity_error()

FPS_GAMES  = ["PLAYERUNKNOWN'S BATTLEGROUNDS", "Rust"]
SAMPLE_N   = 25000
BATCH_SIZE = 16
EPOCH      = 4
MAX_LEN    = 128
N_TOPICS   = 5

STOPWORDS = set([
    "the","a","an","and","or","but","in","on","at","to","for","of","is","it",
    "i","my","this","that","was","are","be","have","has","with","so","its",
    "not","just","game","games","you","your","they","their","we","me","if",
    "do","get","can","all","no","more","very","from","as","up","out","about",
    "when","one","there","by","been","had","what","s","t","re","ve","will",
    "would","could","should","also","than","like","some","time","then","because",
    "don","really","even","still","only","much","other","lot","after","many",
    "too","every","ing","dont","did","got","into","most","now","buy","bought",
    "played","playing","play","hours","hour","steam","early","access","product",
    "received","free"
])

# ───────────────────────────────────────────────
# TXT 출력 설정 (터미널 + 파일 동시 출력)
# ───────────────────────────────────────────────
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
    def isatty(self):
        return False

txt_file = open("fps_analysis_report.txt", "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, txt_file)

print("사용 장치:", device)

# ───────────────────────────────────────────────
# 1. 데이터 로드 및 FPS 필터링
# ───────────────────────────────────────────────
print("\n=== 데이터 로드 ===")
df  = pd.read_csv("steam_data/steam_reviews.csv")
fps = df[df["title"].isin(FPS_GAMES)].dropna(subset=["review"]).copy()
fps["label"]       = (fps["recommendation"] == "Recommended").astype(int)
fps["date_posted"] = pd.to_datetime(fps["date_posted"])
fps["year"]        = fps["date_posted"].dt.year

print(fps.groupby("title").agg(
    리뷰수=("recommendation", "count"),
    추천율=("label", lambda x: f"{x.mean()*100:.1f}%")
))
print(f"총 리뷰: {len(fps):,}건")

# ───────────────────────────────────────────────
# 2. MobileBERT 학습용 균형 샘플링
# ───────────────────────────────────────────────
print(f"\n=== MobileBERT 학습 데이터 샘플링 ({SAMPLE_N:,}건) ===")
half = SAMPLE_N // 2
pos_train = fps[fps["label"] == 1].sample(n=min(half, (fps["label"]==1).sum()), random_state=42)
neg_train = fps[fps["label"] == 0].sample(n=min(half, (fps["label"]==0).sum()), random_state=42)
train_df  = pd.concat([pos_train, neg_train]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"긍정: {len(pos_train):,} / 부정: {len(neg_train):,}")

# ───────────────────────────────────────────────
# 3. 토큰화
# ───────────────────────────────────────────────
print("\n=== 토큰화 중 ===")
tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
inputs    = tokenizer(
    list(train_df["review"].values),
    truncation=True, max_length=MAX_LEN,
    add_special_tokens=True, padding="max_length"
)
labels_arr = train_df["label"].values

tx, vx, ty, vy = train_test_split(inputs["input_ids"],      labels_arr, test_size=0.2, random_state=2026)
tm, vm, _,  _  = train_test_split(inputs["attention_mask"], labels_arr, test_size=0.2, random_state=2026)

def make_loader(ids, masks, labels, shuffle):
    ds = TensorDataset(torch.tensor(ids), torch.tensor(masks), torch.tensor(labels))
    sampler = RandomSampler(ds) if shuffle else SequentialSampler(ds)
    return DataLoader(ds, sampler=sampler, batch_size=BATCH_SIZE)

train_loader = make_loader(tx, tm, ty, shuffle=True)
valid_loader = make_loader(vx, vm, vy, shuffle=False)

# ───────────────────────────────────────────────
# 4. MobileBERT 학습
# ───────────────────────────────────────────────
print("\n=== MobileBERT 학습 시작 ===")
model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=2)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0,
    num_training_steps=len(train_loader) * EPOCH
)

epoch_results = []
for e in range(EPOCH):
    model.train()
    total_loss = 0.0
    bar = tqdm(train_loader, desc=f"Training {e+1}/{EPOCH}", leave=False)
    for batch in bar:
        batch = tuple(t.to(device) for t in batch)
        ids, mask, batch_labels = batch
        model.zero_grad()
        outputs = model(ids, attention_mask=mask, labels=batch_labels)
        loss = outputs.loss
        total_loss += loss.item() / len(train_loader)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        bar.set_postfix({"loss": f"{loss.item():.4f}"})

    model.eval()
    preds_all, true_all = [], []
    for batch in tqdm(valid_loader, desc=f"Validation {e+1}/{EPOCH}", leave=False):
        batch = tuple(t.to(device) for t in batch)
        ids, mask, batch_labels = batch
        with torch.no_grad():
            outputs = model(ids, attention_mask=mask)
        preds = torch.argmax(outputs.logits, dim=-1)
        preds_all.extend(preds.cpu().numpy())
        true_all.extend(batch_labels.cpu().numpy())
    valid_acc = np.sum(np.array(preds_all) == np.array(true_all)) / len(preds_all)
    epoch_results.append((total_loss, valid_acc))
    print(f"Epoch {e+1}: loss={total_loss:.4f} | valid_acc={valid_acc:.4f}")

print("\n=== 학습 완료 ===")
for idx, (loss, vacc) in enumerate(epoch_results, 1):
    print(f"Epoch {idx}: loss={loss:.4f} | valid_acc={vacc:.4f}")

model.save_pretrained("mobilebert_fps")
tokenizer.save_pretrained("mobilebert_fps")
print("모델 저장 완료 → ./mobilebert_fps/")

# ───────────────────────────────────────────────
# 5. 전체 FPS 리뷰 감성 예측
# ───────────────────────────────────────────────
print(f"\n=== 전체 리뷰 감성 예측 ({len(fps):,}건) ===")
model.eval()
all_texts  = list(fps["review"].values)
all_preds  = []
INFER_BATCH = 64

for i in tqdm(range(0, len(all_texts), INFER_BATCH), desc="예측 중"):
    batch_texts = all_texts[i:i+INFER_BATCH]
    enc = tokenizer(
        batch_texts, truncation=True, max_length=MAX_LEN,
        add_special_tokens=True, padding="max_length",
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc)
    preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
    all_preds.extend(preds)

fps["bert_pred"] = all_preds
bert_pos_rate = np.mean(all_preds) * 100
print(f"MobileBERT 긍정 예측 비율: {bert_pos_rate:.1f}%")
print(f"원본 추천율: {fps['label'].mean()*100:.1f}%")

# ───────────────────────────────────────────────
# 6. 텍스트 전처리
# ───────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r"[^a-zA-Z ]", " ", text.lower())
    return " ".join(w for w in text.split() if w not in STOPWORDS and len(w) > 2)

fps["clean_review"] = fps["review"].apply(clean_text)

bert_pos = fps[fps["bert_pred"] == 1]["clean_review"]
bert_neg = fps[fps["bert_pred"] == 0]["clean_review"]
print(f"\nBERT 긍정 리뷰: {len(bert_pos):,}건 / 부정 리뷰: {len(bert_neg):,}건")

# ───────────────────────────────────────────────
# 7. 긍정/부정 키워드 Top 20
# ───────────────────────────────────────────────
def top_keywords(texts, n=20):
    words = []
    for t in texts:
        words += t.split()
    return Counter(words).most_common(n)

pos_kw = top_keywords(bert_pos)
neg_kw = top_keywords(bert_neg)

print("\n=== [BERT 기반] 긍정 Top 20 키워드 ===")
for w, c in pos_kw:
    print(f"  {w}: {c:,}")

print("\n=== [BERT 기반] 부정 Top 20 키워드 ===")
for w, c in neg_kw:
    print(f"  {w}: {c:,}")

# ───────────────────────────────────────────────
# 8. TF-IDF 게임별 특징 키워드
# ───────────────────────────────────────────────
print("\n=== [BERT 기반] TF-IDF 게임별 특징 키워드 ===")
for game in FPS_GAMES:
    subset = fps[(fps["title"] == game) & (fps["bert_pred"] == 1)]["clean_review"]
    tfidf  = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    tfidf.fit(subset)
    scores = zip(tfidf.get_feature_names_out(), tfidf.idf_)
    top    = sorted(scores, key=lambda x: x[1])[:15]
    print(f"\n[{game}] BERT 긍정 리뷰 특징 키워드:")
    for w, s in top:
        print(f"  {w} (idf={s:.2f})")

# ───────────────────────────────────────────────
# 9. LDA 토픽 모델링
# ───────────────────────────────────────────────
print("\n=== [BERT 기반] LDA 토픽 모델링 ===")
pos_sample = bert_pos.sample(n=min(25000, len(bert_pos)), random_state=42)
tfidf_lda  = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X          = tfidf_lda.fit_transform(pos_sample)

lda = LatentDirichletAllocation(
    n_components=N_TOPICS, random_state=42,
    max_iter=20, learning_method="batch"
)
lda.fit(X)

feature_names = tfidf_lda.get_feature_names_out()
topic_labels  = ["핵심 게임플레이", "멀티플레이 / 친구", "그래픽 / 몰입감", "가성비 / 추천", "생존 / 전략"]
for idx, (topic, label) in enumerate(zip(lda.components_, topic_labels)):
    top_words = [feature_names[i] for i in topic.argsort()[:-16:-1]]
    print(f"\n토픽 {idx+1} [{label}]")
    print("  키워드:", ", ".join(top_words))

# ───────────────────────────────────────────────
# 10. 연도별 / 플레이타임별 추천율
# ───────────────────────────────────────────────
yearly = fps.groupby(["year", "title"])["bert_pred"].mean().mul(100).round(1).reset_index()
yearly.columns = ["year", "title", "BERT추천율(%)"]
print("\n=== [BERT 기반] 연도별 추천율 ===")
print(yearly.to_string(index=False))

fps["playtime_group"] = pd.cut(
    fps["hour_played"],
    bins=[0, 10, 50, 200, 500, 99999],
    labels=["0-10h", "10-50h", "50-200h", "200-500h", "500h+"]
)
pt = fps.groupby(["playtime_group", "title"])["bert_pred"].mean().mul(100).round(1).reset_index()
pt.columns = ["플레이타임", "게임", "BERT추천율(%)"]
print("\n=== [BERT 기반] 플레이타임별 추천율 ===")
print(pt.to_string(index=False))

# ───────────────────────────────────────────────
# 11. 종합 요약
# ───────────────────────────────────────────────
pos_top5 = [w for w, _ in pos_kw[:5]]
neg_top5 = [w for w, _ in neg_kw[:5]]
pubg_rt  = fps[fps["title"] == "PLAYERUNKNOWN'S BATTLEGROUNDS"]["bert_pred"].mean() * 100
rust_rt  = fps[fps["title"] == "Rust"]["bert_pred"].mean() * 100

print("\n" + "="*60)
print("FPS 장르 흥행요인 종합 요약 (MobileBERT 기반)")
print("="*60)
PUBG_NAME  = "PLAYERUNKNOWN'S BATTLEGROUNDS"
pubg_df    = fps[fps["title"] == PUBG_NAME]
rust_df    = fps[fps["title"] == "Rust"]
pubg_total = len(pubg_df)
pubg_pos   = (pubg_df["bert_pred"] == 1).sum()
pubg_neg   = (pubg_df["bert_pred"] == 0).sum()
rust_total = len(rust_df)
rust_pos   = (rust_df["bert_pred"] == 1).sum()
rust_neg   = (rust_df["bert_pred"] == 0).sum()

print(f"\n【 📊 감성 분석 결과 요약 】")
print(f"  - PUBG  총 리뷰: {pubg_total:,}건")
print(f"    * 긍정: {pubg_pos:,}건 ({pubg_rt:.1f}%)")
print(f"    * 부정: {pubg_neg:,}건 ({100-pubg_rt:.1f}%)")
print(f"  - Rust  총 리뷰: {rust_total:,}건")
print(f"    * 긍정: {rust_pos:,}건 ({rust_rt:.1f}%)")
print(f"    * 부정: {rust_neg:,}건 ({100-rust_rt:.1f}%)")
print(f"\n[공통 흥행요인 - 긍정]")
print(f"  핵심 키워드: {', '.join(pos_top5)}")
print(f"\n[공통 이탈요인 - 부정]")
print(f"  핵심 키워드: {', '.join(neg_top5)}")
print(f"\n[게임별 BERT 추천율]")
print(f"  PUBG: {pubg_rt:.1f}%")
print(f"  Rust: {rust_rt:.1f}%")
print("="*60)

# ───────────────────────────────────────────────
# 12. 시각화
# ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("FPS 장르 흥행요인 분석 (MobileBERT 기반)", fontsize=16, fontweight="bold")

# ── 좌상단: PUBG 긍/부정 파이차트 ──
ax = axes[0, 0]
pubg_pos_cnt = (fps[fps["title"] == "PLAYERUNKNOWN'S BATTLEGROUNDS"]["bert_pred"] == 1).sum()
pubg_neg_cnt = (fps[fps["title"] == "PLAYERUNKNOWN'S BATTLEGROUNDS"]["bert_pred"] == 0).sum()
ax.pie(
    [pubg_pos_cnt, pubg_neg_cnt],
    labels=["긍정", "부정"],
    colors=["#1D9E75", "#E24B4A"],
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
ax.set_title("PUBG 긍/부정 비율 (BERT 기반)", fontsize=13, fontweight="bold")

# ── 우상단: Rust 긍/부정 파이차트 ──
ax = axes[0, 1]
rust_pos_cnt = (fps[fps["title"] == "Rust"]["bert_pred"] == 1).sum()
rust_neg_cnt = (fps[fps["title"] == "Rust"]["bert_pred"] == 0).sum()
ax.pie(
    [rust_pos_cnt, rust_neg_cnt],
    labels=["긍정", "부정"],
    colors=["#1D9E75", "#E24B4A"],
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
ax.set_title("Rust 긍/부정 비율 (BERT 기반)", fontsize=13, fontweight="bold")

# ── 좌하단: 연도별 추천율 ──
ax = axes[1, 0]
for game, color in zip(FPS_GAMES, ["#1D9E75", "#7F77DD"]):
    subset = yearly[yearly["title"] == game]
    ax.plot(subset["year"], subset["BERT추천율(%)"], marker="o", label=game, color=color)
ax.set_title("연도별 BERT 추천율 변화")
ax.set_ylabel("추천율 (%)")
ax.set_ylim(0, 100)
ax.legend()
ax.grid(axis="y", alpha=0.3)

# ── 우하단: 플레이타임별 추천율 ──
ax = axes[1, 1]
pt_pivot = pt.pivot(index="플레이타임", columns="게임", values="BERT추천율(%)")
x = np.arange(len(pt_pivot))
width = 0.35
for i, (game, color) in enumerate(zip(FPS_GAMES, ["#1D9E75", "#7F77DD"])):
    if game in pt_pivot.columns:
        ax.bar(x + i*width, pt_pivot[game], width, label=game, color=color)
ax.set_title("플레이타임별 BERT 추천율")
ax.set_ylabel("추천율 (%)")
ax.set_xticks(x + width/2)
ax.set_xticklabels(pt_pivot.index)
ax.set_ylim(0, 100)
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("fps_success_factors.png", dpi=150, bbox_inches="tight")
print("\n시각화 저장 완료 → fps_success_factors.png")

# txt 파일 닫기
sys.stdout = sys.__stdout__
txt_file.close()
print("분석 리포트 저장 완료 → fps_analysis_report.txt")