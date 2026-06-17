import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ───────────────────────────────────────────────
# 1. 데이터 로드 및 FPS 필터링
# ───────────────────────────────────────────────
FPS_GAMES = ["PLAYERUNKNOWN'S BATTLEGROUNDS", "Rust"]

df  = pd.read_csv("steam_data/steam_reviews.csv")
fps = df[df["title"].isin(FPS_GAMES)].dropna(subset=["review"]).copy()
fps["label"] = (fps["recommendation"] == "Recommended").astype(int)

print(f"전체 FPS 리뷰: {len(fps):,}건")

# ───────────────────────────────────────────────
# 2. 2,000건 랜덤 샘플링
# ───────────────────────────────────────────────
sample = fps.sample(n=2000, random_state=42).reset_index(drop=True)
print(f"\n=== 라벨 품질 검증 샘플 (2,000건) ===")
print(f"긍정(1): {sample['label'].sum()}건 / 부정(0): {(sample['label']==0).sum()}건")

# ───────────────────────────────────────────────
# 3. 중립 판별 함수
# ───────────────────────────────────────────────
# 중립 키워드: 긍정도 부정도 아닌 애매한 표현
NEUTRAL_PATTERNS = [
    r"\bokay\b", r"\bok\b", r"\balright\b", r"\bnot bad\b",
    r"\bnot great\b", r"\bnot sure\b", r"\bmixed\b",
    r"\bso so\b", r"\bmeh\b", r"\baverage\b", r"\bdecent\b",
    r"\bnothing special\b", r"\bnothing terrible\b",
    r"\bhas (good|bad) and (bad|good)\b",
    r"\bsome good.*some bad\b", r"\bboth good.*bad\b",
    r"\bit depends\b", r"\bhard to (say|tell|judge)\b",
]

# 강한 긍정 키워드
STRONG_POS = [
    r"\bamazing\b", r"\bexcellent\b", r"\bperfect\b", r"\boutstanding\b",
    r"\bfantastic\b", r"\blove (this|it)\b", r"\bbest game\b",
    r"\bhighly recommend\b", r"\b10/10\b", r"\bmasterpiece\b",
]

# 강한 부정 키워드
STRONG_NEG = [
    r"\bterrible\b", r"\bawful\b", r"\bhorrible\b", r"\bgarbage\b",
    r"\bwaste of money\b", r"\brefund\b", r"\buninstall\b",
    r"\bdo not buy\b", r"\bdon't buy\b", r"\bworst game\b",
    r"\bhackers\b", r"\bcheaters\b",
]

def classify_review(row):
    text = str(row["review"]).lower()
    label = row["label"]

    # 중립 패턴 체크
    neutral_score = sum(1 for p in NEUTRAL_PATTERNS if re.search(p, text))
    pos_score     = sum(1 for p in STRONG_POS if re.search(p, text))
    neg_score     = sum(1 for p in STRONG_NEG if re.search(p, text))

    # 중립 판정: 중립 키워드 있고 강한 감성 없는 경우
    if neutral_score >= 1 and pos_score == 0 and neg_score == 0:
        return "neutral"

    # 라벨-텍스트 불일치 판정
    if label == 1 and neg_score >= 2 and pos_score == 0:
        return "mismatch"
    if label == 0 and pos_score >= 2 and neg_score == 0:
        return "mismatch"

    # 명확한 긍정
    if label == 1:
        return "positive"

    # 명확한 부정
    return "negative"

print("\n=== 라벨 분류 중... ===")
sample["verify_label"] = sample.apply(classify_review, axis=1)

# ───────────────────────────────────────────────
# 4. 검증 결과 출력
# ───────────────────────────────────────────────
result = sample["verify_label"].value_counts()
print("\n=== 2,000건 라벨 검증 결과 ===")
for label, count in result.items():
    print(f"  {label:10s}: {count:4d}건 ({count/20:.1f}%)")

# 중립/불일치 샘플 출력
print("\n=== 중립 리뷰 샘플 (5건) ===")
neutral_samples = sample[sample["verify_label"] == "neutral"].head(5)
for _, row in neutral_samples.iterrows():
    print(f"  [{row['title'][:10]}] {str(row['review'])[:100]}...")

print("\n=== 불일치 리뷰 샘플 (5건) ===")
mismatch_samples = sample[sample["verify_label"] == "mismatch"].head(5)
for _, row in mismatch_samples.iterrows():
    print(f"  [라벨:{row['label']}] {str(row['review'])[:100]}...")

# ───────────────────────────────────────────────
# 5. 전체 데이터에서 중립/불일치 제거
# ───────────────────────────────────────────────
print("\n=== 전체 데이터 중립/불일치 제거 ===")
fps["verify_label"] = fps.apply(classify_review, axis=1)

before = len(fps)
fps_clean = fps[fps["verify_label"].isin(["positive", "negative"])].copy()
after = len(fps_clean)

print(f"제거 전: {before:,}건")
print(f"제거 후: {after:,}건")
print(f"제거된 중립/불일치: {before - after:,}건 ({(before-after)/before*100:.1f}%)")
print(f"\n정제 후 긍정: {(fps_clean['label']==1).sum():,}건 ({fps_clean['label'].mean()*100:.1f}%)")
print(f"정제 후 부정: {(fps_clean['label']==0).sum():,}건 ({(1-fps_clean['label'].mean())*100:.1f}%)")


fps_clean[["title", "review", "recommendation", "label", "date_posted", "hour_played"]]\
    .to_csv("steam_fps_clean.csv", index=False, encoding="utf-8")
print("\n정제 데이터 저장 완료 → steam_fps_clean.csv")

# ───────────────────────────────────────────────
# 6. 시각화
# ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("라벨 품질 검증 결과 (2,000건 샘플)", fontsize=14, fontweight="bold")


ax = axes[0]
labels_kr = {"positive": "명확한 긍정", "negative": "명확한 부정",
             "neutral": "중립", "mismatch": "라벨 불일치"}
sizes  = [result.get(k, 0) for k in ["positive", "negative", "neutral", "mismatch"]]
colors = ["#1D9E75", "#E24B4A", "#FFA500", "#7F77DD"]
ax.pie(sizes, labels=[labels_kr[k] for k in ["positive", "negative", "neutral", "mismatch"]],
       colors=colors, autopct="%1.1f%%", startangle=90,
       wedgeprops={"edgecolor": "white", "linewidth": 2})
ax.set_title("2,000건 라벨 검증 분포")


ax = axes[1]
categories = ["정제 전\n전체 데이터", "정제 후\n(중립 제거)"]
pos_counts = [fps["label"].sum(), fps_clean["label"].sum()]
neg_counts = [(fps["label"]==0).sum(), (fps_clean["label"]==0).sum()]
x = np.arange(2)
width = 0.35
ax.bar(x - width/2, pos_counts, width, label="긍정", color="#1D9E75")
ax.bar(x + width/2, neg_counts, width, label="부정", color="#E24B4A")
ax.set_title("정제 전후 긍/부정 분포 비교")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel("리뷰 수")
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("label_verification.png", dpi=150, bbox_inches="tight")
print("시각화 저장 완료 → label_verification.png")
