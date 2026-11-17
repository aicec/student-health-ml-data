"""교육부 「학생건강검사 표본조사 rawdata」(2025, 공공데이터포털 15100360, 이용허락범위 제한 없음)
→ 수업용 표 3종.

원본 CSV(utf-8-sig · 92,620줄 · 110열)에서
  * 혈압이 있는 학년(초4 · 중1 · 고1)만 남기고
  * 혈액검사(비만 학생만 받음 → 편향)와 구강 · 설문 민감 항목은 빼고
  * 코드값을 한글로 풀어
  * 학년군별 1,000명씩 3,000명(수업용) + 별도 500명(숨긴 답 대회)을 뽑는다.

실행:  python convert_health.py <원본 csv 경로> [--seed 20261117]
출력:  public/student_health_3000.csv
       public/hidden_500_input.csv   (성별 열 없음)
       private/hidden_500_answer.csv (공개 안 함)
       public/README_columns.md      (열 사전 초안)
"""
import sys, argparse, pathlib
import pandas as pd
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

FREQ = {1: "안 먹음", 2: "주 1~2번", 3: "주 3~5번", 4: "매일"}
BREAKFAST = {1: "거의 꼭 먹음", 2: "대체로 먹음", 3: "대체로 안 먹음", 4: "거의 안 먹음"}
SLEEP = {1: "6시간 이내", 2: "6~7시간", 3: "7~8시간", 4: "8시간 이상"}
BODYIMG = {1: "매우 마른 편", 2: "약간 마른 편", 3: "보통", 4: "약간 살찐 편", 5: "매우 살찐 편"}
YESNO = {1: "예", 2: "아니오"}
EX_MID = {1: "거의 안 함", 2: "주 1~2일", 3: "주 3~4일", 4: "주 5일 이상"}


def grade_group(r):
    if r["학교급"] == "초" and r["학년"] == 4:
        return "초4"
    if r["학교급"] == "중" and r["학년"] == 1:
        return "중1"
    if r["학교급"] == "고" and r["학년"] == 1:
        return "고1"
    return None


def exercise(r):
    if r["학교급"] == "초":
        v = r.get("주3회이상운동")
        if v == 1:
            return "주 3일 이상"
        if v == 2:
            return "주 3일 미만"
        return None
    return EX_MID.get(r.get("하루30분이상운동"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--seed", type=int, default=20261117)
    a = ap.parse_args()

    df = pd.read_csv(a.src, encoding="utf-8-sig", low_memory=False)
    df.columns = [c.replace("﻿", "") for c in df.columns]
    df["학년군"] = df.apply(grade_group, axis=1)
    df = df[df["학년군"].notna()].copy()
    df = df[df["수축기혈압"].notna() & df["이완기혈압"].notna()]
    df = df[(df["수축기혈압"].between(70, 200)) & (df["이완기혈압"].between(40, 130))]
    df = df[df["키"].between(100, 200) & df["몸무게"].between(15, 150)]

    out = pd.DataFrame({
        "학년군": df["학년군"],
        "성별": df["성별"],
        "도시규모": df["도시규모"],
        "키_cm": df["키"],
        "몸무게_kg": df["몸무게"],
        "체질량_등급": df["체질량지수"],
        "수축기혈압": df["수축기혈압"].astype(int),
        "이완기혈압": df["이완기혈압"].astype(int),
        "라면": df["라면"].map(FREQ),
        "음료수": df["음료수"].map(FREQ),
        "패스트푸드": df["패스트푸드"].map(FREQ),
        "육류": df["육류"].map(FREQ),
        "우유유제품": df["우유유제품"].map(FREQ),
        "과일": df["과일"].map(FREQ),
        "채소": df["채소김치제외"].map(FREQ),
        "아침식사": df["아침식사"].map(BREAKFAST),
        "운동": df.apply(exercise, axis=1),
        "수면": df["하루수면량"].map(SLEEP),
        "자기_체형_인식": df["자아신체상체형"].map(BODYIMG),
        "게임_2시간이상": df["2시간이상게임"].map(YESNO),
    })
    out = out.dropna()
    print("후보 행:", len(out), out["학년군"].value_counts().to_dict())

    rng = np.random.default_rng(a.seed)
    parts, hidden = [], []
    for g, n_train, n_hid in [("초4", 1000, 170), ("중1", 1000, 165), ("고1", 1000, 165)]:
        sub = out[out["학년군"] == g]
        idx = rng.permutation(len(sub))
        parts.append(sub.iloc[idx[:n_train]])
        hidden.append(sub.iloc[idx[n_train:n_train + n_hid]])
    train = pd.concat(parts).sample(frac=1, random_state=a.seed).reset_index(drop=True)
    hid = pd.concat(hidden).sample(frac=1, random_state=a.seed + 1).reset_index(drop=True)
    hid.insert(0, "번호", range(1, len(hid) + 1))

    pub = HERE / "public"; prv = HERE / "private"
    pub.mkdir(exist_ok=True); prv.mkdir(exist_ok=True)
    train.to_csv(pub / "student_health_3000.csv", index=False, encoding="utf-8")
    hid.drop(columns=["성별"]).to_csv(pub / "hidden_500_input.csv", index=False, encoding="utf-8")
    hid[["번호", "성별"]].to_csv(prv / "hidden_500_answer.csv", index=False, encoding="utf-8")

    # 수업 거리 숫자
    print("3000명 학년군:", train["학년군"].value_counts().to_dict())
    print("성별:", train["성별"].value_counts().to_dict())
    print("체질량 등급:", train["체질량_등급"].value_counts().to_dict())
    print(train.groupby("학년군")[["키_cm", "몸무게_kg", "수축기혈압"]].mean().round(1))
    print(train.groupby("성별")[["키_cm", "몸무게_kg", "수축기혈압"]].mean().round(1))
    print("키-수축기혈압 상관:", round(train["키_cm"].corr(train["수축기혈압"]), 3),
          " 몸무게-수축기혈압:", round(train["몸무게_kg"].corr(train["수축기혈압"]), 3))
    print("숨긴 500 성별:", hid["성별"].value_counts().to_dict())


if __name__ == "__main__":
    main()
