#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRiP(cast) skorunun dış validasyonu – tekrarlanabilir analiz betiği
External validation of the TRiP(cast) score – reproducible analysis script

Kullanım / Usage:
    python trip_cast_analysis.py --data veri.xlsx --out sonuclar/

Girdi: 25 sütunlu Excel dosyası (sütun sırası README'de tanımlanmıştır).
Input: 25-column Excel file (column order defined in README).

Çıktılar / Outputs (--out klasörüne):
    tablo2_temel_ozellikler.csv        Tablo 2  – temel özellikler
    tablo3_kesim_degerleri.csv         Tablo 3  – kesim değerleri + yaş kuralları
    tablo4_bilesenler.csv              Tablo 4  – skor bileşenleri (tek değişkenli)
    tablo5_duyarlilik_altgrup.csv      Tablo 5  – duyarlılık / alt grup analizleri
    tablo6_takip_kaybi_senaryo.csv     Tablo 6  – takip kaybı senaryo analizi
    kalibrasyon_gruplari.csv           Şekil 4 verisi
    karar_egrisi.csv                   Şekil 5 verisi
    ozet_sonuclar.json                 Metinde geçen tüm özet istatistikler
    Sekil1_AkisSemasi.png ... Sekil5_KararEgrisi.png
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import mcnemar

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# --------------------------------------------------------------------------
# Sabitler / Constants
# --------------------------------------------------------------------------
SEED = 2026                      # bootstrap tohum değeri
N_BOOT = 2000                    # bootstrap tekrar sayısı
THRESHOLD = 7                    # orijinal yüksek risk eşiği
# Orijinal POT-CAST kalibrasyon denklemi (Nemeth ve ark. 2020, Ek Tablo A2)
INTERCEPT, SLOPE = -6.677015, 0.3332203

COLUMNS = ["id", "sex", "age", "trauma", "cast", "height_m", "bmi", "height_cm", "weight_kg",
           "bmicat", "fam_vte", "own_vte", "cancer", "hosp3m", "varicose", "travel3m",
           "chf", "ra", "ckd", "copd", "ocp", "vte90", "anticoag", "score", "high_risk"]

# Akış şeması sayıları (veri setinde bulunmayan, kayıtlardan gelen sayılar)
FLOW = dict(ed_adults=89116, trauma=17832, immobilised=853,
            excl_admitted=31, excl_anticoag=72, excl_lost=85, period="1 Nisan–31 Mayıs 2026")


# --------------------------------------------------------------------------
# Yardımcı fonksiyonlar / Helpers
# --------------------------------------------------------------------------
def tr(x, nd=1):
    """Sayıyı Türkçe ondalık biçimde yazar (virgül)."""
    return f"{x:.{nd}f}".replace(".", ",")


def wilson(k, n):
    lo, hi = proportion_confint(k, n, method="wilson")
    return k / n, lo, hi


def fmt_prop(k, n, nd=1):
    p, lo, hi = wilson(k, n)
    return f"{tr(100*p, nd)} ({tr(100*lo, 1)}–{tr(100*hi, 1)})"


def fmt_p(p):
    return "<0,001" if p < 0.001 else tr(p, 3)


def delong(y, s):
    """DeLong yöntemiyle AUC ve standart hata; ayrıca yapısal bileşenler."""
    pos, neg = s[y == 1], s[y == 0]
    psi = lambda a, b: (a > b) + 0.5 * (a == b)
    v10 = np.array([psi(p, neg).mean() for p in pos])
    v01 = np.array([psi(pos, q).mean() for q in neg])
    auc = v10.mean()
    se = np.sqrt(v10.var(ddof=1) / len(pos) + v01.var(ddof=1) / len(neg))
    return auc, se, v10, v01


def delong_paired(y, a, b):
    """İki bağımlı ROC eğrisinin DeLong karşılaştırması."""
    _, _, a10, a01 = delong(y, a)
    _, _, b10, b01 = delong(y, b)
    s10, s01 = np.cov(np.vstack([a10, b10])), np.cov(np.vstack([a01, b01]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / len(a10) + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / len(a01)
    diff = a10.mean() - b10.mean()
    return a10.mean(), b10.mean(), diff, 2 * (1 - stats.norm.cdf(abs(diff) / np.sqrt(var)))


def bootstrap_auc(y, s, seed=SEED, n=N_BOOT):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        out.append(roc_auc_score(y[i], s[i]))
    return np.percentile(out, [2.5, 97.5])


def diag(y, pred):
    """2x2 tablo ve tanısal ölçütler."""
    TP = int((pred & (y == 1)).sum()); FN = int((~pred & (y == 1)).sum())
    FP = int((pred & (y == 0)).sum()); TN = int((~pred & (y == 0)).sum())
    se, sp = TP / (TP + FN), TN / (TN + FP)
    lrp = se / (1 - sp) if sp < 1 else float("inf")
    lrn = (1 - se) / sp
    return dict(TP=TP, FN=FN, FP=FP, TN=TN, sens=se, spec=sp, lrp=lrp, lrn=lrn, youden=se + sp - 1)


def net_benefit(y, pred, pt):
    n = len(y)
    return (pred & (y == 1)).sum() / n - (pred & (y == 0)).sum() / n * pt / (1 - pt)


# --------------------------------------------------------------------------
# Veri / Data
# --------------------------------------------------------------------------
def load(path):
    d = pd.read_excel(path)
    if d.shape[1] != len(COLUMNS):
        raise ValueError(f"Beklenen {len(COLUMNS)} sütun, bulunan {d.shape[1]}. README'deki sütun sırasını kontrol edin.")
    d.columns = COLUMNS
    # Puanlamanın veri setindeki skorla uyumunu doğrula
    age_pts = np.select([d.age >= 75, d.age >= 55, d.age >= 35], [3, 2, 1], 0)
    calc = (d.trauma.clip(lower=1) + d.cast + age_pts + (d.sex == 1) + d.bmicat.clip(upper=2)
            + 2 * d.fam_vte.fillna(0) + 4 * d.own_vte + 4 * d.ocp + 3 * d.cancer
            + 2 * ((d.hosp3m + d.travel3m) > 0) + ((d.chf + d.ra + d.ckd + d.copd) > 0) + d.varicose)
    mismatch = int((calc != d.score).sum())
    return d, mismatch


# --------------------------------------------------------------------------
# Analizler / Analyses
# --------------------------------------------------------------------------
def table2(d):
    y = d.vte90.values
    n1, n0 = int(y.sum()), int((y == 0).sum())
    rows = []

    def cont(label, v):
        f = lambda x: f"{x.median():.0f} ({x.quantile(.25):.0f}–{x.quantile(.75):.0f})"
        p = stats.mannwhitneyu(d.loc[y == 1, v].dropna(), d.loc[y == 0, v].dropna()).pvalue
        rows.append([label, f(d[v]), f(d.loc[y == 1, v]), f(d.loc[y == 0, v]), fmt_p(p)])

    def cat(label, m):
        a, b, c = int(m.sum()), int(m[y == 1].sum()), int(m[y == 0].sum())
        p = stats.fisher_exact([[b, n1 - b], [c, n0 - c]])[1]
        rows.append([label, f"{a} ({tr(100*a/len(d))})", f"{b} ({tr(100*b/n1)})", f"{c} ({tr(100*c/n0)})", fmt_p(p)])

    cont("Yaş, yıl, ortanca (ÇAA)", "age")
    for lab, m in [("Yaş <35", d.age < 35), ("Yaş 35–54", d.age.between(35, 54)), ("Yaş 55–74", d.age.between(55, 74)),
                   ("Yaş ≥75", d.age >= 75), ("Erkek cinsiyet", d.sex == 1),
                   ("Travma: düşük risk", d.trauma <= 1), ("Travma: orta risk", d.trauma == 2), ("Travma: yüksek risk", d.trauma == 3),
                   ("İmmobilizasyon: ayak/parmak", d.cast <= 1), ("İmmobilizasyon: diz altı", d.cast == 2), ("İmmobilizasyon: uzun bacak", d.cast == 3),
                   ("BKİ 25–35 kg/m²", d.bmicat == 1), ("BKİ ≥35 kg/m²", d.bmicat >= 2),
                   ("Ailede VTE öyküsü", d.fam_vte == 1), ("Kişisel VTE öyküsü", d.own_vte == 1), ("Kanser (son 5 yıl)", d.cancer == 1),
                   ("Oral kontraseptif/hormon", d.ocp == 1), ("Son 3 ayda hastane yatışı", d.hosp3m == 1),
                   ("Son 3 ayda >6 saat yolculuk", d.travel3m == 1), ("Variköz ven / KVY", d.varicose == 1),
                   ("Kalp yetmezliği", d.chf == 1), ("Romatoid artrit", d.ra == 1), ("Kronik böbrek hastalığı", d.ckd == 1),
                   ("KOAH", d.copd == 1), ("Profilaktik antikoagülan reçetesi", d.anticoag == 1)]:
        cat(lab, m)
    cont("TRiP(cast) puanı, ortanca (ÇAA)", "score")
    cat("Yüksek risk (TRiP(cast) ≥7)", d.score >= THRESHOLD)
    return pd.DataFrame(rows, columns=["Değişken", f"Tüm hastalar (n={len(d)})", f"VTE var (n={n1})", f"VTE yok (n={n0})", "p"])


def discrimination(d):
    y, s, age = d.vte90.values, d.score.values.astype(float), d.age.values.astype(float)
    auc = roc_auc_score(y, s)
    lo, hi = bootstrap_auc(y, s)
    age_pts = np.select([age >= 75, age >= 55, age >= 35], [3, 2, 1], 0)
    s_noage = s - age_pts
    _, auc_age, diff_age, p_age = delong_paired(y, s, age)
    _, auc_noage, _, p_noage = delong_paired(y, s, s_noage.astype(float))
    m1 = sm.Logit(y, sm.add_constant(s)).fit(disp=0)
    m2 = sm.Logit(y, sm.add_constant(np.column_stack([s_noage, age]))).fit(disp=0)
    return dict(auc=auc, auc_boot_ci=[lo, hi], auc_age=auc_age, auc_diff_vs_age=diff_age, p_vs_age=p_age,
                auc_without_age_points=auc_noage, p_vs_without_age=p_noage,
                or_per_point=[float(np.exp(m1.params[1])), *np.exp(m1.conf_int()[1]).tolist()],
                or_nonage_points_age_adjusted=[float(np.exp(m2.params[1])), *np.exp(m2.conf_int()[1]).tolist(), float(m2.pvalues[1])])


def table3(d):
    y, s, age = d.vte90.values, d.score.values, d.age.values
    rules = [(f"≥{c}" + (" (orijinal)" if c == THRESHOLD else ""), s >= c) for c in range(5, 12)]
    rules += [("Yaş ≥55†", age >= 55), ("Yaş ≥65†", age >= 65)]
    rows = []
    for lab, p in rules:
        r = diag(y, p)
        rows.append([lab, f"{int(p.sum())} ({tr(100*p.mean())})", fmt_prop(r["TP"], r["TP"] + r["FN"]),
                     fmt_prop(r["TN"], r["TN"] + r["FP"]), fmt_prop(r["TP"], r["TP"] + r["FP"]),
                     fmt_prop(r["TN"], r["TN"] + r["FN"]), tr(r["lrp"], 2), tr(r["lrn"], 2), tr(r["youden"], 2)])
    df = pd.DataFrame(rows, columns=["Kesim değeri", "Yüksek risk, n (%)", "Duyarlılık, % (%95 GA)", "Özgüllük, % (%95 GA)",
                                     "PPD, % (%95 GA)", "NPD, % (%95 GA)", "LR+", "LR−", "Youden"])
    # ≥7 için ek ölçütler
    r7 = diag(y, s >= THRESHOLD)
    se_lrp = math.sqrt(1/r7["TP"] - 1/(r7["TP"]+r7["FN"]) + 1/r7["FP"] - 1/(r7["FP"]+r7["TN"]))
    se_lrn = math.sqrt(1/r7["FN"] - 1/(r7["TP"]+r7["FN"]) + 1/r7["TN"] - 1/(r7["FP"]+r7["TN"])) if r7["FN"] else float("nan")
    dor = r7["TP"] * r7["TN"] / (r7["FP"] * r7["FN"]) if r7["FN"] else float("inf")
    se_dor = math.sqrt(sum(1/v for v in [r7["TP"], r7["FN"], r7["FP"], r7["TN"]])) if r7["FN"] else float("nan")
    extra = dict(lr_pos=[r7["lrp"], r7["lrp"]*math.exp(-1.96*se_lrp), r7["lrp"]*math.exp(1.96*se_lrp)],
                 lr_neg=[r7["lrn"], r7["lrn"]*math.exp(-1.96*se_lrn), r7["lrn"]*math.exp(1.96*se_lrn)],
                 dor=[dor, dor*math.exp(-1.96*se_dor), dor*math.exp(1.96*se_dor)],
                 accuracy=wilson(r7["TP"] + r7["TN"], len(y)),
                 failure_rate_low_risk=wilson(r7["FN"], r7["TN"] + r7["FN"]))
    # Yaş kuralı ile eşleştirilmiş karşılaştırma (McNemar kesin testi)
    a, b = s >= THRESHOLD, age >= 55
    ev, ne = y == 1, y == 0
    t_sens = [[int((a & b & ev).sum()), int((a & ~b & ev).sum())], [int((~a & b & ev).sum()), int((~a & ~b & ev).sum())]]
    t_spec = [[int((a & b & ne).sum()), int((a & ~b & ne).sum())], [int((~a & b & ne).sum()), int((~a & ~b & ne).sum())]]
    extra["mcnemar_vs_age55"] = dict(sens_table=t_sens, sens_p=mcnemar(t_sens, exact=True).pvalue,
                                     spec_table=t_spec, spec_p=mcnemar(t_spec, exact=True).pvalue)
    return df, extra


def strata_and_calibration(d):
    y, s = d.vte90.values, d.score.values
    strata = [("2–4", s <= 4), ("5–6", (s >= 5) & (s <= 6)), ("7–8", (s >= 7) & (s <= 8)), ("9–10", (s >= 9) & (s <= 10)), ("≥11", s >= 11)]
    st = [(lab, int(m.sum()), int(y[m].sum())) for lab, m in strata]
    # Cochran–Armitage eğilim testi
    sc = np.arange(len(st)); N = np.array([x[1] for x in st]); X = np.array([x[2] for x in st]); p = X.sum() / N.sum()
    T = np.sum(sc * (X - N * p)); V = p * (1 - p) * (np.sum(N * sc**2) - np.sum(N * sc)**2 / N.sum())
    trend_p = 2 * (1 - stats.norm.cdf(abs(T / np.sqrt(V))))
    # Orijinal denkleme göre kalibrasyon
    lp = INTERCEPT + SLOPE * s
    pr = 1 / (1 + np.exp(-lp))
    O, E = int(y.sum()), float(pr.sum())
    oe_lo = stats.chi2.ppf(.025, 2 * O) / 2 / E; oe_hi = stats.chi2.ppf(.975, 2 * (O + 1)) / 2 / E
    citl = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit()
    slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    z = np.sum((y - pr) * (1 - 2 * pr)) / np.sqrt(np.sum((1 - 2 * pr)**2 * pr * (1 - pr)))
    grp = pd.cut(d.score, [0, 4, 5, 6, 7, 8, 10, 20], labels=["2–4", "5", "6", "7", "8", "9–10", "≥11"])
    cg = pd.DataFrame(dict(g=grp, y=y, p=pr)).groupby("g", observed=True).agg(n=("y", "size"), o=("y", "sum"), e=("p", "sum"), pm=("p", "mean"))
    cg["obs"] = cg.o / cg.n * 100; lo, hi = proportion_confint(cg.o, cg.n, method="wilson")
    cg["lo"], cg["hi"], cg["pm"] = lo * 100, hi * 100, cg.pm * 100
    calib = dict(observed=O, expected=E, mean_predicted_pct=100 * pr.mean(), oe=[O / E, oe_lo, oe_hi],
                 citl=[float(citl.params[0]), *citl.conf_int()[0].tolist()],
                 slope=[float(slope.params[1]), *slope.conf_int()[1].tolist(),
                        float(2 * (1 - stats.norm.cdf(abs((slope.params[1] - 1) / slope.bse[1]))))],
                 spiegelhalter=[float(z), float(2 * (1 - stats.norm.cdf(abs(z))))],
                 brier=float(np.mean((pr - y)**2)), brier_null=float(np.mean((y.mean() - y)**2)))
    return st, trend_p, calib, cg.reset_index(), slope


def table4(d):
    y = d.vte90.values; n1, n0 = int(y.sum()), int((y == 0).sum())
    rows = []
    for lab, m in [("Kişisel VTE öyküsü", d.own_vte == 1), ("Yaş ≥55", d.age >= 55),
                   ("Komorbidite (KKY/RA/KBH/KOAH)", (d.chf + d.ra + d.ckd + d.copd) > 0), ("Kanser (son 5 yıl)", d.cancer == 1),
                   ("Son 3 ayda yatış / >6 saat yolculuk", (d.hosp3m + d.travel3m) > 0), ("Ailede VTE öyküsü", d.fam_vte == 1),
                   ("Variköz ven / KVY", d.varicose == 1), ("Uzun bacak immobilizasyonu", d.cast == 3), ("BKİ ≥25 kg/m²", d.bmicat >= 1),
                   ("Orta/yüksek riskli travma", d.trauma >= 2), ("Erkek cinsiyet", d.sex == 1)]:
        a = int((m & (y == 1)).sum()); b = int((m & (y == 0)).sum()); c = n1 - a; dd = n0 - b
        orr = ((a + .5) * (dd + .5)) / ((b + .5) * (c + .5))      # Haldane–Anscombe düzeltmesi
        se = math.sqrt(1/(a + .5) + 1/(b + .5) + 1/(c + .5) + 1/(dd + .5))
        p = stats.fisher_exact([[a, c], [b, dd]])[1]
        rows.append([lab, a, b, f"{tr(orr,2)} ({tr(orr*math.exp(-1.96*se),2)}–{tr(orr*math.exp(1.96*se),2)})", fmt_p(p)])
    return pd.DataFrame(rows, columns=["Bileşen", f"VTE var, n/{n1}", f"VTE yok, n/{n0}", "OO (%95 GA)*", "p"])


def table5(d):
    rows = []
    for lab, m in [("Ana analiz", np.ones(len(d), bool)), ("Antikoagülan alanlar dışlandığında", (d.anticoag != 1).values),
                   ("Düşük riskli travma", (d.trauma <= 1).values),
                   ("Diz altı immobilizasyon", (d.cast == 2).values), ("Yaş <55", (d.age < 55).values), ("Yaş ≥55", (d.age >= 55).values)]:
        x = d[m]; y = x.vte90.values; s = x.score.values.astype(float)
        if y.sum() < 2 or (y == 0).sum() < 2:
            rows.append([lab, len(x), int(y.sum()), int((s >= THRESHOLD).sum()), "—", "—", "—", "—", "—"]); continue
        if lab == "Ana analiz":
            auc = roc_auc_score(y, s); lo, hi = bootstrap_auc(y, s); auc_txt = f"{tr(auc,3)} ({tr(lo,3)}–{tr(hi,3)})†"
        else:
            a, se, _, _ = delong(y, s); auc_txt = f"{tr(a,3)} ({tr(a-1.96*se,3)}–{tr(min(1, a+1.96*se),3)})"
        r = diag(y, s >= THRESHOLD)
        rows.append([lab, len(x), int(y.sum()), int((s >= THRESHOLD).sum()), auc_txt,
                     fmt_prop(r["TP"], r["TP"] + r["FN"]), fmt_prop(r["TN"], r["TN"] + r["FP"]),
                     fmt_prop(r["TP"], r["TP"] + r["FP"]), fmt_prop(r["TN"], r["TN"] + r["FN"])])
    return pd.DataFrame(rows, columns=["Analiz", "n", "VTE, n", "Yüksek risk (≥7), n", "AUC (%95 GA)",
                                       "Duyarlılık, % (%95 GA)", "Özgüllük, % (%95 GA)", "PPD, % (%95 GA)", "NPD, % (%95 GA)"])


def table6(d, n_lost=FLOW["excl_lost"]):
    """Takip kaybı için senaryo ve devrilme noktası analizi (kayıp hastaların puanları bilinmiyor)."""
    y, s = d.vte90.values, d.score.values
    low = s < THRESHOLD; n_low, ev_low = int(low.sum()), int((y[low] == 1).sum())
    share_low = n_low / len(d); rate_all = y.mean(); rate_high = y[~low].mean()
    lost_low = round(n_lost * share_low)
    sc = [("Ana analiz", "Yalnızca takibi tamamlanan hastalar", n_low, ev_low),
          ("S1", f"Kayıpların %{tr(100*share_low)}'ü (n={lost_low}) düşük riskli; bu hastalarda VTE yok", n_low + lost_low, ev_low),
          ("S2", f"Kayıpların %{tr(100*share_low)}'ü (n={lost_low}) düşük riskli; VTE sıklığı %{tr(100*rate_all)} (kohort ortalaması)",
           n_low + lost_low, ev_low + math.ceil(lost_low * rate_all)),
          ("S3", f"Kayıpların tamamı (n={n_lost}) düşük riskli; VTE sıklığı %{tr(100*rate_all)}", n_low + n_lost, ev_low + math.ceil(n_lost * rate_all)),
          ("S4", f"Kayıpların tamamı (n={n_lost}) düşük riskli; VTE sıklığı %{tr(100*rate_high)} (yüksek risk grubu düzeyi)",
           n_low + n_lost, ev_low + math.ceil(n_lost * rate_high))]
    k = 0
    while (ev_low + k) / (n_low + n_lost) <= 0.01:
        k += 1
    sc.append(("Devrilme noktası", f"Başarısızlık oranını %1'in üzerine çıkaran en az olay sayısı (kayıp hastalarda {k} VTE; %{tr(100*k/n_lost)})",
               n_low + n_lost, ev_low + k))
    rows = [[a, b, n, e, fmt_prop(e, n, 2), fmt_prop(n - e, n)] for a, b, n, e in sc]
    return pd.DataFrame(rows, columns=["Senaryo", "Varsayım", "Düşük riskli hasta, n", "VTE, n",
                                       "Başarısızlık oranı, % (%95 GA)", "NPD, % (%95 GA)"]), k


def decision_curve(d):
    y, s, age = d.vte90.values, d.score.values, d.age.values
    strategies = {"Herkese profilaksi": np.ones(len(y), bool), "TRiP(cast) ≥7": s >= 7, "TRiP(cast) ≥8": s >= 8, "Yaş ≥55": age >= 55}
    pts = np.round(np.arange(0.005, 0.1501, 0.005), 3)
    return pd.DataFrame({"esik_olasilik": pts, **{k: [net_benefit(y, v, t) for t in pts] for k, v in strategies.items()}})


# --------------------------------------------------------------------------
# Şekiller / Figures
# --------------------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.8})
ACC, GRAY, ORANGE, GREEN = "#1f4e79", "#8a8a8a", "#c05a1a", "#2e7d32"
comma = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def fig1_flow(d, out):
    n = len(d); y = d.vte90.values; low = d.score.values < THRESHOLD
    f = FLOW
    fig, ax = plt.subplots(figsize=(7.6, 9.6), dpi=300); ax.set_xlim(0, 108); ax.set_ylim(0, 125); ax.axis("off")

    def box(x, yy, w, h, txt, fc="white", align="center", fs=9.5):
        ax.add_patch(FancyBboxPatch((x - w/2, yy - h/2), w, h, boxstyle="square,pad=0", fc=fc, ec="#222", lw=1))
        if align == "center":
            ax.text(x, yy, txt, ha="center", va="center", fontsize=fs, linespacing=1.45)
        else:
            ax.text(x - w/2 + 2, yy, txt, ha="left", va="center", fontsize=fs, linespacing=1.45)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#222", lw=1, mutation_scale=10))
    th = lambda v: f"{v:,}".replace(",", ".")
    cx, W, sx, SW = 34, 50, 82, 34
    box(cx, 116, W, 9, f"Acil servise başvuran erişkin hastalar\n({f['period']})\nn = {th(f['ed_adults'])}")
    box(cx, 96, W, 7, f"Travma hastası olarak değerlendirilenler\nn = {th(f['trauma'])}")
    box(cx, 77, W, 7, f"Alt ekstremite immobilizasyonu uygulananlar\nn = {th(f['immobilised'])}")
    box(cx, 47, W, 7, f"Analize dahil edilen hastalar\nn = {th(n)}")
    box(sx, 106, SW, 6.5, f"Travma dışı başvurular\nn = {th(f['ed_adults']-f['trauma'])}", fc="#f2f2f2", fs=8.5)
    box(sx, 86.5, SW, 8.5, f"Alt ekstremite immobilizasyonu\nuygulanmayan hastalar\nn = {th(f['trauma']-f['immobilised'])}", fc="#f2f2f2", fs=8.5)
    excl = f["excl_admitted"] + f["excl_anticoag"] + f["excl_lost"]
    box(sx, 62, SW + 8, 15, f"Dışlananlar (n = {excl})\n• Hastaneye yatış: n = {f['excl_admitted']}\n• AF ve/veya mekanik kapak nedeniyle\n"
        f"   düzenli antikoagülan kullanımı: n = {f['excl_anticoag']}\n• 90 günlük takip verisine\n   ulaşılamayan: n = {f['excl_lost']}",
        fc="#f2f2f2", align="left", fs=8.5)
    arrow(cx, 111.5, cx, 99.5); arrow(cx, 92.5, cx, 80.5); arrow(cx, 73.5, cx, 50.5)
    for yy, w in [(106, SW), (86.5, SW), (62, SW + 8)]:
        ax.plot([cx, sx - w/2], [yy, yy], color="#222", lw=1); arrow(sx - w/2 - 0.01, yy, sx - w/2, yy)
    ax.plot([cx, cx], [43.5, 38], color="#222", lw=1); ax.plot([18, 62], [38, 38], color="#222", lw=1)
    arrow(18, 38, 18, 33.5); arrow(62, 38, 62, 33.5)
    box(18, 28, 30, 10, f"TRiP(cast) < 7\nDüşük risk\nn = {low.sum()} (%{tr(100*low.mean())})")
    box(62, 28, 30, 10, f"TRiP(cast) ≥ 7\nYüksek risk\nn = {(~low).sum()} (%{tr(100*(~low).mean())})")
    arrow(18, 23, 18, 17.5); arrow(62, 23, 62, 17.5)
    box(18, 12, 30, 10, f"90 günde semptomatik VTE\nn = {y[low].sum()} (%{tr(100*y[low].mean())})", fc="#f2f2f2")
    box(62, 12, 30, 10, f"90 günde semptomatik VTE\nn = {y[~low].sum()} (%{tr(100*y[~low].mean())})", fc="#f2f2f2")
    fig.tight_layout(); fig.savefig(os.path.join(out, "Sekil1_AkisSemasi.png"), bbox_inches="tight"); plt.close(fig)


def fig2_roc(d, out, disc):
    y, s, age = d.vte90.values, d.score.values, d.age.values
    fpr, tpr, thr = roc_curve(y, s); fa, ta, _ = roc_curve(y, age)
    fig, ax = plt.subplots(figsize=(4.6, 4.4), dpi=300)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=GRAY)
    lo, hi = disc["auc_boot_ci"]
    ax.step(fpr, tpr, where="post", lw=2, color=ACC, label=f"TRiP(cast)  AUC {tr(disc['auc'],2)} (%95 GA {tr(lo,2)}–{tr(hi,2)})")
    ax.step(fa, ta, where="post", lw=1.4, color=ORANGE, label=f"Yaş (sürekli)  AUC {tr(disc['auc_age'],2)}")
    i = np.where(thr == THRESHOLD)[0]
    if len(i):
        i = i[0]; ax.plot(fpr[i], tpr[i], "o", ms=7, color=ACC, mec="white", mew=1.5)
        ax.annotate(f"Eşik ≥{THRESHOLD}\nDuyarlılık %{tr(100*tpr[i])}\nÖzgüllük %{tr(100*(1-fpr[i]))}", (fpr[i], tpr[i]),
                    xytext=(0.36, 0.62), fontsize=8.5, arrowprops=dict(arrowstyle="-", color="#444", lw=0.8))
    ax.set_xlabel("1 − Özgüllük"); ax.set_ylabel("Duyarlılık"); ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.02)
    ax.xaxis.set_major_formatter(comma); ax.yaxis.set_major_formatter(comma)
    ax.legend(loc="lower right", frameon=False, fontsize=8); ax.grid(alpha=.25, lw=.5)
    fig.tight_layout(); fig.savefig(os.path.join(out, "Sekil2_ROC.png")); plt.close(fig)


def fig3_strata(st, out):
    lab = [x[0] for x in st]; n = np.array([x[1] for x in st]); e = np.array([x[2] for x in st])
    p = e / n * 100; lo, hi = proportion_confint(e, n, method="wilson"); lo, hi = lo * 100, hi * 100
    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=300)
    cols = [GRAY, GRAY, ACC, ACC, ACC]
    ax.errorbar(range(5), p, yerr=[p - lo, hi - p], fmt="none", ecolor="#333", elinewidth=1, capsize=4)
    for k in range(5):
        ax.plot(k, p[k], "o", ms=8, color=cols[k], mec="white", mew=1.2, zorder=3)
        t = f"{e[k]}/{n[k]}\n(%{tr(p[k])})"
        if k == 4:
            ax.text(k + 0.12, p[k] + 1.5, t, fontsize=8, va="bottom", ha="left")
        else:
            ax.text(k, hi[k] + 2.5, t, fontsize=8, va="bottom", ha="center")
    ax.axvline(1.5, ls=":", color="#555", lw=1); ax.text(1.55, 92, f"Yüksek risk eşiği (≥{THRESHOLD})", fontsize=8, color="#333")
    ax.set_xticks(range(5)); ax.set_xticklabels(lab); ax.set_xlabel("TRiP(cast) puanı"); ax.set_ylabel("90 günlük semptomatik VTE (%)")
    ax.set_ylim(-3, 100); ax.set_xlim(-0.5, 4.6); ax.grid(axis="y", alpha=.25, lw=.5)
    fig.tight_layout(); fig.savefig(os.path.join(out, "Sekil3_PuanaGoreVTE.png")); plt.close(fig)


def fig4_calibration(d, cg, calib, slope_model, out):
    s = d.score.values; lp = INTERCEPT + SLOPE * s
    xs = np.linspace(lp.min(), lp.max(), 300); pr = 100 / (1 + np.exp(-xs))
    ob = 100 / (1 + np.exp(-(slope_model.params[0] + slope_model.params[1] * xs)))
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 4.4), dpi=300)
    for k, ax in enumerate(axs):
        xl, yl = [(10, 100), (3.5, 32)][k]
        ax.plot([0, xl], [0, xl], color="#333", lw=1, label="İdeal kalibrasyon")
        ax.plot(pr, ob, ls="--", color=ORANGE, lw=1.4, label="Kalibrasyon eğrisi")
        for _, r in cg.iterrows():
            if r.pm > xl:
                continue
            ax.errorbar(r.pm, r.obs, yerr=[[r.obs - r.lo], [r.hi - r.obs]], fmt="none", ecolor="#555", elinewidth=0.9, capsize=3, zorder=2)
            ax.plot(r.pm, r.obs, "o", ms=6.5, color=ACC, mec="white", mew=1.1, zorder=3)
            if k == 1 or r.g == "≥11":
                yy, va = (r.hi + 0.6, "bottom") if k == 1 else (r.lo - 2, "top")
                ax.text(r.pm, yy, f"{r.g} puan ({int(r.o)}/{int(r.n)})", rotation=90, fontsize=7.5, va=va, ha="center",
                        bbox=dict(fc="white", ec="none", pad=0.4, alpha=.85), zorder=4)
        ax.set_xlim(0, xl); ax.set_ylim(-0.03 * yl, yl); ax.grid(alpha=.25, lw=.5)
        ax.xaxis.set_major_formatter(comma); ax.yaxis.set_major_formatter(comma)
        ax.set_xlabel("Tahmin edilen risk (%)"); ax.set_ylabel("Gözlenen risk (%)")
    axs[0].set_title("A. Tüm skor grupları", loc="left", fontsize=10)
    axs[1].set_title("B. Yakınlaştırılmış görünüm (2–10 puan)", loc="left", fontsize=10)
    axs[0].legend(frameon=False, fontsize=8, loc="upper left")
    oe, ci, sl = calib["oe"], calib["citl"], calib["slope"]
    axs[0].text(0.4, 52, f"O/B oranı {tr(oe[0],2)} ({tr(oe[1],2)}–{tr(oe[2],2)})\nKesişim {tr(ci[0],2)} ({tr(ci[1],2)}–{tr(ci[2],2)})\n"
                f"Eğim {tr(sl[0],2)} ({tr(sl[1],2)}–{tr(sl[2],2)})", fontsize=8, bbox=dict(fc="white", ec="#ccc", pad=4), zorder=5)
    fig.tight_layout(); fig.savefig(os.path.join(out, "Sekil4_Kalibrasyon.png")); plt.close(fig)


def fig5_dca(d, out):
    y, s, age = d.vte90.values, d.score.values, d.age.values
    pts = np.linspace(0.005, 0.15, 120)
    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=300)
    ax.plot(pts * 100, [net_benefit(y, np.ones(len(y), bool), t) for t in pts], color=GRAY, lw=1.4, label="Herkese profilaksi")
    ax.axhline(0, color="#333", lw=1, label="Hiç kimseye profilaksi")
    ax.plot(pts * 100, [net_benefit(y, s >= 7, t) for t in pts], color=ACC, lw=2, label="TRiP(cast) ≥7")
    ax.plot(pts * 100, [net_benefit(y, s >= 8, t) for t in pts], color=ORANGE, lw=1.4, ls="--", label="TRiP(cast) ≥8")
    ax.plot(pts * 100, [net_benefit(y, age >= 55, t) for t in pts], color=GREEN, lw=1.4, ls="-.", label="Yaş ≥55")
    ax.set_ylim(-0.02, 0.025); ax.set_xlabel("Eşik olasılık (%)"); ax.set_ylabel("Net fayda")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.3f}".replace(".", ",")))
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.25, lw=.5)
    fig.tight_layout(); fig.savefig(os.path.join(out, "Sekil5_KararEgrisi.png")); plt.close(fig)


# --------------------------------------------------------------------------
# Ana akış / Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="TRiP(cast) dış validasyon analizi")
    ap.add_argument("--data", required=True, help="Excel veri dosyası (.xlsx)")
    ap.add_argument("--out", default="sonuclar", help="Çıktı klasörü")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d, mismatch = load(a.data)
    y = d.vte90.values
    print(f"Hasta sayısı: {len(d)} | VTE: {int(y.sum())} | Skor uyumsuzluğu: {mismatch}")

    t2 = table2(d); t2.to_csv(os.path.join(a.out, "tablo2_temel_ozellikler.csv"), index=False, encoding="utf-8-sig")
    disc = discrimination(d)
    t3, extra = table3(d); t3.to_csv(os.path.join(a.out, "tablo3_kesim_degerleri.csv"), index=False, encoding="utf-8-sig")
    st, trend_p, calib, cg, slope_model = strata_and_calibration(d)
    cg.to_csv(os.path.join(a.out, "kalibrasyon_gruplari.csv"), index=False, encoding="utf-8-sig")
    t4 = table4(d); t4.to_csv(os.path.join(a.out, "tablo4_bilesenler.csv"), index=False, encoding="utf-8-sig")
    t5 = table5(d); t5.to_csv(os.path.join(a.out, "tablo5_duyarlilik_altgrup.csv"), index=False, encoding="utf-8-sig")
    t6, tip = table6(d); t6.to_csv(os.path.join(a.out, "tablo6_takip_kaybi_senaryo.csv"), index=False, encoding="utf-8-sig")
    dca = decision_curve(d); dca.to_csv(os.path.join(a.out, "karar_egrisi.csv"), index=False, encoding="utf-8-sig")

    summary = dict(n=len(d), events=int(y.sum()), incidence=wilson(int(y.sum()), len(d)), score_mismatch=mismatch,
                   age_mean_sd=[float(d.age.mean()), float(d.age.std())], female=int((d.sex == 2).sum()),
                   discrimination=disc, threshold7=extra, strata=st, trend_p=trend_p, calibration=calib,
                   tipping_point_events=tip,
                   anticoag=dict(total=int((d.anticoag == 1).sum()), low_risk=int(((d.anticoag == 1) & (d.score < 7)).sum()),
                                 high_risk=int(((d.anticoag == 1) & (d.score >= 7)).sum())),
                   net_benefit_per100_at_2pct={k: 100 * v for k, v in dca.loc[dca.esik_olasilik == 0.02].iloc[0].items() if k != "esik_olasilik"})
    with open(os.path.join(a.out, "ozet_sonuclar.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=float)

    fig1_flow(d, a.out); fig2_roc(d, a.out, disc); fig3_strata(st, a.out)
    fig4_calibration(d, cg, calib, slope_model, a.out); fig5_dca(d, a.out)

    print(f"AUC {disc['auc']:.3f} (bootstrap %95 GA {disc['auc_boot_ci'][0]:.3f}–{disc['auc_boot_ci'][1]:.3f})")
    print(f"O/B {calib['oe'][0]:.2f} | kalibrasyon eğimi {calib['slope'][0]:.2f} | devrilme noktası {tip} olay")
    print(f"Çıktılar: {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
