# -*- coding: utf-8 -*-
"""
====================================================================
 02. パルス整形とアイパターン (Pulse Shaping & Eye Diagram)
====================================================================

このスクリプトでは、ディジタル通信で帯域制限された通信路を通して
データを送る際に不可欠な「パルス整形」の概念を学びます。

■ 学習のゴール
  1. 矩形パルスの問題点（無限の帯域幅、ISI）を理解する
  2. ナイキストの第1条件（ISIフリー条件）を理解する
  3. Raised Cosine (RC) フィルタの仕組みとロールオフ率 α の効果を理解する
  4. Root Raised Cosine (RRC) フィルタと送受信でのマッチドフィルタを理解する
  5. アイパターンを用いたISI評価方法を身につける

■ 使い方
  各セクションを順に読み進め、グラフの変化を確認してください。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# --- 日本語フォント設定 ---
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'Hiragino Sans']
rcParams['axes.unicode_minus'] = False

print("=" * 60)
print(" セクション 1: なぜパルス整形が必要か？")
print("=" * 60)
print("""
ディジタル通信では、ビット列 (0, 1, 1, 0, ...) をアナログ波形に変換して
通信路に送り出します。最も単純な方法は矩形パルス（NRZ）を使うことです。

しかし、矩形パルスには大きな問題があります:
  1. 周波数スペクトルが sinc 関数になり、帯域幅が無限大
  2. 帯域制限されたチャネルを通ると波形が崩れ、
     隣のシンボルと干渉する（符号間干渉: ISI）

ISI が発生すると、受信側でビットの判定を誤る確率が上がります。
""")

# ── 矩形パルスとそのスペクトル ──
T_sym = 1.0  # シンボル周期
samples_per_sym = 64
t_pulse = np.linspace(-2 * T_sym, 2 * T_sym, samples_per_sym * 4 + 1)

# 矩形パルス
rect_pulse = np.where(np.abs(t_pulse) <= T_sym / 2, 1.0, 0.0)

# 周波数スペクトル
f_axis = np.linspace(-5, 5, 1000)
rect_spectrum = T_sym * np.sinc(f_axis * T_sym)  # sinc = sin(πx)/(πx)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

ax1.plot(t_pulse / T_sym, rect_pulse, 'b-', linewidth=2)
ax1.set_title('矩形パルス（時間領域）')
ax1.set_xlabel('t / T (正規化時間)')
ax1.set_ylabel('振幅')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.2, 1.3)

ax2.plot(f_axis, np.abs(rect_spectrum), 'r-', linewidth=2)
ax2.fill_between(f_axis, 0, np.abs(rect_spectrum), alpha=0.1, color='red')
ax2.set_title('矩形パルスの周波数スペクトル |sinc(fT)|')
ax2.set_xlabel('周波数 f·T (正規化周波数)')
ax2.set_ylabel('振幅')
ax2.grid(True, alpha=0.3)

plt.suptitle('矩形パルスの問題: 帯域が無限に広がる', fontsize=13)
plt.tight_layout()
plt.show()

print("""
【観察ポイント】
スペクトルが sinc 関数になり、高い周波数にまでエネルギーが広がっています。
実際の通信路には帯域制限があるため、このパルスをそのまま使うことはできません。
→ パルス整形フィルタが必要！
""")

print("=" * 60)
print(" セクション 2: ナイキストの第1条件と Raised Cosine フィルタ")
print("=" * 60)
print("""
■ ナイキストの第1条件（ISI フリー条件）
  パルス p(t) が以下を満たすとき、サンプリング時点で ISI が発生しません:

    p(nT) = { 1  (n = 0)
            { 0  (n ≠ 0)

  つまり、パルスの「裾」が他のシンボルのサンプリング時点でちょうど 0 になれば、
  隣のシンボルとの干渉がなくなります。

■ Raised Cosine (RC) フィルタ
  ナイキスト条件を満たす代表的なフィルタです。
  ロールオフ率 α (0 ≤ α ≤ 1) で帯域幅と時間的な減衰の速さを制御します。

  時間領域:
    p(t) = sinc(t/T) · cos(πα·t/T) / (1 - (2αt/T)²)

  周波数領域:
    α = 0: 理想的な矩形スペクトル（sinc パルス、減衰が遅い）
    α = 1: 緩やかに立ち下がる（減衰が速い、帯域幅は2倍）

  α を大きくすると:
    ✓ 時間的な減衰が速くなり、タイミング誤差に強くなる
    ✗ 必要な帯域幅が広くなる
""")


def raised_cosine(t, T, alpha):
    """Raised Cosine パルスの時間波形"""
    result = np.zeros_like(t, dtype=float)
    for i, ti in enumerate(t):
        if np.abs(ti) < 1e-12:
            result[i] = 1.0
        elif alpha > 0 and np.abs(np.abs(2 * alpha * ti / T) - 1.0) < 1e-12:
            result[i] = (np.pi / (4 * T)) * np.sinc(1.0 / (2 * alpha))
        else:
            sinc_val = np.sinc(ti / T)
            cos_val = np.cos(np.pi * alpha * ti / T)
            denom = 1 - (2 * alpha * ti / T) ** 2
            result[i] = sinc_val * cos_val / denom
    return result


def rc_spectrum(f, T, alpha):
    """Raised Cosine の周波数特性"""
    H = np.zeros_like(f, dtype=float)
    for i, fi in enumerate(f):
        fi_abs = np.abs(fi)
        if fi_abs <= (1 - alpha) / (2 * T):
            H[i] = T
        elif fi_abs <= (1 + alpha) / (2 * T):
            H[i] = T / 2 * (1 + np.cos(np.pi * T / alpha * (fi_abs - (1 - alpha) / (2 * T))))
        else:
            H[i] = 0.0
    return H


# ── ロールオフ率 α の比較 ──
t = np.linspace(-6, 6, 2000)
f = np.linspace(-1.5, 1.5, 2000)
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for alpha, color in zip(alphas, colors):
    p = raised_cosine(t, 1.0, alpha)
    ax1.plot(t, p, color=color, linewidth=1.5, label=f'α = {alpha}')

    H = rc_spectrum(f, 1.0, alpha)
    ax2.plot(f, H, color=color, linewidth=1.5, label=f'α = {alpha}')

# ナイキスト条件の視覚化: t = nT で p(nT) = 0 (n≠0)
ax1.scatter([0], [1], color='black', zorder=5, s=60)
for n in range(-5, 6):
    if n != 0:
        ax1.scatter([n], [0], color='black', zorder=5, s=30, marker='x')

ax1.set_title('Raised Cosine パルス（時間領域）')
ax1.set_xlabel('t / T')
ax1.set_ylabel('p(t)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.3, 1.1)
ax1.annotate('ナイキスト条件:\np(nT)=0 (n≠0)', xy=(1, 0), xytext=(2.5, 0.5),
             fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))

ax2.set_title('Raised Cosine スペクトル（周波数領域）')
ax2.set_xlabel('f·T')
ax2.set_ylabel('|H(f)|')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('ロールオフ率 α による Raised Cosine フィルタの変化', fontsize=13)
plt.tight_layout()
plt.show()

print("""
【観察ポイント】
- 時間領域: α を大きくするほど、パルスの「裾」が速く減衰します。
  → タイミングの誤差（ジッタ）に対して頑健になります。
  → しかし t = nT (n≠0) では常に 0 になっています（ナイキスト条件）。

- 周波数領域: α = 0 では帯域 W = 1/(2T) の矩形、
  α = 1 では帯域 W = 1/T の余弦状に広がります。
  → α が大きいほど帯域効率は悪くなりますが、実装しやすくなります。
""")

print("=" * 60)
print(" セクション 3: Root Raised Cosine (RRC) フィルタ")
print("=" * 60)
print("""
■ マッチドフィルタの概念
  実際の通信システムでは、送信側と受信側の両方にフィルタを配置します。
  送受信フィルタの畳み込みが Raised Cosine になるように設計すると、
  SNR を最大化できます（マッチドフィルタ定理）。

  具体的には:
    送信フィルタ: √RC (Root Raised Cosine)
    受信フィルタ: √RC (Root Raised Cosine)

    √RC * √RC = RC  （畳み込みの結果が Raised Cosine）

  これにより:
    - 送信側: 帯域制限されたパルスを送出
    - 受信側: SNR を最大化しつつ ISI フリーを実現
""")


def root_raised_cosine(t, T, alpha, num_taps=None):
    """Root Raised Cosine パルスの時間波形"""
    result = np.zeros_like(t, dtype=float)
    for i, ti in enumerate(t):
        if np.abs(ti) < 1e-12:
            result[i] = (1 / T) * (1 + alpha * (4 / np.pi - 1))
        elif alpha > 0 and np.abs(np.abs(ti) - T / (4 * alpha)) < 1e-12:
            val = (alpha / (T * np.sqrt(2))) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
            )
            result[i] = val
        else:
            num = np.sin(np.pi * ti / T * (1 - alpha)) + \
                  4 * alpha * ti / T * np.cos(np.pi * ti / T * (1 + alpha))
            den = np.pi * ti / T * (1 - (4 * alpha * ti / T) ** 2)
            result[i] = num / (den * T)
    # Normalize
    result = result / np.max(np.abs(result))
    return result


# ── RRC と RC の比較 ──
t = np.linspace(-6, 6, 2000)
alpha = 0.35  # 一般的なロールオフ率

rc_pulse = raised_cosine(t, 1.0, alpha)
rrc_pulse = root_raised_cosine(t, 1.0, alpha)

# RRC の自己相関（畳み込み）→ RC になることを確認
rrc_conv = np.convolve(rrc_pulse, rrc_pulse, mode='same')
rrc_conv = rrc_conv / np.max(rrc_conv)  # 正規化

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(t, rrc_pulse, 'b-', linewidth=2, label='RRC (送信フィルタ)')
ax1.plot(t, rc_pulse, 'r--', linewidth=2, label='RC (理想的な合成応答)')
ax1.set_title(f'RRC vs RC パルス (α = {alpha})')
ax1.set_xlabel('t / T')
ax1.set_ylabel('振幅')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-6, 6)

ax2.plot(t, rc_pulse, 'r--', linewidth=2, label='Raised Cosine (理論)')
ax2.plot(t, rrc_conv, 'g-', linewidth=2, label='RRC ⊛ RRC (畳み込み)')
ax2.set_title('マッチドフィルタ: RRC ⊛ RRC = RC')
ax2.set_xlabel('t / T')
ax2.set_ylabel('振幅')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-6, 6)

plt.suptitle('Root Raised Cosine フィルタとマッチドフィルタ', fontsize=13)
plt.tight_layout()
plt.show()

print("""
【観察ポイント】
- 左図: RRC は RC と形が異なり、ゼロ交差点が t = nT とは限りません。
  単独では ISI フリーではありません。

- 右図: RRC を2回（送信+受信）畳み込むと、RC と一致します！
  つまり、システム全体としてナイキスト条件を満たします。
""")

print("=" * 60)
print(" セクション 4: ベースバンド伝送のシミュレーション")
print("=" * 60)
print("""
実際にランダムなビット列を送信し、パルス整形の効果を確認しましょう。

送信過程:
  1. ランダムなシンボル列を生成 (±1)
  2. 各シンボルにパルス波形を乗せて送信信号を合成
  3. 整形されたパルスの重ね合わせを観察
""")

np.random.seed(42)
num_symbols = 20
symbols = 2 * np.random.randint(0, 2, num_symbols) - 1  # ±1 のランダムシンボル

sps = 32  # samples per symbol
t_sym = np.arange(num_symbols * sps) / sps  # 時間軸

# パルス整形関数
def shape_signal(symbols, pulse_func, T, alpha, sps, span=8):
    """シンボル列にパルス整形を適用"""
    t_pulse = np.arange(-span * sps, span * sps + 1) / sps
    pulse = pulse_func(t_pulse, T, alpha)
    pulse = pulse / np.max(np.abs(pulse))

    # アップサンプリング
    upsampled = np.zeros(len(symbols) * sps)
    upsampled[::sps] = symbols

    # パルス整形（畳み込み）
    shaped = np.convolve(upsampled, pulse, mode='same')
    return shaped


fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# (a) 矩形パルス (NRZ)
nrz_signal = np.repeat(symbols, sps)
axes[0].plot(t_sym, nrz_signal, 'b-', linewidth=1.5)
axes[0].set_title('(a) 矩形パルス (NRZ) — 帯域無限大、実用的でない')
axes[0].set_ylabel('振幅')
axes[0].grid(True, alpha=0.3)
for n in range(num_symbols):
    axes[0].axvline(x=n, color='gray', linestyle=':', alpha=0.3)

# (b) Raised Cosine (α = 0.35)
rc_signal = shape_signal(symbols, raised_cosine, 1.0, 0.35, sps)
axes[1].plot(t_sym, rc_signal, 'r-', linewidth=1.5)
axes[1].set_title('(b) Raised Cosine (α = 0.35) — 帯域制限 + ISI フリー')
axes[1].set_ylabel('振幅')
axes[1].grid(True, alpha=0.3)
for n in range(num_symbols):
    axes[1].axvline(x=n, color='gray', linestyle=':', alpha=0.3)
# サンプリング時点をマーク
sample_times = np.arange(num_symbols)
sample_values = rc_signal[::sps][:num_symbols]
axes[1].scatter(sample_times, sample_values, color='red', zorder=5, s=40, label='判定点')
axes[1].legend()

# (c) Raised Cosine (α = 0.35) + 各シンボルの寄与を個別表示
axes[2].set_title('(c) 各シンボルのパルスの重ね合わせ（α = 0.35）')
t_single = np.arange(-4 * sps, (num_symbols + 4) * sps) / sps
for n in range(num_symbols):
    t_shifted = t_single - n
    p = symbols[n] * raised_cosine(t_shifted, 1.0, 0.35)
    mask = (t_single >= 0) & (t_single <= num_symbols)
    axes[2].plot(t_single[mask], p[mask], alpha=0.4, linewidth=0.8)
axes[2].plot(t_sym, rc_signal, 'k-', linewidth=2, label='合成信号')
axes[2].set_xlabel('時間 t / T')
axes[2].set_ylabel('振幅')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle('ベースバンド伝送信号の比較', fontsize=13)
plt.tight_layout()
plt.show()

print("""
【観察ポイント】
- (a) NRZ: 角ばった波形で、帯域幅が無限大です。
- (b) RC: 滑らかな波形ですが、判定点 (赤い点) では正確に ±1 の値を取ります。
  → ISI フリー！
- (c) 各シンボルのパルスが重なり合って合成信号を作っている様子が分かります。
  ナイキスト条件により、他のシンボルの判定点では各パルスが 0 になっています。
""")

print("=" * 60)
print(" セクション 5: アイパターン（Eye Diagram）")
print("=" * 60)
print("""
■ アイパターンとは？
  アイパターンは、受信波形を1シンボル周期ごとに重ね書きしたものです。
  信号品質を視覚的に評価する最も重要なツールの一つです。

  評価のポイント:
  - 「目」が大きく開いている → ISI が少ない → 良い判定マージン
  - 「目」が閉じている → ISI が大きい → ビット誤り率が高くなる
  - 目の中央の垂直方向の開き → ノイズマージン
  - 水平方向の開き → タイミングマージン
""")

np.random.seed(0)
num_symbols_eye = 500
symbols_eye = 2 * np.random.randint(0, 2, num_symbols_eye) - 1

alphas_eye = [0.0, 0.25, 0.5, 1.0]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, alpha in enumerate(alphas_eye):
    signal = shape_signal(symbols_eye, raised_cosine, 1.0, alpha, sps)

    # アイパターン: 2シンボル周期分ずつ重ね書き
    for n in range(50, num_symbols_eye - 2):
        start = n * sps
        end = start + 2 * sps
        if end <= len(signal):
            t_eye = np.linspace(0, 2, 2 * sps)
            axes[idx].plot(t_eye, signal[start:end], 'b-', alpha=0.15, linewidth=0.5)

    axes[idx].set_title(f'アイパターン (α = {alpha})')
    axes[idx].set_xlabel('時間 t / T')
    axes[idx].set_ylabel('振幅')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='判定時点')
    axes[idx].legend(fontsize=8)

plt.suptitle('ロールオフ率 α によるアイパターンの変化', fontsize=13)
plt.tight_layout()
plt.show()

print("""
【観察ポイント】
- α = 0 (理想 sinc パルス): 目は開いていますが、裾の減衰が遅いため
  タイミング誤差に弱く、実装では使えません。
- α = 0.25 〜 0.5: 実用的なバランス。目が十分に開いています。
- α = 1.0: 最も目が大きく開き、タイミングマージンも最大ですが、
  帯域効率は α = 0 の半分になります。

■ まとめ
  ┌─────────────────────────────────────────────────────┐
  │  パルス整形の本質                                     │
  │                                                       │
  │  ・矩形パルスは帯域無限 → 帯域制限が必要            │
  │  ・ナイキスト条件: p(nT) = δ[n] → ISI フリー        │
  │  ・Raised Cosine: α で帯域幅 vs 時間減衰を制御      │
  │  ・Root RC を送受信に使用 → マッチドフィルタで最適  │
  │  ・アイパターン: ISI・ノイズ・タイミングの品質評価   │
  └─────────────────────────────────────────────────────┘

次のスクリプト (03_awgn.py) では、この整形された信号が
AWGN チャネルを通過したときの影響を学びます。
""")
