# -*- coding: utf-8 -*-
"""
====================================================================
 03. AWGN（加法性白色ガウス雑音）と変調方式
     (Additive White Gaussian Noise & Digital Modulation)
====================================================================

このスクリプトでは、通信路雑音の最も基本的なモデルである AWGN と、
代表的なディジタル変調方式の性能を学びます。

■ 学習のゴール
  1. AWGN の統計的性質（ガウス分布、白色性）を理解する
  2. 連続領域での AWGN の影響（波形劣化）を確認する
  3. 離散領域（コンスタレーション）で変調方式の耐ノイズ性を比較する
  4. BER（ビット誤り率）の理論値とシミュレーション値を比較する
  5. 各変調方式（BPSK, QPSK, 8PSK, 16QAM, 64QAM）の特徴を理解する

■ 使い方
  各セクションを順に読み進め、グラフの変化を確認してください。
  SNR の値を変えて、ノイズの影響がどう変わるか試しましょう。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.special import erfc

# --- 日本語フォント設定 ---
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'Hiragino Sans']
rcParams['axes.unicode_minus'] = False

print("=" * 60)
print(" セクション 1: AWGN とは何か？")
print("=" * 60)
print("""
■ AWGN (Additive White Gaussian Noise) の3つの性質

  (A) Additive（加法性）:
      受信信号 = 送信信号 + 雑音
      r(t) = s(t) + n(t)

      → 雑音は信号に足し算で加わります。掛け算やフェージングのような
        複雑な干渉ではなく、最もシンプルなモデルです。

  (W) White（白色）:
      雑音の電力スペクトル密度が全周波数帯で一定
      S_n(f) = N₀/2  [W/Hz]

      → どの周波数にも均等にノイズが乗ります。
      → 「白色」は白い光が全波長を含むことに由来します。
      → 自己相関関数はデルタ関数: R_n(τ) = (N₀/2)δ(τ)

  (G) Gaussian（ガウス性）:
      雑音の振幅は正規分布に従う
      n(t) ~ N(0, σ²)

      → 中心極限定理により、多数の独立な微小雑音源の合計は
        ガウス分布に近づきます。熱雑音がその代表例です。
""")

# ── AWGN の性質の可視化 ──
np.random.seed(42)
N = 10000
sigma = 1.0
noise = np.random.normal(0, sigma, N)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# (a) 時間波形
axes[0, 0].plot(noise[:500], 'b-', linewidth=0.5, alpha=0.8)
axes[0, 0].set_title('(a) AWGN の時間波形')
axes[0, 0].set_xlabel('サンプル')
axes[0, 0].set_ylabel('振幅')
axes[0, 0].grid(True, alpha=0.3)

# (b) ヒストグラム（ガウス分布の確認）
axes[0, 1].hist(noise, bins=80, density=True, alpha=0.7, color='skyblue', edgecolor='blue')
x_gauss = np.linspace(-4 * sigma, 4 * sigma, 200)
pdf_gauss = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-x_gauss ** 2 / (2 * sigma ** 2))
axes[0, 1].plot(x_gauss, pdf_gauss, 'r-', linewidth=2, label=f'N(0, {sigma}²)')
axes[0, 1].set_title('(b) ヒストグラム vs ガウス分布')
axes[0, 1].set_xlabel('振幅')
axes[0, 1].set_ylabel('確率密度')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# (c) 自己相関（白色性の確認）
autocorr = np.correlate(noise[:1000], noise[:1000], mode='full')
autocorr = autocorr / np.max(autocorr)
lags = np.arange(-999, 1000)
axes[1, 0].plot(lags, autocorr, 'g-', linewidth=0.5)
axes[1, 0].set_title('(c) 自己相関関数 → δ(τ) に近い')
axes[1, 0].set_xlabel('ラグ τ')
axes[1, 0].set_ylabel('正規化自己相関')
axes[1, 0].set_xlim(-50, 50)
axes[1, 0].grid(True, alpha=0.3)

# (d) パワースペクトル密度（白色性の確認）
psd = np.abs(np.fft.fft(noise)) ** 2 / N
freqs_psd = np.fft.fftfreq(N)
pos = freqs_psd > 0
axes[1, 1].plot(freqs_psd[pos], 10 * np.log10(psd[pos]), 'purple', alpha=0.5, linewidth=0.5)
axes[1, 1].axhline(y=10 * np.log10(np.mean(psd[pos])), color='red', linestyle='--',
                    label=f'平均 PSD = {10 * np.log10(np.mean(psd[pos])):.1f} dB')
axes[1, 1].set_title('(d) パワースペクトル密度 → ほぼ一定（白色）')
axes[1, 1].set_xlabel('正規化周波数')
axes[1, 1].set_ylabel('PSD [dB]')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('AWGN の基本性質', fontsize=13)
plt.tight_layout()
plt.show()

print("""
【観察ポイント】
- (a): 不規則に振動する波形。振幅に周期性はありません。
- (b): ヒストグラムがガウス分布（赤線）にきれいに一致しています。
- (c): 自己相関がτ=0でのみピーク → 白色（無相関）
- (d): PSDがほぼ水平 → 全周波数に均等にエネルギーが分布（白色）
""")

print("=" * 60)
print(" セクション 2: 連続領域での AWGN の影響")
print("=" * 60)
print("""
送信信号 s(t) に AWGN n(t) が加わった受信信号 r(t) = s(t) + n(t) を
視覚的に確認しましょう。

SNR（信号対雑音電力比）を変えて、劣化の度合いを比較します。

  SNR [dB] = 10 · log₁₀(P_signal / P_noise)
""")

# ── 送信信号の生成 ──
t = np.linspace(0, 1, 1000)
f_carrier = 5
signal = np.sin(2 * np.pi * f_carrier * t)
P_signal = np.mean(signal ** 2)

snr_dB_list = [20, 10, 3, 0]

fig, axes = plt.subplots(len(snr_dB_list), 1, figsize=(14, 12))

for i, snr_dB in enumerate(snr_dB_list):
    snr_linear = 10 ** (snr_dB / 10)
    P_noise = P_signal / snr_linear
    sigma_noise = np.sqrt(P_noise)
    noise = np.random.normal(0, sigma_noise, len(t))
    received = signal + noise

    axes[i].plot(t, signal, 'b-', linewidth=1, alpha=0.5, label='送信信号 s(t)')
    axes[i].plot(t, received, 'r-', linewidth=0.5, alpha=0.8, label='受信信号 r(t) = s(t) + n(t)')
    axes[i].set_title(f'SNR = {snr_dB} dB  (σ_noise = {sigma_noise:.3f})')
    axes[i].set_ylabel('振幅')
    axes[i].legend(fontsize=8, loc='upper right')
    axes[i].grid(True, alpha=0.3)

axes[-1].set_xlabel('時間 [秒]')
plt.suptitle('連続領域における AWGN の影響 — SNR による比較', fontsize=13)
plt.tight_layout()
plt.show()

print("""
【観察ポイント】
- SNR = 20 dB: ノイズはほとんど見えず、元の信号がきれいに見えます。
- SNR = 10 dB: わずかにノイズが乗っていますが、波形は追跡可能。
- SNR = 3 dB: ノイズと信号が同程度で、波形が歪み始めます。
- SNR = 0 dB: 信号電力 = ノイズ電力。元の波形がほぼ見えなくなります。

★ 実験: snr_dB_list の値を変えて試してみましょう。
  例えば [-3, -6] を追加すると、ノイズが信号より大きい状態を観察できます。
""")

print("=" * 60)
print(" セクション 3: ディジタル変調方式の基礎")
print("=" * 60)
print("""
ディジタル変調では、ビット列をシンボル（信号点）に割り当てて送信します。
I-Q 平面（同相・直交成分）上の点として表現されます。

■ 主な変調方式:
  BPSK  : 1ビット/シンボル, 2点 (位相変調)
  QPSK  : 2ビット/シンボル, 4点 (位相変調)
  8PSK  : 3ビット/シンボル, 8点 (位相変調)
  16QAM : 4ビット/シンボル, 16点 (振幅+位相変調)
  64QAM : 6ビット/シンボル, 64点 (振幅+位相変調)

シンボルあたりのビット数が多いほど:
  ✓ 帯域効率が高い（同じ帯域で多くのデータを送れる）
  ✗ ノイズに弱い（信号点間の距離が近くなるため）
""")


# ── 変調方式の定義 ──
def bpsk_constellation():
    """BPSK: ±1"""
    return np.array([-1 + 0j, 1 + 0j]), 1


def qpsk_constellation():
    """QPSK: 45°間隔の4点"""
    return np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2), 2


def psk8_constellation():
    """8PSK: 45°間隔の8点"""
    angles = np.array([0, 1, 2, 3, 4, 5, 6, 7]) * 2 * np.pi / 8
    return np.exp(1j * angles), 3


def qam16_constellation():
    """16QAM: 4x4格子"""
    points = []
    for i in [-3, -1, 1, 3]:
        for q in [-3, -1, 1, 3]:
            points.append(i + 1j * q)
    c = np.array(points)
    c = c / np.sqrt(np.mean(np.abs(c) ** 2))  # 平均電力を1に正規化
    return c, 4


def qam64_constellation():
    """64QAM: 8x8格子"""
    points = []
    for i in [-7, -5, -3, -1, 1, 3, 5, 7]:
        for q in [-7, -5, -3, -1, 1, 3, 5, 7]:
            points.append(i + 1j * q)
    c = np.array(points)
    c = c / np.sqrt(np.mean(np.abs(c) ** 2))
    return c, 6


# ── コンスタレーション表示 ──
mod_funcs = {
    'BPSK': bpsk_constellation,
    'QPSK': qpsk_constellation,
    '8PSK': psk8_constellation,
    '16QAM': qam16_constellation,
    '64QAM': qam64_constellation,
}

fig, axes = plt.subplots(1, 5, figsize=(18, 4))

for idx, (name, func) in enumerate(mod_funcs.items()):
    constellation, bps = func()
    axes[idx].scatter(constellation.real, constellation.imag,
                      c='blue', s=60, zorder=5, edgecolors='black')
    axes[idx].set_title(f'{name}\n({bps} bit/symbol, {len(constellation)} points)')
    axes[idx].set_xlabel('I (同相)')
    axes[idx].set_ylabel('Q (直交)')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_aspect('equal')
    lim = max(np.abs(constellation.real).max(), np.abs(constellation.imag).max()) * 1.4
    axes[idx].set_xlim(-lim, lim)
    axes[idx].set_ylim(-lim, lim)
    axes[idx].axhline(0, color='gray', linewidth=0.5)
    axes[idx].axvline(0, color='gray', linewidth=0.5)

plt.suptitle('主要なディジタル変調方式のコンスタレーション', fontsize=13)
plt.tight_layout()
plt.show()

print("""
【観察ポイント】
- BPSK: 最もシンプル。I軸上の2点のみ。信号点間距離が最大 → 最もノイズに強い。
- QPSK: I-Q平面の4象限に1点ずつ。BPSKと同じ帯域で2倍のデータを送れる。
- 8PSK: 円周上に8点。帯域効率 ↑ だが、隣接点との距離 ↓
- 16QAM/64QAM: 格子状。高い帯域効率だが、内側の点はノイズに非常に弱い。
""")

print("=" * 60)
print(" セクション 4: 離散領域での AWGN — コンスタレーションの劣化")
print("=" * 60)
print("""
AWGN チャネルを通過すると、受信シンボルは I, Q 各成分に独立な
ガウスノイズが加わり、理想的な信号点のまわりに「ばらつき」ます。

  受信シンボル r = s + n
  n ~ CN(0, σ²)  (複素ガウス雑音)

SNR(Eb/N0) を変化させて、コンスタレーションがどう劣化するか確認しましょう。
""")


def simulate_awgn_channel(constellation, bps, num_symbols, EbN0_dB):
    """AWGN チャネルのシミュレーション"""
    # ランダムにシンボルを選択
    indices = np.random.randint(0, len(constellation), num_symbols)
    tx_symbols = constellation[indices]

    # Eb/N0 から ノイズ分散を計算
    EbN0_linear = 10 ** (EbN0_dB / 10)
    # Es/N0 = bps * Eb/N0
    EsN0_linear = bps * EbN0_linear
    # ノイズの標準偏差 (複素雑音: I, Q 各成分が σ²/2 の分散)
    noise_std = np.sqrt(1 / (2 * EsN0_linear))

    # AWGN の付加
    noise = noise_std * (np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols))
    rx_symbols = tx_symbols + noise

    return tx_symbols, rx_symbols, indices


np.random.seed(0)
num_sym = 2000
EbN0_list = [20, 10, 5, 0]

# QPSK での SNR 変化
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
constellation_qpsk, bps_qpsk = qpsk_constellation()

for i, EbN0_dB in enumerate(EbN0_list):
    tx, rx, _ = simulate_awgn_channel(constellation_qpsk, bps_qpsk, num_sym, EbN0_dB)
    axes[i].scatter(rx.real, rx.imag, c='red', s=2, alpha=0.3, label='受信')
    axes[i].scatter(constellation_qpsk.real, constellation_qpsk.imag,
                    c='blue', s=100, zorder=5, edgecolors='black', label='理想点')
    axes[i].set_title(f'Eb/N₀ = {EbN0_dB} dB')
    axes[i].set_xlabel('I')
    axes[i].set_ylabel('Q')
    axes[i].set_aspect('equal')
    axes[i].grid(True, alpha=0.3)
    axes[i].set_xlim(-2, 2)
    axes[i].set_ylim(-2, 2)
    if i == 0:
        axes[i].legend(fontsize=8)

plt.suptitle('QPSK コンスタレーションの AWGN による劣化', fontsize=13)
plt.tight_layout()
plt.show()

print("""
【観察ポイント】
- Eb/N₀ = 20 dB: 理想信号点の周りにほぼ集中。誤りはほぼゼロ。
- Eb/N₀ = 10 dB: 少しばらつくが、判定領域内に収まっている。
- Eb/N₀ = 5 dB: かなりばらつき、一部は隣の判定領域に侵入。
- Eb/N₀ = 0 dB: ばらつきが非常に大きく、頻繁に誤りが発生。

ノイズ分布が2次元のガウス分布（等方的な円形の広がり）であることに注目！
""")

# ── 各変調方式の比較 (Eb/N0 = 10 dB) ──
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
EbN0_compare = 10

for idx, (name, func) in enumerate(mod_funcs.items()):
    const, bps = func()
    _, rx, _ = simulate_awgn_channel(const, bps, 3000, EbN0_compare)
    axes[idx].scatter(rx.real, rx.imag, c='red', s=2, alpha=0.3)
    axes[idx].scatter(const.real, const.imag, c='blue', s=60, zorder=5, edgecolors='black')
    axes[idx].set_title(f'{name} (Eb/N₀={EbN0_compare} dB)')
    axes[idx].set_xlabel('I')
    axes[idx].set_ylabel('Q')
    axes[idx].set_aspect('equal')
    axes[idx].grid(True, alpha=0.3)
    lim = max(np.abs(const.real).max(), np.abs(const.imag).max()) * 2
    axes[idx].set_xlim(-lim, lim)
    axes[idx].set_ylim(-lim, lim)

plt.suptitle(f'各変調方式のコンスタレーション (Eb/N₀ = {EbN0_compare} dB)', fontsize=13)
plt.tight_layout()
plt.show()

print("""
【観察ポイント】
同じ Eb/N₀ でも、変調方式によってばらつきの「相対的な影響」が異なります。
- BPSK/QPSK: 信号点間距離が大きいため、まだ十分に判定可能。
- 8PSK: 隣接点が近いため、ばらつきが判定境界に達しやすい。
- 16QAM/64QAM: 内側の点は特にマージンが小さく、誤りやすい。
""")

print("=" * 60)
print(" セクション 5: BER（ビット誤り率）— 理論値 vs シミュレーション")
print("=" * 60)
print("""
■ BER (Bit Error Rate) とは？
  送信ビット数に対する誤りビット数の割合です。

■ 理論的 BER（AWGN チャネル）
  - BPSK:  BER = (1/2) erfc(√(Eb/N₀))
  - QPSK:  BER = (1/2) erfc(√(Eb/N₀))  ← BPSK と同じ！
  - 8PSK:  BER ≈ (1/3) erfc(√(3·Eb/N₀)·sin(π/8))
  - M-QAM: BER ≈ (2(√M-1))/(√M·log₂M) erfc(√(3·log₂M·Eb/N₀/(2(M-1))))

  QPSKとBPSKのBERが同じなのは重要なポイントです:
  QPSKは帯域効率2倍なのに、ビット誤り率は同じです！
""")


def theoretical_ber(EbN0_dB, modulation):
    """変調方式ごとの理論 BER"""
    EbN0 = 10 ** (EbN0_dB / 10)
    if modulation in ['BPSK', 'QPSK']:
        return 0.5 * erfc(np.sqrt(EbN0))
    elif modulation == '8PSK':
        return (1 / 3) * erfc(np.sqrt(3 * EbN0) * np.sin(np.pi / 8))
    elif modulation == '16QAM':
        M = 16
        return (3 / (2 * np.log2(M))) * erfc(np.sqrt(3 * np.log2(M) * EbN0 / (2 * (M - 1))))
    elif modulation == '64QAM':
        M = 64
        return (7 / (3 * np.log2(M))) * erfc(np.sqrt(3 * np.log2(M) * EbN0 / (2 * (M - 1))))


def simulate_ber(constellation, bps, EbN0_dB, num_symbols=100000):
    """BER のモンテカルロシミュレーション"""
    _, rx, tx_indices = simulate_awgn_channel(constellation, bps, num_symbols, EbN0_dB)

    # 最尤検出（最も近い信号点を選ぶ）
    rx_indices = np.argmin(np.abs(rx[:, None] - constellation[None, :]), axis=1)

    # ビット誤りのカウント（グレイ符号を仮定し、インデックスのXORでビット誤りを数える）
    bit_errors = 0
    for i in range(len(tx_indices)):
        diff = tx_indices[i] ^ rx_indices[i]
        bit_errors += bin(diff).count('1')

    total_bits = num_symbols * bps
    return bit_errors / total_bits


# ── BER カーブの描画 ──
EbN0_range = np.arange(0, 16, 0.5)
EbN0_sim_points = np.arange(0, 14, 2)

fig, ax = plt.subplots(figsize=(10, 7))

colors_ber = {'BPSK': '#1f77b4', 'QPSK': '#ff7f0e', '8PSK': '#2ca02c',
              '16QAM': '#d62728', '64QAM': '#9467bd'}

for name, func in mod_funcs.items():
    # 理論曲線
    ber_theory = theoretical_ber(EbN0_range, name)
    ber_theory = np.maximum(ber_theory, 1e-8)  # 下限
    ax.semilogy(EbN0_range, ber_theory, '-', color=colors_ber[name],
                linewidth=2, label=f'{name} (理論)')

    # シミュレーション
    const, bps = func()
    ber_sim = []
    for ebn0 in EbN0_sim_points:
        ber = simulate_ber(const, bps, ebn0, num_symbols=50000)
        ber_sim.append(max(ber, 1e-8))
    ax.semilogy(EbN0_sim_points, ber_sim, 'o', color=colors_ber[name],
                markersize=8, markeredgecolor='black')

ax.set_xlabel('Eb/N₀ [dB]', fontsize=12)
ax.set_ylabel('BER (ビット誤り率)', fontsize=12)
ax.set_title('BER カーブ: 理論値 (線) vs シミュレーション (○)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, which='both', alpha=0.3)
ax.set_ylim(1e-6, 1)
ax.set_xlim(0, 15)

plt.tight_layout()
plt.show()

print("""
【観察ポイント】
- 実線（理論値）と○（シミュレーション）がよく一致していることを確認！
- BPSK と QPSK の曲線が重なっている → 帯域効率2倍でも BER は同じ！
- 8PSK は QPSK より約 3 dB 劣化。帯域効率の代償です。
- 16QAM は QPSK より帯域効率2倍だが、約 4 dB の SNR 劣化。
- 64QAM は最も帯域効率が高い (6 bit/symbol) が、最もノイズに弱い。

■ 通信システム設計のトレードオフ
  ┌──────────────────────────────────────────────────────┐
  │  帯域効率 ↑  ⟺  必要 SNR ↑  ⟺  ノイズ耐性 ↓   │
  │                                                       │
  │  BPSK  → 低速だが頑健（衛星通信、深宇宙通信）      │
  │  QPSK  → BPSKと同じBERで2倍の速度（衛星、Wi-Fi）   │
  │  16QAM → 高速だがSNR要求が高い（Wi-Fi, LTE）       │
  │  64QAM → 最高速だが良好なチャネルが必要（5G, 光）   │
  └──────────────────────────────────────────────────────┘
""")

print("=" * 60)
print(" セクション 6: 離散 AWGN チャネルモデル")
print("=" * 60)
print("""
■ 連続 AWGN vs 離散 AWGN

  連続チャネル:
    r(t) = s(t) + n(t),  n(t) は連続時間ガウス過程

  離散チャネル (標本化後):
    r[k] = s[k] + n[k],  n[k] ~ N(0, σ²) は iid ガウス列

  通信システムでは、整合フィルタ（マッチドフィルタ）の出力を
  シンボルレートで標本化した後のモデルが離散 AWGN チャネルです。

  離散チャネルでの分析は以下の利点があります:
  - 数学的に扱いやすい
  - シンボルごとの判定理論（ML判定）が直接適用可能
  - 情報理論的な容量計算が容易

■ チャネル容量 (Shannon limit)
  C = W · log₂(1 + SNR)  [bit/s]

  ここで W は帯域幅、SNR は信号対雑音電力比です。
  これは、AWGN チャネルで誤りなく通信可能な最大レートの理論限界です。
""")

# ── 離散 AWGN のデモ ──
np.random.seed(123)
num_demo = 50
bits = np.random.randint(0, 2, num_demo)
symbols_bpsk = 2 * bits - 1  # BPSK: 0 → -1, 1 → +1

fig, axes = plt.subplots(3, 1, figsize=(14, 9))
snrs_demo = [20, 5, 0]

for i, snr_dB in enumerate(snrs_demo):
    snr_lin = 10 ** (snr_dB / 10)
    sigma_n = np.sqrt(1 / (2 * snr_lin))
    noise_disc = np.random.normal(0, sigma_n, num_demo)
    rx = symbols_bpsk + noise_disc

    # 判定
    decided = np.sign(rx)
    errors = decided != symbols_bpsk
    num_errors = np.sum(errors)

    k = np.arange(num_demo)
    axes[i].stem(k, symbols_bpsk, linefmt='b-', markerfmt='bo', basefmt=' ',
                 label='送信シンボル s[k]')
    axes[i].stem(k + 0.2, rx, linefmt='r-', markerfmt='r^', basefmt=' ',
                 label='受信シンボル r[k]')
    axes[i].axhline(0, color='gray', linewidth=1, linestyle='--', label='判定境界')

    # 誤りをハイライト
    if np.any(errors):
        axes[i].scatter(k[errors] + 0.2, rx[errors], color='yellow', s=100,
                        zorder=10, edgecolors='red', linewidths=2, label=f'誤り ({num_errors}個)')

    axes[i].set_title(f'離散 AWGN チャネル (Eb/N₀ = {snr_dB} dB, 誤り数: {num_errors}/{num_demo})')
    axes[i].set_ylabel('振幅')
    axes[i].legend(fontsize=8, loc='upper right')
    axes[i].grid(True, alpha=0.3)
    axes[i].set_ylim(-3, 3)

axes[-1].set_xlabel('シンボルインデックス k')
plt.suptitle('離散 AWGN チャネルモデル (BPSK)', fontsize=13)
plt.tight_layout()
plt.show()

print("""
【観察ポイント】
- 青（送信）と赤（受信）の差がノイズです。
- 判定境界（y=0）を越えた受信シンボル（黄色でハイライト）がビット誤りです。
- SNR が低いほど、受信点が判定境界を越えやすくなります。

■ まとめ
  ┌──────────────────────────────────────────────────────┐
  │  AWGN の本質                                          │
  │                                                       │
  │  ・加法性: r = s + n（シンプルだが強力なモデル）      │
  │  ・白色: 全周波数に均等なエネルギー → δ 自己相関     │
  │  ・ガウス: 中心極限定理の帰結、熱雑音の良い近似      │
  │                                                       │
  │  ・連続 AWGN: r(t) = s(t) + n(t)                     │
  │  ・離散 AWGN: r[k] = s[k] + n[k], n[k] ~ iid N(0,σ²) │
  │                                                       │
  │  ・BER は Eb/N₀ で整理される                          │
  │  ・帯域効率 vs ノイズ耐性のトレードオフが核心         │
  │  ・Shannon 限界が究極の理論限界を与える              │
  └──────────────────────────────────────────────────────┘

おめでとうございます！3つの解説スクリプトを通じて、
サンプリング → パルス整形 → AWGN という通信の基本チェーンを学びました。

Streamlit アプリ (app.py) では、これらのパラメータをインタラクティブに
操作して、さらに深く理解を深めることができます。

  > streamlit run app.py
""")
