# -*- coding: utf-8 -*-
"""
====================================================================
 01. サンプリング定理とエイリアシング (Sampling Theorem & Aliasing)
====================================================================

このスクリプトでは、通信工学における最も基礎的な概念である
「サンプリング定理（標本化定理）」と「エイリアシング」を
段階的に学びます。

■ 学習のゴール
  1. 連続信号を離散化する「標本化」の数学的意味を理解する
  2. ナイキストレート（Nyquist rate）の意味を体感する
  3. ナイキスト条件を満たさない場合に何が起きるか（エイリアシング）を視覚的に確認する
  4. sinc 補間による信号の完全再構成を確認する

■ 使い方
  - 各セクションごとにコメントを読み、print() や plt.show() の出力を確認しながら
    パラメータを変えて実行してみてください。
  - 「★ 実験してみよう」の箇所は自分でパラメータを変えて試す部分です。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# --- 日本語フォント設定（環境に応じて変更してください）---
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'Hiragino Sans']
rcParams['axes.unicode_minus'] = False

print("=" * 60)
print(" セクション 1: 連続信号とは何か")
print("=" * 60)
print("""
通信の世界では、音声やセンサーの出力など、時間的に連続な
アナログ信号を扱います。これをコンピュータで処理するには、
信号を「離散化（サンプリング）」する必要があります。

まず、元となる連続信号を作ってみましょう。
ここでは周波数 f₀ = 5 Hz の正弦波を考えます。

  x(t) = sin(2π f₀ t)
""")

# ── 連続信号の生成 ──
f0 = 5          # 信号の周波数 [Hz]
T_total = 1.0   # 観測時間 [秒]
t_cont = np.linspace(0, T_total, 10000)  # 十分に細かい時間軸（連続の近似）
x_cont = np.sin(2 * np.pi * f0 * t_cont)

plt.figure(figsize=(12, 4))
plt.plot(t_cont, x_cont, 'b-', linewidth=1.5, label=f'連続信号 x(t) = sin(2π·{f0}·t)')
plt.xlabel('時間 t [秒]')
plt.ylabel('振幅')
plt.title(f'連続正弦波信号 (f₀ = {f0} Hz)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("=" * 60)
print(" セクション 2: サンプリング（標本化）とは")
print("=" * 60)
print(f"""
連続信号 x(t) を一定間隔 Tₛ で値を取り出す操作を
「サンプリング（標本化）」と呼びます。

サンプリング周波数 fₛ = 1/Tₛ は、1秒間に何回値を取り出すかを表します。

■ サンプリング定理（シャノン-ナイキストの定理）
  元の信号に含まれる最大周波数を f_max とすると、
  信号を完全に復元するには

    fₛ ≥ 2 · f_max   （ナイキストレート）

  を満たす必要があります。

今の例では f₀ = {f0} Hz なので、ナイキストレートは
  2 × {f0} = {2 * f0} Hz
です。

以下で、ナイキストレートを「満たす場合」と「満たさない場合」を比較しましょう。
""")

# ── サンプリング周波数を変えて比較 ──
fs_list = [50, 20, 10, 8]  # サンプリング周波数の候補
# 50 Hz → ナイキスト条件を十分に満たす (fₛ >> 2f₀)
# 20 Hz → ちょうど十分 (fₛ = 4f₀)
# 10 Hz → ギリギリ (fₛ = 2f₀ : ナイキストレートちょうど)
#  8 Hz → 条件を満たさない！(fₛ < 2f₀) → エイリアシング発生

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.ravel()

for i, fs in enumerate(fs_list):
    n_samples = int(T_total * fs)
    t_sampled = np.arange(n_samples) / fs
    x_sampled = np.sin(2 * np.pi * f0 * t_sampled)

    nyquist_ok = fs >= 2 * f0
    status = "✓ ナイキスト条件OK" if nyquist_ok else "✗ エイリアシング発生！"
    color = 'green' if nyquist_ok else 'red'

    axes[i].plot(t_cont, x_cont, 'b-', alpha=0.3, linewidth=1, label='元の連続信号')
    axes[i].stem(t_sampled, x_sampled, linefmt=f'{color[0]}-', markerfmt=f'{color[0]}o',
                 basefmt='k-', label=f'サンプル点 (fₛ={fs} Hz)')
    axes[i].set_title(f'fₛ = {fs} Hz  ({status})', fontsize=11,
                      color='darkgreen' if nyquist_ok else 'darkred')
    axes[i].set_xlabel('時間 t [秒]')
    axes[i].set_ylabel('振幅')
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.3)
    axes[i].set_xlim(0, 0.5)  # 最初の0.5秒を拡大

fig.suptitle(f'サンプリング周波数の違いによる標本化の比較 (元信号: f₀={f0} Hz)', fontsize=13)
plt.tight_layout()
plt.show()

print("""
【観察ポイント】
- fₛ = 50 Hz, 20 Hz: サンプル点が元の波形を正確に追跡しています。
- fₛ = 10 Hz: ナイキストレートちょうどですが、位相によっては
  うまく復元できない場合があります（理論的には可能）。
- fₛ = 8 Hz: ナイキスト条件を満たさないため、サンプル点から
  復元される波形は元と異なるものになります。これがエイリアシングです！

★ 実験してみよう: f0 の値を変えたり、fs_list に新しい値を追加して
  どのような変化が起きるか確認してみましょう。
""")

print("=" * 60)
print(" セクション 3: 周波数領域で見るエイリアシング")
print("=" * 60)
print("""
エイリアシングの本質を理解するには、周波数領域で見ることが重要です。

サンプリングは、時間領域では「信号 × インパルス列」に相当しますが、
周波数領域では「スペクトルが fₛ ごとに周期的に繰り返される」
ことを意味します。

  X_s(f) = fₛ Σ X(f - n·fₛ)   (n = ..., -1, 0, 1, ...)

fₛ < 2·f_max のとき、この繰り返しスペクトル同士が重なり合い（fold）、
元の信号が区別できなくなります。これがエイリアシングの正体です。
""")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.ravel()

for i, fs in enumerate(fs_list):
    n_samples = int(T_total * fs)
    t_sampled = np.arange(n_samples) / fs
    x_sampled = np.sin(2 * np.pi * f0 * t_sampled)

    # FFT（周波数スペクトル）
    X = np.fft.fft(x_sampled)
    freqs = np.fft.fftfreq(n_samples, d=1.0 / fs)

    # 正の周波数のみ表示
    pos_mask = freqs >= 0
    magnitude = np.abs(X[pos_mask]) / n_samples * 2

    axes[i].stem(freqs[pos_mask], magnitude, linefmt='b-', markerfmt='bo', basefmt='k-')
    axes[i].axvline(x=f0, color='green', linestyle='--', alpha=0.7, label=f'真の周波数 f₀={f0} Hz')
    axes[i].axvline(x=fs / 2, color='red', linestyle=':', alpha=0.7, label=f'ナイキスト周波数 fₛ/2={fs / 2} Hz')
    axes[i].set_title(f'周波数スペクトル (fₛ = {fs} Hz)')
    axes[i].set_xlabel('周波数 [Hz]')
    axes[i].set_ylabel('振幅')
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.3)

fig.suptitle('周波数領域で見るエイリアシング', fontsize=13)
plt.tight_layout()
plt.show()

print("""
【観察ポイント】
- fₛが十分大きい場合: 真の周波数 f₀ にのみピークが立ちます。
- fₛ < 2·f₀ の場合: ピークが本来の周波数とは別の場所に現れます。
  これは元の周波数が fₛ で「折り返された」エイリアスです。

  エイリアス周波数 = |f₀ - n·fₛ|  (最も小さい正の値)

★ 実験してみよう: f0 = 12 Hz, fs = 10 Hz として実行してみましょう。
  12 Hz の信号が何 Hz のエイリアスとして現れるか計算で予測し、
  グラフと比較してみてください。
  → 答え: |12 - 10| = 2 Hz
""")

print("=" * 60)
print(" セクション 4: sinc 補間による信号の完全再構成")
print("=" * 60)
print("""
サンプリング定理の核心は、ナイキスト条件を満たしていれば
離散サンプルから元の連続信号を「完全に」復元できるという点です。

復元の公式（Whittaker-Shannon 補間公式）：

  x(t) = Σ x[n] · sinc((t - nTₛ) / Tₛ)

ここで sinc(u) = sin(πu) / (πu) です。
各サンプル点に sinc 関数を置き、その重ね合わせで元の信号を再構成します。

sinc 関数の重要な性質:
  - sinc(0) = 1
  - sinc(n) = 0  (n が 0 以外の整数)

つまり、各 sinc 関数は対応するサンプル点でのみ値を持ち、
他のサンプル点では必ず 0 になるため、干渉しません。
これがナイキスト（第1）条件の本質です。
""")


def sinc_reconstruction(t_sampled, x_sampled, t_recon, Ts):
    """sinc 補間による信号再構成"""
    x_recon = np.zeros_like(t_recon)
    for n in range(len(t_sampled)):
        x_recon += x_sampled[n] * np.sinc((t_recon - t_sampled[n]) / Ts)
    return x_recon


# ── 再構成の比較 ──
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.ravel()

t_recon = np.linspace(0, T_total, 2000)  # 再構成用の時間軸
x_true = np.sin(2 * np.pi * f0 * t_recon)  # 理想的な連続信号

for i, fs in enumerate(fs_list):
    Ts = 1.0 / fs
    n_samples = int(T_total * fs)
    t_sampled = np.arange(n_samples) / fs
    x_sampled = np.sin(2 * np.pi * f0 * t_sampled)

    # sinc 補間で再構成
    x_reconstructed = sinc_reconstruction(t_sampled, x_sampled, t_recon, Ts)

    # 再構成誤差
    error = np.max(np.abs(x_true - x_reconstructed))

    axes[i].plot(t_recon, x_true, 'b-', alpha=0.3, linewidth=1, label='元の連続信号')
    axes[i].plot(t_recon, x_reconstructed, 'r-', linewidth=1.5, label='sinc 補間で再構成')
    axes[i].stem(t_sampled, x_sampled, linefmt='g-', markerfmt='go',
                 basefmt='k-', label='サンプル点')
    axes[i].set_title(f'fₛ = {fs} Hz  (最大誤差: {error:.4f})')
    axes[i].set_xlabel('時間 t [秒]')
    axes[i].set_ylabel('振幅')
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.3)
    axes[i].set_xlim(0.05, 0.45)  # 端の効果を避けて表示

fig.suptitle('sinc 補間による信号再構成の比較', fontsize=13)
plt.tight_layout()
plt.show()

print("""
【観察ポイント】
- fₛ ≥ 2f₀ の場合: 赤い再構成波形が青い元信号とほぼ完全に一致します！
  これがサンプリング定理の威力です。
- fₛ < 2f₀ の場合: sinc 補間しても元の信号は復元できません。
  一度エイリアシングが起きると、もう元には戻せないのです。

■ まとめ
  ┌─────────────────────────────────────────────┐
  │  サンプリング定理の本質                       │
  │                                               │
  │  fₛ ≥ 2·f_max を満たせば:                     │
  │    離散サンプル → sinc 補間 → 完全な復元     │
  │                                               │
  │  fₛ < 2·f_max の場合:                         │
  │    スペクトルが折り返し → エイリアシング      │
  │    → 情報が失われ復元不可能                   │
  └─────────────────────────────────────────────┘
""")

print("=" * 60)
print(" セクション 5: 複合信号のサンプリング")
print("=" * 60)
print("""
実際の信号は単一の正弦波ではなく、複数の周波数成分を含みます。
ここでは、3つの周波数成分を持つ信号でサンプリング定理を確認します。

  x(t) = sin(2π·3·t) + 0.5·sin(2π·7·t) + 0.3·sin(2π·15·t)

この信号の最大周波数は f_max = 15 Hz なので、
ナイキストレートは 2 × 15 = 30 Hz です。
""")

# 複合信号
def composite_signal(t):
    return np.sin(2 * np.pi * 3 * t) + 0.5 * np.sin(2 * np.pi * 7 * t) + 0.3 * np.sin(2 * np.pi * 15 * t)

t_fine = np.linspace(0, 1, 10000)
x_comp = composite_signal(t_fine)

fs_comp_list = [100, 30, 20]  # 十分 / ちょうど / 不足

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for i, fs in enumerate(fs_comp_list):
    Ts = 1.0 / fs
    n_samples = int(1.0 * fs)
    t_s = np.arange(n_samples) / fs
    x_s = composite_signal(t_s)

    x_r = sinc_reconstruction(t_s, x_s, t_fine, Ts)
    error = np.max(np.abs(x_comp - x_r))

    nyquist_ok = fs >= 30
    status = "✓ OK" if nyquist_ok else "✗ エイリアシング"

    axes[i].plot(t_fine, x_comp, 'b-', alpha=0.3, label='元の信号')
    axes[i].plot(t_fine, x_r, 'r-', linewidth=1.5, label='再構成')
    axes[i].set_title(f'fₛ={fs} Hz ({status})\n最大誤差: {error:.4f}', fontsize=10)
    axes[i].set_xlabel('時間 [秒]')
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.3)
    axes[i].set_xlim(0.1, 0.5)

fig.suptitle('複合信号 (f_max=15 Hz) のサンプリングと再構成', fontsize=13)
plt.tight_layout()
plt.show()

print("""
【まとめ】
複合信号でも同じ原理が成立します。
fₛ ≥ 2·f_max (= 30 Hz) を満たせば完全に復元可能ですが、
fₛ = 20 Hz では 15 Hz の成分がエイリアスとして混入し、
元の信号に戻せなくなります。

次のスクリプト (02_pulse_shaping.py) では、この離散化された信号を
通信路で送信する際に必要な「パルス整形」について学びます。
""")
