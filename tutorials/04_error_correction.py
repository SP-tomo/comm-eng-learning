# -*- coding: utf-8 -*-
"""
====================================================================
 04. 誤り訂正と符号化理論
     (Error Correction and Coding Theory)
====================================================================

このスクリプトでは、通信の信頼性を向上させるための「誤り訂正符号」の
基礎について学びます。学部3年生向けに、線形代数を用いた行列表現から、
ハミング符号のシンドローム復号、そして符号化利得までを網羅します。

■ 学習のゴール
  1. 情報ビットと冗長ビットの概念を理解する
  2. 生成行列(G)による符号化とパリティ検査行列(H)を理解する
  3. シンドローム(S)を用いた誤り検出と訂正のメカニズムを学ぶ
  4. 符号化利得 (Coding Gain) を確認し、SNRの改善度を評価する
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.special import erfc
import math

# --- 日本語フォント設定 ---
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'Hiragino Sans']
rcParams['axes.unicode_minus'] = False

print("=" * 60)
print(" セクション 1: 誤り訂正の基礎と線形ブロック符号")
print("=" * 60)
print("""
■ 冗長性の付加
  送信したい情報ビットに「冗長なビット（パリティ）」を付け加えることで、
  受信側でノイズによるエラーを検知し、さらには元の正しいビットへ
  「訂正」することができます。

■ 線形ブロック符号
  情報ベクトル d (kビット) に対して、符号語 c (nビット) を生成します。
  n > k であり、(n-k)ビットが冗長ビットです。
  符号化は、生成行列 G (k × n) を用いて以下のように表されます。

      c = d G  (mod 2)

  ここでは、もっとも有名な「ハミング (7, 4) 符号」を扱います。
  - n = 7 (全体のビット数)
  - k = 4 (情報ビット数)
  - n - k = 3 (パリティビット数)
""")

# ハミング(7,4)符号の行列定義
# 情報ビット d = [d1, d2, d3, d4]
# パリティビット p = [p1, p2, p3]
# p1 = d1 + d2 + d4
# p2 = d1 + d3 + d4
# p3 = d2 + d3 + d4

G = np.array([
    [1, 0, 0, 0,  1, 1, 0],
    [0, 1, 0, 0,  1, 0, 1],
    [0, 0, 1, 0,  0, 1, 1],
    [0, 0, 0, 1,  1, 1, 1]
])

H = np.array([
    [1, 1, 0, 1,  1, 0, 0],
    [1, 0, 1, 1,  0, 1, 0],
    [0, 1, 1, 1,  0, 0, 1]
])

print("【行列の定義】")
print("生成行列 G (4x7):")
print(G)
print("\\nパリティ検査行列 H (3x7):")
print(H)
print("\\n※ G と H は、GH^T = 0 (mod 2) を満たすように設計されています。")

print("\\n" + "=" * 60)
print(" セクション 2: 符号化、通信路、シンドローム復号のシミュレーション")
print("=" * 60)

# 1. 符号化
d = np.array([1, 0, 1, 1])
print(f"1. 送信したい情報ビット d: {d}")

c = np.dot(d, G) % 2
print(f"2. 符号化後の送信語 c = dG: {c}")

# 2. 通信路 (1ビットのエラーを発生させる)
error_pos = 2  # 3番目のビット(インデックス2)を反転
e = np.zeros(7, dtype=int)
e[error_pos] = 1

r = (c + e) % 2
print(f"3. 通信路でのエラー e     : {e}")
print(f"4. 受信語 r = c + e       : {r}")

# 3. シンドローム計算と復号
# S = r H^T = (c + e) H^T = c H^T + e H^T = e H^T
S = np.dot(r, H.T) % 2
print(f"5. シンドローム S = rH^T  : {S}")

print("\\n【復号プロセス】")
if np.all(S == 0):
    print("シンドロームが [0 0 0] なので、エラーなしと判定します。")
else:
    # Hの列ベクトルとシンドロームを比較
    for i in range(7):
        if np.array_equal(S, H[:, i]):
            print(f"シンドローム S が H の {i+1} 列目と一致しました！")
            print(f"→ {i+1} ビット目にエラーが発生したと推定し、訂正します。")
            r_corrected = r.copy()
            r_corrected[i] = (r_corrected[i] + 1) % 2
            print(f"訂正後の符号語: {r_corrected}")
            print(f"抽出した情報d': {r_corrected[:4]}")
            if np.array_equal(r_corrected[:4], d):
                print("✅ 誤り訂正に成功し、元の情報を復元しました！")
            break

print("\\n" + "=" * 60)
print(" セクション 3: 符号化利得 (Coding Gain) の評価")
print("=" * 60)
print("""
誤り訂正符号を導入することで、SNR（Eb/N0）がどれだけ改善するかを
BER（ビット誤り率）カーブを描いて比較します。

- 符号化率 (Code Rate) R = k/n = 4/7
- 符号化なし BPSK: 
    P_b = Q(sqrt(2 * Eb/N0))
- 符号化あり BPSK (硬判定復号): 
    通信路エラー確率 p_c = Q(sqrt(2 * R * Eb/N0))
    ※冗長ビットの分、1ビットあたりの送信エネルギーが減るため。
    
    ブロック誤り確率 P_w は、2ビット以上のエラーが起きる確率:
    P_w = sum_{i=2}^{7} nCr(7,i) p_c^i (1-p_c)^(7-i)
    
    近似的に、ビット誤り率 P_b は (3/7) P_w となります。
""")

def nCr(n, r):
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))

ebn0_db = np.arange(0, 11, 0.5)
ebn0_lin = 10**(ebn0_db/10)

# 1. 符号化なし (Uncoded BPSK)
ber_uncoded = 0.5 * erfc(np.sqrt(ebn0_lin))

# 2. 符号化あり (Hamming 7,4 Hard Decision)
R = 4 / 7
p_c = 0.5 * erfc(np.sqrt(ebn0_lin * R))

ber_coded = np.zeros_like(p_c)
for idx_p, p in enumerate(p_c):
    p_w = 0
    for i in range(2, 8):
        p_w += nCr(7, i) * (p**i) * ((1-p)**(7-i))
    # ハミング(7,4)の誤りパターンの重み分布に基づく近似: P_b ≈ (3/7) P_w
    ber_coded[idx_p] = (3/7) * p_w

fig, ax = plt.subplots(figsize=(10, 7))
ax.semilogy(ebn0_db, ber_uncoded, 'b-', linewidth=2, label='符号化なし (Uncoded BPSK)')
ax.semilogy(ebn0_db, ber_coded, 'r--', linewidth=2, label='ハミング(7,4)符号化')

ax.set_xlabel('Eb/N₀ [dB]', fontsize=12)
ax.set_ylabel('BER (ビット誤り率)', fontsize=12)
ax.set_title('符号化利得 (Coding Gain) の確認', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, which='both', alpha=0.4)
ax.set_ylim(1e-6, 1)
ax.set_xlim(0, 10)

# コーディングゲインの注釈
ax.annotate('低いEb/N0では\\n符号化ロスが発生', xy=(2, 1e-2), xytext=(3, 1e-1),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))
ax.annotate('高いEb/N0で\\n符号化利得(Coding Gain)が\\n得られる', xy=(8, 1e-4), xytext=(5, 1e-5),
            arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6))

plt.tight_layout()
plt.show()

print("""
【観察ポイント】
- グラフが交差する点（約 5.5 dB付近）に注目してください。
- SNRが低い（ノイズが大きい）領域では、冗長ビットを追加するために
  エネルギーが分散されるデメリット（符号化ロス）が上回ります。
- SNRが高い領域では、エラー訂正の効果がロスを上回り、
  同じBERを達成するための必要Eb/N0が減少します（符号化利得）。
""")
