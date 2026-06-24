# -*- coding: utf-8 -*-
"""
====================================================================
 通信工学インタラクティブ学習システム
 (Communication Engineering Interactive Learning System)
====================================================================
起動方法: streamlit run app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import erfc

st.set_page_config(
    page_title="通信工学 学習システム",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── カスタムCSS ──
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1E88E5;
        font-size: 2.2em;
        margin-bottom: 0.3em;
    }
    .sub-title {
        text-align: center;
        color: #666;
        font-size: 1em;
        margin-bottom: 2em;
    }
    .concept-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        border-radius: 12px;
        padding: 1.2em;
        margin: 1em 0;
        border-left: 5px solid #1E88E5;
    }
    .formula-box {
        background: #f5f5f5;
        border-radius: 8px;
        padding: 1em;
        margin: 0.5em 0;
        font-family: 'Courier New', monospace;
        border: 1px solid #ddd;
    }
    .insight-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
        border-radius: 12px;
        padding: 1.2em;
        margin: 1em 0;
        border-left: 5px solid #43A047;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📡 通信工学 インタラクティブ学習システム</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">サンプリング定理 / パルス整形 / AWGN チャネル</div>', unsafe_allow_html=True)

# ====================================================================
#  ヘルパー関数
# ====================================================================

def raised_cosine(t, T, alpha):
    """Raised Cosine パルス"""
    result = np.zeros_like(t, dtype=float)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            result[i] = 1.0
        elif alpha > 0 and abs(abs(2 * alpha * ti / T) - 1.0) < 1e-12:
            result[i] = (np.pi / (4 * T)) * np.sinc(1.0 / (2 * alpha))
        else:
            result[i] = np.sinc(ti / T) * np.cos(np.pi * alpha * ti / T) / (1 - (2 * alpha * ti / T) ** 2)
    return result


def rc_spectrum(f, T, alpha):
    """Raised Cosine の周波数特性"""
    H = np.zeros_like(f, dtype=float)
    for i, fi in enumerate(f):
        fi_abs = abs(fi)
        if fi_abs <= (1 - alpha) / (2 * T):
            H[i] = T
        elif fi_abs <= (1 + alpha) / (2 * T):
            H[i] = T / 2 * (1 + np.cos(np.pi * T / alpha * (fi_abs - (1 - alpha) / (2 * T))))
    return H


def sinc_reconstruction(t_sampled, x_sampled, t_recon, Ts):
    """sinc 補間による再構成"""
    x_recon = np.zeros_like(t_recon)
    for n in range(len(t_sampled)):
        x_recon += x_sampled[n] * np.sinc((t_recon - t_sampled[n]) / Ts)
    return x_recon


def get_constellation(mod_type):
    """変調方式のコンスタレーション取得"""
    if mod_type == 'BPSK':
        return np.array([-1 + 0j, 1 + 0j]), 1
    elif mod_type == 'QPSK':
        return np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2), 2
    elif mod_type == '8PSK':
        angles = np.arange(8) * 2 * np.pi / 8
        return np.exp(1j * angles), 3
    elif mod_type == '16QAM':
        pts = [i + 1j * q for i in [-3, -1, 1, 3] for q in [-3, -1, 1, 3]]
        c = np.array(pts)
        return c / np.sqrt(np.mean(np.abs(c) ** 2)), 4
    elif mod_type == '64QAM':
        pts = [i + 1j * q for i in [-7, -5, -3, -1, 1, 3, 5, 7] for q in [-7, -5, -3, -1, 1, 3, 5, 7]]
        c = np.array(pts)
        return c / np.sqrt(np.mean(np.abs(c) ** 2)), 6


def theoretical_ber(EbN0_dB, modulation):
    """理論 BER"""
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


# ====================================================================
#  タブ構成
# ====================================================================

tab1, tab2, tab3 = st.tabs([
    "🔬 サンプリング定理 & エイリアシング",
    "🎛️ パルス整形 & アイパターン",
    "📊 AWGN & 変調方式"
])

# ====================================================================
#  タブ 1: サンプリング定理
# ====================================================================

with tab1:
    st.markdown("## 🔬 サンプリング定理とエイリアシング")

    st.markdown("""
    <div class="concept-box">
    <strong>📚 サンプリング定理（Shannon-Nyquist）</strong><br>
    信号の最大周波数を <em>f<sub>max</sub></em> とすると、
    <em>f<sub>s</sub> ≥ 2·f<sub>max</sub></em> のレートでサンプリングすれば
    元の信号を完全に復元できます。<br>
    <em>2·f<sub>max</sub></em> を <strong>ナイキストレート</strong> と呼びます。
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### パラメータ設定")
        f0 = st.slider("信号周波数 f₀ [Hz]", 1.0, 20.0, 5.0, 0.5, key="samp_f0")
        fs = st.slider("サンプリング周波数 fₛ [Hz]", 2.0, 80.0, 30.0, 1.0, key="samp_fs")

        nyquist_rate = 2 * f0
        is_ok = fs >= nyquist_rate

        if is_ok:
            st.success(f"✅ ナイキスト条件OK  (fₛ = {fs} ≥ 2f₀ = {nyquist_rate})")
        else:
            st.error(f"❌ エイリアシング発生！ (fₛ = {fs} < 2f₀ = {nyquist_rate})")

        show_reconstruction = st.checkbox("sinc 補間による再構成を表示", value=True, key="samp_recon")

    with col2:
        T_total = 1.0
        t_cont = np.linspace(0, T_total, 5000)
        x_cont = np.sin(2 * np.pi * f0 * t_cont)

        n_samples = max(int(T_total * fs), 2)
        t_sampled = np.arange(n_samples) / fs
        t_sampled = t_sampled[t_sampled <= T_total]
        x_sampled = np.sin(2 * np.pi * f0 * t_sampled)

        fig = make_subplots(rows=2, cols=1, subplot_titles=["時間領域", "周波数スペクトル"],
                            vertical_spacing=0.15)

        # 時間領域
        fig.add_trace(go.Scatter(x=t_cont, y=x_cont, mode='lines',
                                 name='元の連続信号', line=dict(color='#1E88E5', width=2)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=t_sampled, y=x_sampled, mode='markers',
                                 name='サンプル点',
                                 marker=dict(color='#E53935' if not is_ok else '#43A047',
                                             size=8, symbol='circle')),
                      row=1, col=1)

        if show_reconstruction and len(t_sampled) > 1:
            Ts = 1.0 / fs
            t_recon = np.linspace(0.05, T_total - 0.05, 1000)
            x_recon = sinc_reconstruction(t_sampled, x_sampled, t_recon, Ts)
            fig.add_trace(go.Scatter(x=t_recon, y=x_recon, mode='lines',
                                     name='sinc 補間で再構成',
                                     line=dict(color='#FF9800', width=2, dash='dash')),
                          row=1, col=1)

        # 周波数スペクトル
        if len(x_sampled) > 1:
            X = np.fft.fft(x_sampled)
            freqs = np.fft.fftfreq(len(x_sampled), d=1.0 / fs)
            pos_mask = freqs >= 0
            magnitude = np.abs(X[pos_mask]) / len(x_sampled) * 2

            fig.add_trace(go.Bar(x=freqs[pos_mask], y=magnitude,
                                 name='スペクトル',
                                 marker_color='#7E57C2', opacity=0.7),
                          row=2, col=1)

            fig.add_vline(x=f0, line_dash="dash", line_color="green",
                          annotation_text=f"f₀={f0}Hz", row=2, col=1)
            fig.add_vline(x=fs / 2, line_dash="dot", line_color="red",
                          annotation_text=f"fₛ/2={fs / 2}Hz", row=2, col=1)

        fig.update_layout(height=600, template='plotly_white',
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        fig.update_xaxes(title_text="時間 t [秒]", row=1)
        fig.update_xaxes(title_text="周波数 [Hz]", row=2)
        fig.update_yaxes(title_text="振幅", row=1)
        fig.update_yaxes(title_text="振幅", row=2)

        st.plotly_chart(fig, width='stretch')

    st.markdown("""
    <div class="insight-box">
    <strong>💡 ポイント</strong><br>
    • <strong>fₛ ≥ 2f₀</strong> のとき：スペクトルが折り返さず、sinc 補間で完全復元可能<br>
    • <strong>fₛ < 2f₀</strong> のとき：周波数が折り返し（エイリアシング）、復元不可能<br>
    • スライダーでサンプリング周波数を <strong>ナイキストレート (2f₀) を跨いで</strong> 変化させてみてください！
    </div>
    """, unsafe_allow_html=True)


# ====================================================================
#  タブ 2: パルス整形 & アイパターン
# ====================================================================

with tab2:
    st.markdown("## 🎛️ パルス整形とアイパターン")

    st.markdown("""
    <div class="concept-box">
    <strong>📚 ナイキストの第1条件（ISI フリー条件）</strong><br>
    パルス p(t) が <em>p(nT) = δ[n]</em> を満たすとき、
    サンプリング時点で符号間干渉（ISI）が発生しません。<br>
    <strong>Raised Cosine (RC) フィルタ</strong> はこの条件を満たす代表的なフィルタです。
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### パラメータ設定")
        alpha = st.slider("ロールオフ率 α", 0.0, 1.0, 0.35, 0.05, key="ps_alpha")
        sps = st.slider("シンボルあたりサンプル数", 8, 64, 32, 8, key="ps_sps")
        num_sym_eye = st.slider("アイパターンの重ね書き数", 20, 200, 100, 10, key="ps_eye_n")

        st.markdown(f"""
        **α = {alpha:.2f} の特性:**
        - 帯域幅: **{1 / 2 * (1 + alpha):.2f} / T** (正規化)
        - 帯域効率: **{1 / (1 + alpha):.2f}** symbol/s/Hz
        """)

    with col2:
        # ── サブプロット: インパルス応答 / 周波数特性 / アイパターン ──
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=[
                                "インパルス応答 (時間領域)",
                                "周波数特性",
                                f"アイパターン (α={alpha})"
                            ],
                            horizontal_spacing=0.08)

        t_pulse = np.linspace(-6, 6, 1000)
        f_axis = np.linspace(-1.5, 1.5, 1000)

        # (1) 複数の α のインパルス応答を重ね描き + 選択した α を強調
        for a_ref in [0.0, 0.25, 0.5, 0.75, 1.0]:
            p = raised_cosine(t_pulse, 1.0, a_ref)
            is_selected = abs(a_ref - alpha) < 0.01
            fig.add_trace(go.Scatter(
                x=t_pulse, y=p, mode='lines',
                name=f'α={a_ref}',
                line=dict(width=3 if is_selected else 1,
                          dash='solid' if is_selected else 'dot'),
                opacity=1.0 if is_selected else 0.4,
            ), row=1, col=1)

        # ナイキスト条件の確認点
        for n in range(-5, 6):
            fig.add_trace(go.Scatter(x=[n], y=[1 if n == 0 else 0], mode='markers',
                                     marker=dict(size=6, color='black', symbol='x'),
                                     showlegend=False), row=1, col=1)

        # (2) 周波数特性
        H_selected = rc_spectrum(f_axis, 1.0, alpha)
        fig.add_trace(go.Scatter(x=f_axis, y=H_selected, mode='lines',
                                 name=f'|H(f)| α={alpha}',
                                 line=dict(color='#E53935', width=2),
                                 showlegend=False),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=f_axis, y=rc_spectrum(f_axis, 1.0, 0.0), mode='lines',
                                 name='α=0 (理想)',
                                 line=dict(color='gray', width=1, dash='dash'),
                                 showlegend=False),
                      row=1, col=2)

        # (3) アイパターン
        np.random.seed(0)
        symbols_eye = 2 * np.random.randint(0, 2, num_sym_eye + 20) - 1

        # パルス整形
        span = 6
        t_filt = np.arange(-span * sps, span * sps + 1) / sps
        pulse = raised_cosine(t_filt, 1.0, alpha)
        pulse = pulse / np.max(np.abs(pulse))

        upsampled = np.zeros(len(symbols_eye) * sps)
        upsampled[::sps] = symbols_eye
        shaped = np.convolve(upsampled, pulse, mode='same')

        t_eye_norm = np.linspace(0, 2, 2 * sps)
        for n in range(10, num_sym_eye):
            start = n * sps
            end = start + 2 * sps
            if end <= len(shaped):
                fig.add_trace(go.Scatter(
                    x=t_eye_norm, y=shaped[start:end], mode='lines',
                    line=dict(color='rgba(30, 136, 229, 0.15)', width=1),
                    showlegend=False
                ), row=1, col=3)

        fig.add_vline(x=1.0, line_dash="dash", line_color="red", row=1, col=3)

        fig.update_layout(height=450, template='plotly_white',
                          legend=dict(orientation="h", yanchor="bottom", y=1.08, font_size=10))
        fig.update_xaxes(title_text="t / T", row=1, col=1)
        fig.update_xaxes(title_text="f·T", row=1, col=2)
        fig.update_xaxes(title_text="t / T", row=1, col=3)
        fig.update_yaxes(title_text="p(t)", row=1, col=1)
        fig.update_yaxes(title_text="|H(f)|", row=1, col=2)
        fig.update_yaxes(title_text="振幅", row=1, col=3)

        st.plotly_chart(fig, width='stretch')

    st.markdown("""
    <div class="insight-box">
    <strong>💡 ポイント</strong><br>
    • <strong>α = 0</strong>：最も帯域効率が良いが、パルスの減衰が遅くタイミング誤差に弱い<br>
    • <strong>α = 1</strong>：アイの開口が最大でタイミングに強いが、帯域幅が2倍必要<br>
    • アイパターンの「目」が大きいほど、ノイズやタイミング誤差への <strong>マージン</strong> が大きい<br>
    • 判定時点（赤い点線 t=T）でのアイの開きが <strong>ノイズマージン</strong> を決定する
    </div>
    """, unsafe_allow_html=True)


# ====================================================================
#  タブ 3: AWGN & 変調方式
# ====================================================================

with tab3:
    st.markdown("## 📊 AWGN チャネルと変調方式")

    st.markdown("""
    <div class="concept-box">
    <strong>📚 AWGN (Additive White Gaussian Noise)</strong><br>
    受信信号 <em>r = s + n</em>　（<strong>加法性</strong>: 信号にノイズが足し算で加わる）<br>
    ノイズ <em>n</em> は全周波数に均等（<strong>白色</strong>）で、振幅がガウス分布（<strong>ガウス性</strong>）に従います。
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### パラメータ設定")
        mod_type = st.selectbox("変調方式", ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM'],
                                index=1, key="awgn_mod")
        EbN0_dB = st.slider("Eb/N₀ [dB]", -2.0, 20.0, 10.0, 0.5, key="awgn_ebn0")
        num_symbols = st.slider("シミュレーションシンボル数", 500, 5000, 2000, 500,
                                key="awgn_nsym")

        constellation, bps = get_constellation(mod_type)
        st.markdown(f"""
        **{mod_type} の特性:**
        - シンボル数: **{len(constellation)}**
        - ビット/シンボル: **{bps}**
        - 理論 BER: **{theoretical_ber(EbN0_dB, mod_type):.2e}**
        """)

    with col2:
        np.random.seed(42)

        # AWGN チャネルシミュレーション
        indices = np.random.randint(0, len(constellation), num_symbols)
        tx_symbols = constellation[indices]

        EbN0_lin = 10 ** (EbN0_dB / 10)
        EsN0_lin = bps * EbN0_lin
        noise_std = np.sqrt(1 / (2 * EsN0_lin))
        noise = noise_std * (np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols))
        rx_symbols = tx_symbols + noise

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[
                                f"コンスタレーション ({mod_type}, Eb/N₀={EbN0_dB} dB)",
                                "BER カーブ"
                            ],
                            horizontal_spacing=0.12)

        # (1) コンスタレーション
        fig.add_trace(go.Scatter(
            x=rx_symbols.real, y=rx_symbols.imag, mode='markers',
            name='受信シンボル',
            marker=dict(color='rgba(229, 57, 53, 0.3)', size=3),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=constellation.real, y=constellation.imag, mode='markers',
            name='理想信号点',
            marker=dict(color='#1E88E5', size=10, symbol='diamond',
                        line=dict(width=1, color='black')),
        ), row=1, col=1)

        lim = max(np.abs(constellation.real).max(), np.abs(constellation.imag).max()) * 2.5
        fig.update_xaxes(range=[-lim, lim], title_text="I (同相)", row=1, col=1)
        fig.update_yaxes(range=[-lim, lim], title_text="Q (直交)", row=1, col=1,
                         scaleanchor="x", scaleratio=1)

        # (2) BER カーブ
        EbN0_range = np.arange(0, 16, 0.5)
        colors_ber = {'BPSK': '#1E88E5', 'QPSK': '#FF9800', '8PSK': '#43A047',
                      '16QAM': '#E53935', '64QAM': '#7E57C2'}

        for mod_name in ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM']:
            ber_th = theoretical_ber(EbN0_range, mod_name)
            ber_th = np.maximum(ber_th, 1e-8)
            is_selected = mod_name == mod_type
            fig.add_trace(go.Scatter(
                x=EbN0_range, y=ber_th, mode='lines',
                name=mod_name,
                line=dict(color=colors_ber[mod_name],
                          width=3 if is_selected else 1,
                          dash='solid' if is_selected else 'dot'),
                opacity=1.0 if is_selected else 0.5,
            ), row=1, col=2)

        # 現在の動作点
        current_ber = theoretical_ber(EbN0_dB, mod_type)
        fig.add_trace(go.Scatter(
            x=[EbN0_dB], y=[max(current_ber, 1e-8)], mode='markers',
            name=f'現在の動作点 (BER={current_ber:.2e})',
            marker=dict(color='red', size=14, symbol='star', line=dict(width=2, color='black')),
        ), row=1, col=2)

        fig.update_yaxes(type="log", range=[-6, 0], title_text="BER", row=1, col=2)
        fig.update_xaxes(title_text="Eb/N₀ [dB]", row=1, col=2)

        fig.update_layout(height=500, template='plotly_white',
                          legend=dict(orientation="h", yanchor="bottom", y=1.08, font_size=10))

        st.plotly_chart(fig, width='stretch')

    # ── 連続領域の波形表示 ──
    st.markdown("### 連続領域での AWGN の影響")

    t_wave = np.linspace(0, 1, 1000)
    signal_wave = np.sin(2 * np.pi * 5 * t_wave)
    P_sig = np.mean(signal_wave ** 2)
    snr_lin_wave = 10 ** (EbN0_dB / 10)
    sigma_wave = np.sqrt(P_sig / snr_lin_wave)
    noise_wave = np.random.normal(0, sigma_wave, len(t_wave))
    rx_wave = signal_wave + noise_wave

    fig_wave = go.Figure()
    fig_wave.add_trace(go.Scatter(x=t_wave, y=signal_wave, mode='lines',
                                   name='送信信号 s(t)',
                                   line=dict(color='#1E88E5', width=2)))
    fig_wave.add_trace(go.Scatter(x=t_wave, y=rx_wave, mode='lines',
                                   name='受信信号 r(t) = s(t) + n(t)',
                                   line=dict(color='#E53935', width=1),
                                   opacity=0.7))
    fig_wave.update_layout(height=300, template='plotly_white',
                            xaxis_title="時間 [秒]", yaxis_title="振幅",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_wave, width='stretch')

    # ── ノイズのヒストグラム ──
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        fig_hist = go.Figure()
        noise_only = rx_wave - signal_wave
        fig_hist.add_trace(go.Histogram(x=noise_only, nbinsx=60, name='ノイズ分布',
                                         marker_color='rgba(126, 87, 194, 0.7)',
                                         histnorm='probability density'))
        x_g = np.linspace(-4 * sigma_wave, 4 * sigma_wave, 200)
        pdf_g = (1 / (sigma_wave * np.sqrt(2 * np.pi))) * np.exp(-x_g ** 2 / (2 * sigma_wave ** 2))
        fig_hist.add_trace(go.Scatter(x=x_g, y=pdf_g, mode='lines',
                                       name=f'N(0, {sigma_wave:.3f}²)',
                                       line=dict(color='red', width=2)))
        fig_hist.update_layout(height=300, template='plotly_white',
                                title="ノイズのヒストグラム vs ガウス分布",
                                xaxis_title="振幅", yaxis_title="確率密度")
        st.plotly_chart(fig_hist, width='stretch')

    with col_h2:
        st.markdown("""
        <div class="insight-box">
        <strong>💡 AWGN の3つの性質</strong><br><br>
        <strong>A (Additive)</strong>: r(t) = s(t) + n(t)　信号にノイズが「加算」される<br><br>
        <strong>W (White)</strong>: パワースペクトル密度が全周波数で一定<br>
        → 自己相関関数 = δ(τ)<br><br>
        <strong>G (Gaussian)</strong>: 振幅が正規分布 N(0, σ²) に従う<br>
        → 中心極限定理により、多数の独立な微小雑音源の合計<br><br>
        スライダーで <strong>Eb/N₀</strong> を変化させ、波形とコンスタレーションが
        どう劣化するか確認しましょう！
        </div>
        """, unsafe_allow_html=True)


# ── サイドバーの情報 ──
with st.sidebar:
    st.markdown("## 📖 学習ガイド")
    st.markdown("""
    **推奨学習順序:**
    1. 🔬 サンプリング定理
    2. 🎛️ パルス整形
    3. 📊 AWGN & 変調

    ---

    **操作のヒント:**
    - スライダーを動かしてパラメータを変化させてください
    - グラフはズーム・パン操作が可能です
    - コンスタレーションでは変調方式を切り替えて比較できます

    ---

    **数式リファレンス:**

    **サンプリング定理:**
    fₛ ≥ 2·f_max

    **Raised Cosine:**
    p(t) = sinc(t/T) · cos(παt/T) / (1-(2αt/T)²)

    **BER (BPSK/QPSK):**
    BER = ½ erfc(√(Eb/N₀))

    **Shannon容量:**
    C = W·log₂(1 + SNR)
    """)
