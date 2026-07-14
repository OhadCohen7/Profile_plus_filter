"""
ACS Motion Tools — Streamlit App
Tabs:
  1. 3rd-Order Motion Profile Generator (Ruckig)
  2. BiQuad Filter Designer
Run: streamlit run acs_motion_tools.py
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ACS Motion Tools",
    page_icon="⚙️",
    layout="wide",
)

st.markdown("""
<style>
    .flag-banner {
        background: #1a2240;
        border-left: 3px solid #5b8af5;
        border-radius: 0 6px 6px 0;
        padding: 8px 14px;
        font-size: 13px;
        color: #9ab4f5;
        font-family: monospace;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚙️ ACS Motion Tools")

tab_profile, tab_filter = st.tabs(["📈 Motion Profile Generator", "🎛️ BiQuad Filter Designer"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Motion Profile Generator
# ══════════════════════════════════════════════════════════════════════════════
with tab_profile:

    # ── imports (guarded so missing ruckig doesn't break the filter tab) ──────
    try:
        from ruckig import InputParameter, OutputParameter, Result, Ruckig, Trajectory
        ruckig_ok = True
    except ImportError:
        ruckig_ok = True

    if not ruckig_ok:
        st.error(
            "The `ruckig` package is not installed. "
            "Add it to your `requirements.txt` and redeploy, or run `pip install ruckig`."
        )
    else:
        # ── Axis class ────────────────────────────────────────────────────────
        class Axis:
            def __init__(self, name, vel, acc, dec, jerk, units):
                self.name = name
                self.vel = vel
                self.acc = acc
                self.dec = dec
                self.jerk = jerk
                self.units = units

            def time_to_perform(self, pos0, pos1, settling_time=0):
                inp = InputParameter(1)
                out = OutputParameter(1)
                inp.current_position = [pos0]
                inp.current_velocity = [0]
                inp.current_acceleration = [0]
                inp.target_position = [pos1]
                inp.target_velocity = [0]
                inp.target_acceleration = [0]
                inp.max_velocity = [self.vel]
                inp.max_acceleration = [self.acc]
                inp.max_jerk = [self.jerk]
                otg = Ruckig(1)
                trajectory = Trajectory(1)
                result = otg.calculate(inp, trajectory)
                if result == Result.ErrorInvalidInput:
                    raise Exception("Invalid input!")
                return trajectory.duration + settling_time

            def online_trajectory(self, pos0, pos1):
                position, speed, acc, time = [], [], [], []
                inp = InputParameter(1)
                out = OutputParameter(1)
                inp.current_position = [pos0]
                inp.current_velocity = [0]
                inp.current_acceleration = [0]
                inp.target_position = [pos1]
                inp.target_velocity = [0]
                inp.target_acceleration = [0]
                inp.max_velocity = [self.vel]
                inp.max_acceleration = [self.acc]
                inp.max_jerk = [self.jerk]
                otg = Ruckig(1, 0.001)
                res = Result.Working
                while res == Result.Working:
                    res = otg.update(inp, out)
                    time.append(out.time)
                    position.append(out.new_position)
                    speed.append(out.new_velocity)
                    acc.append(out.new_acceleration)
                    out.pass_to_input(inp)
                return time, position, speed, acc

        # ── helpers ───────────────────────────────────────────────────────────
        def generate_profile(start, finish, vel, acc, jerk):
            axis = Axis("axis", vel, acc, acc, jerk, "uu")
            t_total = axis.time_to_perform(start, finish)
            time, position, speed, accel = axis.online_trajectory(start, finish)

            position = [p[0] for p in position]
            speed    = [s[0] for s in speed]
            accel    = [a[0] for a in accel]
            acc_rms  = float(np.sqrt(np.mean(np.square(accel))))

            def make_fig(x, y, title, ytitle, hover_y):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode="lines",
                    hovertemplate=f"Time: %{{x:.3f}} s<br>{hover_y}: %{{y:.3f}}",
                ))
                fig.update_layout(
                    title=title,
                    xaxis_title="Time [sec]",
                    yaxis_title=ytitle,
                    hovermode="x unified",
                    margin=dict(t=40, b=30),
                )
                return fig

            fig_pos = make_fig(time, position, "Position",     "(User Units)",        "Position")
            fig_vel = make_fig(time, speed,    "Velocity",     "(User Units)/sec",    "Velocity")
            fig_acc = make_fig(time, accel,    "Acceleration", "(User Units)/sec²",   "Acceleration")

            return (fig_pos, fig_vel, fig_acc), t_total, acc_rms

        # ── presets ───────────────────────────────────────────────────────────
        PRESETS = {
            "IM X axis":     {"speed": 1200.0, "acc": 24000.0,  "jerk": 600000.0},
            "IM Y axis":     {"speed": 1200.0, "acc": 12000.0,  "jerk": 185000.0},
            "IM Theta axis": {"speed":   22.0, "acc":   140.0,  "jerk":   1500.0},
            "IM Polarizer":  {"speed":   53.0, "acc":  3400.0,  "jerk": 235000.0},
        }

        # ── layout ────────────────────────────────────────────────────────────
        ctrl_col, plot_col = st.columns([1, 2.5], gap="large")

        with ctrl_col:
            st.subheader("Parameters")

            axis_choice = st.selectbox("Predefined kinematics", list(PRESETS.keys()))
            preset = PRESETS[axis_choice]

            start  = st.number_input("Start position (UU)",    value=0.0,                step=10.0, format="%.3f")
            finish = st.number_input("End position (UU)",      value=50.0,               step=10.0, format="%.3f")
            vel    = st.number_input("Speed (UU/s)",           value=preset["speed"],    step=10.0, format="%.3f")
            acc    = st.number_input("Acc & Dec (UU/s²)",      value=preset["acc"],      step=10.0, format="%.3f")
            jerk   = st.number_input("Jerk (UU/s³)",          value=preset["jerk"],     step=10.0, format="%.3f")

            run_profile = st.button("Generate profile", use_container_width=True, type="primary")

        with plot_col:
            if run_profile:
                try:
                    with st.spinner("Computing trajectory…"):
                        figs, t_total, acc_rms = generate_profile(start, finish, vel, acc, jerk)

                    st.success(f"**Time to perform:** `{t_total:.4f}` s")
                    st.info(f"**RMS Acceleration** (Proportional to RMS current): `{acc_rms:.3f}` UU/s²")

                    st.plotly_chart(figs[0], use_container_width=True)
                    st.plotly_chart(figs[1], use_container_width=True)
                    st.plotly_chart(figs[2], use_container_width=True)

                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.info("Set the parameters on the left and click **Generate profile**.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BiQuad Filter Designer
# ══════════════════════════════════════════════════════════════════════════════
with tab_filter:

    # ── core math ─────────────────────────────────────────────────────────────
    def calculate_biquad_params(filter_type, f_hz, width_hz, atten_db):
      """
      Calculates SLVB0 NF, DF, ND, DD based on filter requirements.
      atten_db: Positive value for attenuation (reduction) or gain (boost) in dB.
      """
      filter_type = filter_type.lower()
      
      # Convert dB to absolute units: Absolute = 10^(dB/20)
      # Reference: Source 5 (Slide 9), Source 6 (Slide 14)
      atten_abs = 10**(atten_db / 20.0)
      
      if filter_type == 'notch':
          # Notch Formula: Source 16, page 342
          nf = df = f_hz
          nd = width_hz / (2 * f_hz)  # Numerator Damping
          dd = nd * atten_abs         # Denominator Damping (must be > ND for a dip)
          
      elif filter_type in ['anti-notch', 'band-pass']:
          # Inverse of Notch logic to create a peak: Source 16, page 342
          nf = df = f_hz
          dd = width_hz / (2 * f_hz)
          nd = dd * atten_abs         # Numerator Damping (must be > DD for a peak)
          
      elif filter_type == 'lpf':
          # 2nd Order Lag: Source 16, page 344
          # Low frequency gain = (NF/DF)^2. So NF = DF * sqrt(Gain_abs)
          df = f_hz
          nf = df * np.sqrt(atten_abs)
          nd = dd = 0.707
      else:
          raise ValueError("Unknown filter type. Use 'notch', 'anti-notch', or 'lpf'.")
  
      # Clamp values to valid ACS ranges: 0.1-4000Hz, 0.01-1.0 damping
      # Reference: Source 16, page 341
      nf, df = np.clip([nf, df], 0.1, 4000)
      nd, dd = np.clip([nd, dd], 0.01, 1.0)
      return nf, df, nd, dd
  
    def compute_bode(nf: float, df: float, nd: float, dd: float):
        omega_n = 2 * np.pi * nf
        omega_d = 2 * np.pi * df
        num = [1, 2 * nd * omega_n, omega_n ** 2]
        den = [1, 2 * dd * omega_d, omega_d ** 2]
        sys = signal.TransferFunction(num, den)
        w_hz = np.logspace(0, np.log10(4000), 600)
        _, mag, phase = signal.bode(sys, 2 * np.pi * w_hz)
        return w_hz, mag, phase

    def make_bode_figure(freqs, mag, phase, title: str) -> go.Figure:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
            subplot_titles=("Magnitude (dB)", "Phase (°)"),
        )
        fig.add_trace(go.Scatter(
            x=freqs, y=mag, mode="lines", name="Magnitude",
            line=dict(color="#5b8af5", width=2),
            fill="tozeroy", fillcolor="rgba(91,138,245,0.07)",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=freqs, y=phase, mode="lines", name="Phase",
            line=dict(color="#3dd68c", width=2),
        ), row=2, col=1)

        log_x = dict(
            type="log", range=[0, np.log10(4000)],
            tickvals=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000],
            ticktext=["1", "2", "5", "10", "20", "50", "100", "200", "500", "1k", "2k", "4k"],
            showgrid=True, gridcolor="rgba(128,128,128,0.2)",
            title_text="Frequency (Hz)",
        )
        fig.update_xaxes(log_x, row=1, col=1)
        fig.update_xaxes(log_x, row=2, col=1)
        fig.update_yaxes(title_text="Magnitude (dB)", showgrid=True, gridcolor="rgba(128,128,128,0.2)", row=1, col=1)
        fig.update_yaxes(title_text="Phase (°)",      showgrid=True, gridcolor="rgba(128,128,128,0.2)", row=2, col=1)
        fig.update_layout(
            title=dict(text=title, font=dict(size=13), x=0.0, xanchor="left"),
            showlegend=False, height=500,
            margin=dict(l=10, r=10, t=50, b=10),
            hovermode="x unified",
        )
        return fig

    # ── layout ────────────────────────────────────────────────────────────────
    ctrl_col2, plot_col2 = st.columns([1, 2.5], gap="large")

    with ctrl_col2:
        st.subheader("Filter configuration")

        filter_type = st.selectbox(
            "Filter type",
            options=["notch", "anti-notch", "lpf"],
            format_func=lambda x: {
                "notch":     "Notch",
                "anti-notch": "Anti-notch / Band-pass",
                "lpf":       "LPF (2nd-order lag)",
            }[x],
            key="filter_type_sel",
        )

        freq = st.number_input(
            "Center / cutoff frequency (Hz)",
            min_value=0.1, max_value=4000.0, value=100.0, step=1.0, format="%.1f",
        )

        if filter_type != "lpf":
            width = st.number_input(
                "Filter width (Hz)",
                min_value=0.1, max_value=4000.0, value=20.0, step=1.0, format="%.1f",
            )
        else:
            width = 0.0
            st.caption("Width is not used for LPF.")

        atten_label = "Gain (absolute)" if filter_type == "lpf" else "Attenuation (absolute)"
        atten = st.number_input(atten_label, min_value=0.01, value=5.0, step=0.1, format="%.2f")

        run_filter = st.button("Compute & plot", use_container_width=True, type="primary", key="run_filter")

    with plot_col2:
        if run_filter:
            try:
                nf, df, nd, dd = calculate_biquad_params(filter_type, freq, width, atten)
                freqs, mag, phase = compute_bode(nf, df, nd, dd)

                bode_title = (
                    f"Bode Plot — {filter_type.upper()}  |  "
                    f"NF: {nf:.1f} Hz   DF: {df:.1f} Hz   ND: {nd:.3f}   DD: {dd:.3f}"
                )
                st.plotly_chart(make_bode_figure(freqs, mag, phase, bode_title), use_container_width=True)

                st.markdown("**Calculated SLVB0 parameters**")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("SLVB0NF(0)", f"{nf:.2f}")
                c2.metric("SLVB0DF(0)", f"{df:.2f}")
                c3.metric("SLVB0ND(0)", f"{nd:.4f}")
                c4.metric("SLVB0DD(0)", f"{dd:.4f}")

                st.markdown(
                    '<div class="flag-banner">Set <b>MFLAGS(0).16 = 1</b> to enable this filter.</div>',
                    unsafe_allow_html=True,
                )

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("Configure the filter on the left and click **Compute & plot**.")
