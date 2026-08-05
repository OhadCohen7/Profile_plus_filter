"""
Motion Tools — Streamlit App
Tabs:
  1. 3rd-Order Motion Profile Generator (pure-Python, no ruckig dependency)
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
    page_title="Motion Tools",
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

st.title("⚙️ Motion Tools")

tab_profile, tab_filter = st.tabs(["📈 Motion Profile Generator", "🎛️ BiQuad Filter Designer"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Motion Profile Generator (pure-Python 7-phase S-curve, no ruckig)
# ══════════════════════════════════════════════════════════════════════════════
with tab_profile:

    def jerk_limited_profile(pos0, pos1, v_max, a_max, j_max, dt=0.001):
        """
        Pure-Python 7-phase jerk-limited (S-curve) trajectory generator.
        Equivalent to Ruckig for single-DOF point-to-point moves.
        Returns (time, position, velocity, acceleration) where position/velocity/
        acceleration entries are [value] lists, matching the original Ruckig format.
        """
        dist = pos1 - pos0
        if abs(dist) < 1e-12:
            return [0.0], [[pos0]], [[0.0]], [[0.0]]
        sign = 1 if dist > 0 else -1
        d = abs(dist)

        def calc_block(v_lim):
            """Timing and distance for one accel (or decel) block at peak velocity v_lim."""
            t_j = min(a_max / j_max, np.sqrt(v_lim / j_max))
            a_peak = j_max * t_j
            v1 = 0.5 * j_max * t_j ** 2           # velocity at end of jerk-up phase
            t_a = max(0.0, (v_lim - 2 * v1) / a_peak)  # constant-accel duration
            d1 = j_max * t_j ** 3 / 6
            d2 = v1 * t_a + 0.5 * a_peak * t_a ** 2
            v2 = v1 + a_peak * t_a
            d3 = v2 * t_j + 0.5 * a_peak * t_j ** 2 - j_max * t_j ** 3 / 6
            return t_j, a_peak, t_a, d1 + d2 + d3

        t_j, a_peak, t_a, d_acc = calc_block(v_max)

        if 2 * d_acc > d:
            # Move too short to reach v_max — binary-search for achievable peak velocity
            lo, hi = 0.0, v_max
            for _ in range(80):
                mid = (lo + hi) / 2
                _, _, _, da = calc_block(mid)
                if 2 * da < d:
                    lo = mid
                else:
                    hi = mid
            v_lim = (lo + hi) / 2
            t_j, a_peak, t_a, d_acc = calc_block(v_lim)
            t_cruise = 0.0
        else:
            v_lim = v_max
            t_cruise = (d - 2 * d_acc) / v_lim

        # 7 phases: (duration, jerk_value)
        phases = [
            (t_j,      +j_max),   # 1 jerk up
            (t_a,       0.0),     # 2 constant accel
            (t_j,      -j_max),   # 3 jerk down  → peak velocity reached
            (t_cruise,  0.0),     # 4 cruise
            (t_j,      -j_max),   # 5 jerk down  → start decel
            (t_a,       0.0),     # 6 constant decel
            (t_j,      +j_max),   # 7 jerk up    → rest
        ]

        time_arr, pos_arr, vel_arr, acc_arr = [], [], [], []
        t, p, v, a = 0.0, float(pos0), 0.0, 0.0

        for dur, jerk in phases:
            if dur < 1e-9:
                continue
            n = max(1, int(round(dur / dt)))
            dt_i = dur / n
            for _ in range(n):
                time_arr.append(t)
                pos_arr.append([p])
                vel_arr.append([v * sign])
                acc_arr.append([a * sign])
                # Exact 3rd-order integration within step
                p += sign * (v * dt_i + 0.5 * a * dt_i ** 2 + (1 / 6) * jerk * dt_i ** 3)
                v += a * dt_i + 0.5 * jerk * dt_i ** 2
                a += jerk * dt_i
                t += dt_i

        # Final point
        time_arr.append(t)
        pos_arr.append([pos1])
        vel_arr.append([0.0])
        acc_arr.append([0.0])
        return time_arr, pos_arr, vel_arr, acc_arr

    def generate_profile(start, finish, vel, acc, jerk, mass_kg, kt, unit_scale, friction_n,
                         duty_cycle_pct, motor_type, ke, resistance, bus_voltage):
        time, position, speed, accel = jerk_limited_profile(start, finish, vel, acc, jerk)
        t_total = time[-1]

        position = [p[0] for p in position]
        speed    = [s[0] for s in speed]
        accel    = [a[0] for a in accel]
        acc_rms  = float(np.sqrt(np.mean(np.square(accel))))

        # ── Current ──────────────────────────────────────────────────────────
        # F = m*a (in SI) + friction opposing motion direction;  I = F / Kt
        accel_si       = [a * unit_scale for a in accel]
        friction_force = [friction_n * np.sign(v) for v in speed]
        force          = [mass_kg * a + f for a, f in zip(accel_si, friction_force)]
        current        = [f / kt for f in force]

        current_rms_profile = float(np.sqrt(np.mean(np.square(current))))
        current_peak        = float(np.max(np.abs(current)))

        # Duty-cycle-weighted RMS:  I_rms_total = I_rms_profile * sqrt(duty)
        duty = max(0.0, min(1.0, duty_cycle_pct / 100.0))
        current_rms_total = current_rms_profile * duty

        # ── Back EMF ─────────────────────────────────────────────────────────
        # Convert speed to SI (m/s or rad/s) then apply Ke
        if motor_type == "Linear (V/(m/s))":
            # speed is in UU/s; unit_scale converts UU → m
            speed_si = [s * unit_scale for s in speed]          # m/s
            bemf     = [ke * abs(s) for s in speed_si]          # V
        else:
            # Rotary: speed is in RPM (UU == RPM assumed)
            # Ke unit: V/RPM  →  BEMF = Ke * |RPM|
            bemf = [ke * abs(s) for s in speed]                  # V

        bemf_peak = float(np.max(bemf))

        # ── Bus sufficiency ───────────────────────────────────────────────────
        # Required voltage at peak: V_required = BEMF_peak + I_peak * R
        ir_peak      = current_peak * resistance
        v_required   = bemf_peak + ir_peak
        bus_ok       = (bus_voltage is not None) and (bus_voltage >= v_required)
        bus_margin   = (bus_voltage - v_required) if bus_voltage is not None else None

        # ── Voltage trace (instantaneous) ─────────────────────────────────────
        if bus_voltage is not None:
            voltage = [b + abs(i) * resistance for b, i in zip(bemf, current)]
        else:
            voltage = None

        # ── Figures ───────────────────────────────────────────────────────────
        def make_fig(x, y, title, ytitle, hover_y, extra_traces=None):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines",
                hovertemplate=f"Time: %{{x:.3f}} s<br>{hover_y}: %{{y:.3f}}",
            ))
            if extra_traces:
                for tr in extra_traces:
                    fig.add_trace(tr)
            fig.update_layout(
                title=title, xaxis_title="Time [sec]", yaxis_title=ytitle,
                hovermode="x unified", margin=dict(t=40, b=30),
            )
            return fig

        fig_pos = make_fig(time, position, "Position",      "(User Units)",      "Position")
        fig_vel = make_fig(time, speed,    "Velocity",      "(User Units)/sec",  "Velocity")
        fig_acc = make_fig(time, accel,    "Acceleration",  "(User Units)/sec²", "Acceleration")
        fig_cur = make_fig(time, current,  "Motor Current", "Amps",              "Current")

        fig_volt = None
        if voltage is not None:
            bus_line = go.Scatter(
                x=[time[0], time[-1]], y=[bus_voltage, bus_voltage],
                mode="lines", name="Bus voltage",
                line=dict(color="red", dash="dash", width=1.5),
                hovertemplate=f"Bus: {bus_voltage:.1f} V",
            )
            fig_volt = make_fig(
                time, voltage,
                "Required Motor Voltage (BEMF + I·R)", "Voltage (V)", "V_motor",
                extra_traces=[bus_line],
            )
            fig_volt.update_layout(showlegend=True)

        results = dict(
            t_total=t_total,
            acc_rms=acc_rms,
            current_rms_profile=current_rms_profile,
            current_rms_total=current_rms_total,
            current_peak=current_peak,
            bemf_peak=bemf_peak,
            ir_peak=ir_peak,
            v_required=v_required,
            bus_ok=bus_ok,
            bus_margin=bus_margin,
        )
        return (fig_pos, fig_vel, fig_acc, fig_cur, fig_volt), results

    # ── Presets ───────────────────────────────────────────────────────────────
    PRESETS = {
        "X axis (IM)":     {"speed": 1200.0, "acc": 24000.0,  "jerk": 600000.0},
        "Y axis (IM)":     {"speed": 1200.0, "acc": 12000.0,  "jerk": 185000.0},
        "Theta axis (IM)": {"speed":   22.0, "acc":   140.0,  "jerk":   1500.0},
        "Pol (IM)":  {"speed":   53.0, "acc":  3400.0,  "jerk": 235000.0},
    }

    # ── Layout ────────────────────────────────────────────────────────────────
    ctrl_col, plot_col = st.columns([1, 2.5], gap="large")

    with ctrl_col:
        st.subheader("Parameters")

        axis_choice = st.selectbox("Predefined kinematics", list(PRESETS.keys()))
        preset = PRESETS[axis_choice]

        start  = st.number_input("Start position (UU)",  value=0.0,             step=10.0, format="%.3f")
        finish = st.number_input("End position (UU)",    value=50.0,            step=10.0, format="%.3f")
        vel    = st.number_input("Speed (UU/s)",         value=preset["speed"], step=10.0, format="%.3f")
        acc    = st.number_input("Acc & Dec (UU/s²)",    value=preset["acc"],   step=10.0, format="%.3f")
        jerk   = st.number_input("Jerk (UU/s³)",        value=preset["jerk"],  step=10.0, format="%.3f")

        st.markdown("---")
        st.caption("Current calculation (optional)")

        position_unit = st.selectbox(
            "What do User Units (UU) represent?",
            ["Millimeters", "Meters"],
            index=0,
            help="Needed to convert acceleration to m/s² for F = m·a.",
        )
        unit_scale = 0.001 if position_unit == "Millimeters" else 1.0

        mass_kg    = st.number_input("Moving mass (kg)",                       value=1.0, min_value=0.0,  step=0.1,  format="%.3f")
        kt         = st.number_input("Force/torque constant Kt (N/A or N·m/A)",value=1.0, min_value=1e-9, step=0.1,  format="%.4f")
        friction_n = st.number_input(
            "Friction force (N)", value=0.0, min_value=0.0, step=0.1, format="%.3f",
            help="Magnitude opposing motion, added to m·a.",
        )
        duty_cycle_pct = st.number_input(
            "Duty cycle (%)", value=100.0, min_value=0.1, max_value=100.0, step=1.0, format="%.1f",
            help="Fraction of time the move is active. "
                 "I_rms_total = I_rms_profile × √(duty). "
                 "100% means the motor is always moving.",
        )

        st.markdown("---")
        st.caption("Bus voltage check (optional)")

        motor_type = st.selectbox(
            "Motor type",
            ["Linear (V/(m/s))", "Rotary (V/RPM)"],
            help="Sets the unit of the Back-EMF constant Ke.",
        )
        ke_label = "Ke — Back-EMF constant (V·s/m)" if motor_type == "Linear (V/(m/s))" \
                   else "Ke — Back-EMF constant (V/RPM)"
        ke         = st.number_input(ke_label,              value=1.0, min_value=0.0, step=0.01, format="%.4f")
        resistance = st.number_input("Winding resistance R (Ω)", value=1.0, min_value=0.0, step=0.1,  format="%.4f")

        enable_bus = st.checkbox("Enter bus voltage", value=False)
        bus_voltage = None
        if enable_bus:
            bus_voltage = st.number_input("DC bus voltage (V)", value=48.0, min_value=0.0, step=1.0, format="%.1f")

        run_profile = st.button("Generate profile", use_container_width=True, type="primary")

    with plot_col:
        if run_profile:
            try:
                with st.spinner("Computing trajectory…"):
                    figs, res = generate_profile(
                        start, finish, vel, acc, jerk,
                        mass_kg, kt, unit_scale, friction_n,
                        duty_cycle_pct, motor_type, ke, resistance, bus_voltage,
                    )

                st.success(f"**Time to perform:** `{res['t_total']:.4f}` s")

                # ── Current metrics ───────────────────────────────────────────
                st.markdown("**Current**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Peak Current",           f"{res['current_peak']:.3f} A")
                c2.metric("RMS Current (profile)",  f"{res['current_rms_profile']:.3f} A",
                          help="RMS over the move duration only.")
                c3.metric(f"RMS Current (duty={duty_cycle_pct:.0f}%)",
                          f"{res['current_rms_total']:.3f} A",
                          help="I_rms_profile × (duty) — Irms over the entire sequence according to the given duty cycle.")

                # ── Bus voltage metrics ───────────────────────────────────────
                if bus_voltage is not None:
                    st.markdown("**Bus voltage check**")
                    v1, v2, v3, v4 = st.columns(4)
                    v1.metric("BEMF peak",      f"{res['bemf_peak']:.2f} V")
                    v2.metric("I·R peak",       f"{res['ir_peak']:.2f} V")
                    v3.metric("V required peak",f"{res['v_required']:.2f} V",
                              help="BEMF_peak + I_peak·R")
                    margin = res['bus_margin']
                    margin_str = f"{margin:+.2f} V"
                    if res['bus_ok']:
                        v4.metric("Bus margin", margin_str, delta=margin_str,
                                  delta_color="normal")
                        st.success(f"✅ Bus sufficient — {margin:.2f} V headroom above peak demand.")
                    else:
                        v4.metric("Bus deficit", margin_str, delta=margin_str,
                                  delta_color="inverse")
                        st.error(f"❌ Bus insufficient — {abs(margin):.2f} V short of peak demand "
                                 f"({res['v_required']:.2f} V required, {bus_voltage:.1f} V available).")

                # ── Plots ─────────────────────────────────────────────────────
                st.plotly_chart(figs[0], use_container_width=True)
                st.plotly_chart(figs[1], use_container_width=True)
                st.plotly_chart(figs[2], use_container_width=True)
                st.plotly_chart(figs[3], use_container_width=True)
                if figs[4] is not None:
                    st.plotly_chart(figs[4], use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("Set the parameters on the left and click **Generate profile**.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BiQuad Filter Designer
# ══════════════════════════════════════════════════════════════════════════════
with tab_filter:

    def calculate_biquad_params(filter_type, f_hz, width_hz, atten_db):
        """
        Calculates SLVB0 NF, DF, ND, DD based on filter requirements.
        atten_db: Positive value for attenuation (reduction) or gain (boost) in dB.
        """
        filter_type = filter_type.lower()

        # Convert dB to absolute units: Absolute = 10^(dB/20)
        atten_abs = 10 ** (atten_db / 20.0)

        if filter_type == 'notch':
            nf = df = f_hz
            nd = width_hz / (2 * f_hz)
            dd = nd * atten_abs

        elif filter_type in ['anti-notch', 'band-pass']:
            nf = df = f_hz
            dd = width_hz / (2 * f_hz)
            nd = dd * atten_abs

        elif filter_type == 'lpf':
            df = f_hz
            nf = df * np.sqrt(atten_abs)
            nd = dd = 0.707

        else:
            raise ValueError("Unknown filter type. Use 'notch', 'anti-notch', or 'lpf'.")

        # Clamp to valid ranges: 0.1–4000 Hz, 0.01–1.0 damping
        nf, df = np.clip([nf, df], 0.1, 4000)
        nd, dd = np.clip([nd, dd], 0.01, 1.0)
        return float(nf), float(df), float(nd), float(dd)

    def compute_bode(nf, df, nd, dd):
        omega_n = 2 * np.pi * nf
        omega_d = 2 * np.pi * df
        num = [1, 2 * nd * omega_n, omega_n ** 2]
        den = [1, 2 * dd * omega_d, omega_d ** 2]
        sys = signal.TransferFunction(num, den)
        w_hz = np.logspace(0, np.log10(4000), 600)
        _, mag, phase = signal.bode(sys, 2 * np.pi * w_hz)
        return w_hz, mag, phase

    def make_bode_figure(freqs, mag, phase, title):
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
            ticktext=["1","2","5","10","20","50","100","200","500","1k","2k","4k"],
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

    # ── Layout ────────────────────────────────────────────────────────────────
    ctrl_col2, plot_col2 = st.columns([1, 2.5], gap="large")

    with ctrl_col2:
        st.subheader("Filter configuration")
        default_value = "manual"
        filter_type = st.selectbox(
            "Filter type",
            options=["notch", "anti-notch", "lpf", "manual"],
            format_func=lambda x: {
                "notch":      "Notch",
                "anti-notch": "Anti-notch / Band-pass",
                "lpf":        "LPF",
                "manual":     "Manual BiQuad (SLVB) parameters",
            }[x],
            index=options.index(default_value),
            key="filter_type_sel",
        )

        if filter_type == "manual":
            # ── Manual BiQuad (SLVB) parameters ────────────────────────────────────────────
            st.caption("Enter the four SLVB0 parameters directly (Valid range shown).")

            st.markdown(r"""
            $$H(s) = \frac{s^2 + 2\zeta_N\,\omega_N s + \omega_N^2}{s^2 + 2\zeta_D\,\omega_D s + \omega_D^2}$$
            """)

            man_nf = st.number_input(
                "SLVB0NF — Numerator Frequency, ωN (Hz)",
                min_value=0.1, max_value=4000.0, value=100.0, step=1.0, format="%.2f",
                help="Valid range: 0.1 – 4000 Hz",
            )
            man_df = st.number_input(
                "SLVB0DF — Denominator Frequency, ωD (Hz)",
                min_value=0.1, max_value=4000.0, value=100.0, step=1.0, format="%.2f",
                help="Valid range: 0.1 – 4000 Hz",
            )
            man_nd = st.number_input(
                "SLVB0ND — Numerator Damping, ζN",
                min_value=0.01, max_value=1.0, value=0.1, step=0.01, format="%.4f",
                help="Valid range: 0.01 – 1.0",
            )
            man_dd = st.number_input(
                "SLVB0DD — Denominator Damping, ζD",
                min_value=0.01, max_value=1.0, value=0.5, step=0.01, format="%.4f",
                help="Valid range: 0.01 – 1.0",
            )

        else:
            # ── Computed modes ────────────────────────────────────────────────
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

            atten_label = "Gain (dB)" if filter_type == "lpf" else "Attenuation (dB)"
            atten = st.number_input(atten_label, min_value=0.01, value=5.0, step=0.1, format="%.2f")

        run_filter = st.button("Compute & plot", use_container_width=True, type="primary", key="run_filter")

    with plot_col2:
        if run_filter:
            try:
                if filter_type == "manual":
                    nf, df, nd, dd = man_nf, man_df, man_nd, man_dd
                    bode_label = "MANUAL"
                else:
                    nf, df, nd, dd = calculate_biquad_params(filter_type, freq, width, atten)
                    bode_label = filter_type.upper()

                freqs, mag, phase = compute_bode(nf, df, nd, dd)

                bode_title = (
                    f"Bode Plot — {bode_label}  |  "
                    f"NF: {nf:.2f} Hz   DF: {df:.2f} Hz   ND: {nd:.4f}   DD: {dd:.4f}"
                )
                st.plotly_chart(make_bode_figure(freqs, mag, phase, bode_title), use_container_width=True)

                st.markdown("**SLVB0 parameters**")
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
