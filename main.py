from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import bisect

app = FastAPI(title="Dynamicspring aA Web API")

# ============================================
# 0. CSV 로드 (서버 시작할 때 한 번만)
# ============================================

df_below = pd.read_csv("below 5.csv")
df_below = df_below.dropna(subset=["L/2ro", "S/2ro", "aA"])

df_above = pd.read_csv("above 5.csv")
df_above = df_above.dropna(subset=["L/2ro", "2ro/S", "aA"])

L_values_below = sorted(df_below["L/2ro"].unique())
L_values_above = sorted(df_above["L/2ro"].unique())

curves_below = {
    L: df_below[df_below["L/2ro"] == L].sort_values("S/2ro")
    for L in L_values_below
}

curves_above = {
    L: df_above[df_above["L/2ro"] == L].sort_values("2ro/S")
    for L in L_values_above
}


# ============================================
# 1. 보간 + 회귀 공통 함수들
# ============================================

def interp_in_x(L, x, curves, x_col):
    """하나의 L/2r0 곡선에서 x 방향(S/2r0 또는 2r0/S)으로 선형보간."""
    sub = curves[L]
    x_vals = sub[x_col].values
    a_vals = sub["aA"].values

    if x <= x_vals[0]:
        return a_vals[0]
    if x >= x_vals[-1]:
        return a_vals[-1]

    return np.interp(x, x_vals, a_vals)


def aA_value_generic(L_target, x_target, L_values, curves, x_col):
    """L 방향 + x 방향 2단계 보간으로 aA 계산."""
    if L_target <= L_values[0]:
        return float(interp_in_x(L_values[0], x_target, curves, x_col))
    if L_target >= L_values[-1]:
        return float(interp_in_x(L_values[-1], x_target, curves, x_col))

    i = bisect.bisect_left(L_values, L_target)
    L1, L2 = L_values[i - 1], L_values[i]

    a1 = interp_in_x(L1, x_target, curves, x_col)
    a2 = interp_in_x(L2, x_target, curves, x_col)

    t = (L_target - L1) / (L2 - L1)
    return float(a1 + (a2 - a1) * t)


def interpolated_curve_generic(L_target, df, L_values, curves, x_col, num_points=200):
    """주어진 L_target에 대해 x 전체 구간에서 aA(L, x) 곡선을 계산."""
    x_min = df[x_col].min()
    x_max = df[x_col].max()
    x_grid = np.linspace(x_min, x_max, num_points)
    a_grid = np.array([
        aA_value_generic(L_target, x, L_values, curves, x_col)
        for x in x_grid
    ])
    return x_grid, a_grid


def fit_poly_with_r2(x, y, max_deg=6, target_r2=0.99):
    """
    (x, y)에 대해 1~max_deg차 다항식 피팅.
    R^2 >= target_r2 만족하는 최소 차수 선택.
    """
    best = None
    for deg in range(1, max_deg + 1):
        coeffs = np.polyfit(x, y, deg)
        p = np.poly1d(coeffs)
        y_pred = p(x)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot

        best = (deg, coeffs, r2)
        if r2 >= target_r2:
            break
    return best  # (deg, coeffs, r2)


def poly_to_string(coeffs, var_name="x"):
    """
    np.polyfit 계수 → 사람이 읽기 쉬운 수식 문자열.
    예: aA ≈ 0.001(S/2r0)^3 - 0.02(S/2r0)^2 + ...
    """
    s = ""
    n = len(coeffs)
    for i, c in enumerate(coeffs):
        power = n - i - 1
        if abs(c) < 1e-12:
            continue

        c_str = f"{c:.6g}"

        if s == "":
            if c < 0:
                sign = "-"
                c_str = c_str.lstrip("-")
            else:
                sign = ""
        else:
            if c < 0:
                sign = " - "
                c_str = c_str.lstrip("-")
            else:
                sign = " + "

        if power == 0:
            term = f"{c_str}"
        elif power == 1:
            term = f"{c_str}{var_name}"
        else:
            term = f"{c_str}{var_name}^{power}"

        s += sign + term

    return "aA ≈ " + s if s else "aA ≈ 0"


def compute_one(L_input: float, S_input: float):
    """
    Dynamicspring_ver2 의 핵심 계산 로직 (그래프 부분 제외).
    """
    if S_input <= 5.0:
        mode = "below"
        x_input = S_input              # x = S/2r0
        x_col = "S/2ro"
        df_use = df_below
        L_values = L_values_below
        curves = curves_below
        x_label = "S/2r0"
    else:
        mode = "above"
        x_input = 1.0 / S_input        # x = 2r0/S = 1/(S/2r0)
        x_col = "2ro/S"
        df_use = df_above
        L_values = L_values_above
        curves = curves_above
        x_label = "2r0/S"

    # 보간 곡선
    x_grid, a_grid = interpolated_curve_generic(
        L_input, df_use, L_values, curves, x_col
    )
    # 점 aA 값
    a_at_point = aA_value_generic(
        L_input, x_input, L_values, curves, x_col
    )

    # 회귀
    deg, coeffs, r2 = fit_poly_with_r2(x_grid, a_grid, max_deg=6, target_r2=0.99)
    var_symbol = f"({x_label})"
    eq_str = poly_to_string(coeffs, var_name=var_symbol)

    return {
        "mode": mode,
        "L_over_2r0": L_input,
        "S_over_2r0": S_input,
        "x_label": x_label,
        "x_value": x_input,
        "aA": a_at_point,
        "poly_degree": deg,
        "poly_R2": r2,
        "poly_equation": eq_str,
    }


# ============================================
# 2. FastAPI 엔드포인트
# ============================================

@app.get("/", response_class=HTMLResponse)
def index():
    """간단한 입력 폼 페이지"""
    html = """
    <html>
    <head>
      <title>Dynamicspring aA Calculator</title>
      <meta charset="utf-8" />
    </head>
    <body>
      <h2>Dynamicspring aA Calculator</h2>
      <form action="/calc" method="get">
        <label>L/2r₀: <input type="number" step="0.0001" name="L" required></label><br><br>
        <label>S/2r₀: <input type="number" step="0.0001" name="S" required></label><br><br>
        <button type="submit">계산</button>
      </form>
      <p>또는 /calc?L=57.8&S=2.0 형식으로 직접 호출할 수 있습니다.</p>
    </body>
    </html>
    """
    return html


@app.get("/calc")
def calc(
    L: float = Query(..., description="L/2r0 값"),
    S: float = Query(..., description="S/2r0 값"),
):
    """
    예: /calc?L=57.8&S=2.0
    JSON으로 aA, 회귀식, R^2 등 반환
    """
    res = compute_one(L, S)
    return res
