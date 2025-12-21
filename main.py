import io
import bisect
import numpy as np
import pandas as pd

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response

# Matplotlib는 서버에서 GUI 없이 렌더링해야 하므로 Agg 사용
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = FastAPI(title="Dynamicspring Web App")

# =========================
# 1) 데이터 로드 (서버 시작 시 1회)
# =========================
BELOW_CSV = "below_5.csv"
ABOVE_CSV = "above_5.csv"

df_below = pd.read_csv(BELOW_CSV).dropna(subset=["L/2ro", "S/2ro", "aA"])
df_above = pd.read_csv(ABOVE_CSV).dropna(subset=["L/2ro", "2ro/S", "aA"])

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


# =========================
# 2) 보간 + 회귀 유틸
# =========================
def interp_in_x(L, x, curves, x_col):
    sub = curves[L]
    x_vals = sub[x_col].values
    a_vals = sub["aA"].values

    # 범위 밖 클램핑
    if x <= x_vals[0]:
        return float(a_vals[0])
    if x >= x_vals[-1]:
        return float(a_vals[-1])

    return float(np.interp(x, x_vals, a_vals))


def aA_value_generic(L_target, x_target, L_values, curves, x_col):
    # L 범위 밖이면 끝값 사용
    if L_target <= L_values[0]:
        return interp_in_x(L_values[0], x_target, curves, x_col)
    if L_target >= L_values[-1]:
        return interp_in_x(L_values[-1], x_target, curves, x_col)

    i = bisect.bisect_left(L_values, L_target)
    L1, L2 = L_values[i - 1], L_values[i]

    a1 = interp_in_x(L1, x_target, curves, x_col)
    a2 = interp_in_x(L2, x_target, curves, x_col)

    t = (L_target - L1) / (L2 - L1)
    return float(a1 + (a2 - a1) * t)


def interpolated_curve_generic(L_target, df, L_values, curves, x_col, num_points=250):
    x_min = float(df[x_col].min())
    x_max = float(df[x_col].max())
    x_grid = np.linspace(x_min, x_max, num_points)
    a_grid = np.array([aA_value_generic(L_target, x, L_values, curves, x_col) for x in x_grid])
    return x_grid, a_grid


def fit_poly_with_r2(x, y, max_deg=6, target_r2=0.99):
    best = None
    for deg in range(1, max_deg + 1):
        coeffs = np.polyfit(x, y, deg)
        p = np.poly1d(coeffs)
        y_pred = p(x)

        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 1.0

        best = (deg, coeffs, r2)
        if r2 >= target_r2:
            break
    return best  # (deg, coeffs, r2)


def poly_to_string(coeffs, var_name="x"):
    # aA ≈ c0*(x)^n + ... 형태로 표시
    n = len(coeffs)
    terms = []
    for i, c in enumerate(coeffs):
        power = n - i - 1
        if abs(c) < 1e-12:
            continue

        c_abs = abs(c)
        if power == 0:
            core = f"{c_abs:.6g}"
        elif power == 1:
            core = f"{c_abs:.6g}{var_name}"
        else:
            core = f"{c_abs:.6g}{var_name}^{power}"

        sign = "-" if c < 0 else "+"
        terms.append((sign, core))

    if not terms:
        return "aA ≈ 0"

    # 첫 항은 +면 부호 생략
    first_sign, first_core = terms[0]
    s = ("" if first_sign == "+" else "-") + first_core
    for sign, core in terms[1:]:
        s += f" {sign} {core}"
    return "aA ≈ " + s


def compute_one(L_input: float, S_input: float):
    """
    S/2r0 <= 5 => below_5.csv, x = S/2r0
    S/2r0 >  5 => above_5.csv, x = 2r0/S = 1/(S/2r0)
    """
    if S_input <= 5.0:
        mode = "below"
        x_label = "S/2r0"
        x_col = "S/2ro"
        x_value = float(S_input)
        df_use = df_below
        L_values = L_values_below
        curves = curves_below
    else:
        mode = "above"
        x_label = "2r0/S"
        x_col = "2ro/S"
        x_value = float(1.0 / S_input)
        df_use = df_above
        L_values = L_values_above
        curves = curves_above

    x_grid, a_grid = interpolated_curve_generic(L_input, df_use, L_values, curves, x_col)
    a_at_point = aA_value_generic(L_input, x_value, L_values, curves, x_col)

    deg, coeffs, r2 = fit_poly_with_r2(x_grid, a_grid, max_deg=6, target_r2=0.99)
    eq = poly_to_string(coeffs, var_name=f"({x_label})")

    return {
        "mode": mode,
        "L_over_2r0": float(L_input),
        "S_over_2r0": float(S_input),
        "x_label": x_label,
        "x_value_used": x_value,
        "aA": float(a_at_point),
        "poly_degree": int(deg),
        "poly_R2": float(r2),
        "poly_equation": eq,
    }


# =========================
# 3) 웹 UI (HTML + JS)
# =========================
@app.get("/", response_class=HTMLResponse)
def home():
    html = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Dynamicspring aA Web App</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }
    .wrap { max-width: 900px; margin: 0 auto; }
    .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; margin-bottom: 16px; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: end; }
    label { display:block; font-size: 14px; color:#374151; margin-bottom: 6px; }
    input { width: 220px; padding: 10px; border-radius: 10px; border: 1px solid #d1d5db; }
    button { padding: 10px 14px; border-radius: 10px; border: 0; background: #111827; color: white; cursor:pointer; }
    button:disabled { opacity: .6; cursor: not-allowed; }
    .hint { color:#6b7280; font-size: 13px; }
    pre { white-space: pre-wrap; word-break: break-word; background:#0b1020; color:#d1d5db; padding: 12px; border-radius: 12px; }
    img { max-width: 100%; border-radius: 12px; border: 1px solid #e5e7eb; }
    .err { color:#b91c1c; font-weight:600; }
    .ok  { color:#065f46; font-weight:600; }
  </style>
</head>
<body>
<div class="wrap">
  <h2>Dynamic Spring Interaction Factor(aA) 계산기</h2>
  <div class="card">
    <div class="row">
      <div>
        <label>L/2r₀</label>
        <input id="L" type="number" step="0.0001" placeholder="예: 57.8" />
      </div>
      <div>
        <label>S/2r₀</label>
        <input id="S" type="number" step="0.0001" placeholder="예: 2.0" />
      </div>
      <div>
        <button id="btn" onclick="run()">계산</button>
      </div>
    </div>
    <p class="hint">
      S/2r₀ ≤ 5 → below 데이터 사용 (x=S/2r₀) / S/2r₀ > 5 → above 데이터 사용 (x=2r₀/S=1/(S/2r₀))
    </p>
    <div id="status"></div>
  </div>

  <div class="card">
    <h3>결과</h3>
    <pre id="out">여기에 결과가 표시됩니다.</pre>
  </div>

  <div class="card">
    <h3>그래프</h3>
    <div class="hint">원본 곡선(10/25/100) + 보간 곡선 + 회귀 곡선 + 입력점</div>
    <img id="plot" alt="plot will appear here" />
  </div>
</div>

<script>
async function run(){
  const btn = document.getElementById("btn");
  const status = document.getElementById("status");
  const out = document.getElementById("out");
  const plot = document.getElementById("plot");

  const L = document.getElementById("L").value;
  const S = document.getElementById("S").value;

  status.innerHTML = "";
  out.textContent = "계산 중...";
  plot.removeAttribute("src");

  if(!L || !S){
    status.innerHTML = '<p class="err">L/2r₀, S/2r₀ 값을 모두 입력해 주세요.</p>';
    out.textContent = "입력값이 비어 있습니다.";
    return;
  }

  btn.disabled = true;
  status.innerHTML = '<p class="hint">요청 중... (무료 Render는 첫 호출이 30~60초 느릴 수 있어요)</p>';

  try{
    const resp = await fetch(`/api/calc?L=${encodeURIComponent(L)}&S=${encodeURIComponent(S)}`);
    if(!resp.ok){
      const txt = await resp.text();
      throw new Error(txt);
    }
    const data = await resp.json();
    out.textContent = JSON.stringify(data, null, 2);

    // plot 이미지는 별도 엔드포인트에서 PNG로 받음
    const ts = Date.now(); // 캐시 방지
    plot.src = `/plot.png?L=${encodeURIComponent(L)}&S=${encodeURIComponent(S)}&ts=${ts}`;

    status.innerHTML = '<p class="ok">완료!</p>';
  }catch(e){
    status.innerHTML = '<p class="err">에러: ' + String(e).slice(0,300) + '</p>';
    out.textContent = String(e);
  }finally{
    btn.disabled = false;
  }
}
</script>

</body>
</html>
    """
    return html


# =========================
# 4) API: 계산 결과(JSON)
# =========================
@app.get("/api/calc")
def api_calc(
    L: float = Query(..., description="L/2r0"),
    S: float = Query(..., description="S/2r0"),
):
    res = compute_one(L, S)
    return JSONResponse(res)


# =========================
# 5) 그래프 PNG 생성
# =========================
@app.get("/plot.png")
def plot_png(
    L: float = Query(..., description="L/2r0"),
    S: float = Query(..., description="S/2r0"),
):
    res = compute_one(L, S)

    # 아래/위 모드에 따라 어떤 데이터셋을 그릴지 선택
    if res["mode"] == "below":
        x_col = "S/2ro"
        x_label = "S/2r0"
        L_values = L_values_below
        curves = curves_below
        df_use = df_below
        x_value = float(S)
    else:
        x_col = "2ro/S"
        x_label = "2r0/S"
        L_values = L_values_above
        curves = curves_above
        df_use = df_above
        x_value = float(1.0 / S)

    # 보간 곡선
    x_grid, a_grid = interpolated_curve_generic(float(L), df_use, L_values, curves, x_col)

    # 회귀 곡선
    # JSON에 있는 회귀식 문자열만으로는 계수 복원이 어렵기 때문에 여기서 다시 피팅함
    deg, coeffs, r2 = fit_poly_with_r2(x_grid, a_grid, max_deg=6, target_r2=0.99)
    p = np.poly1d(coeffs)
    a_reg = p(x_grid)

    # 원본 곡선 + 보간 + 회귀 + 점 표시
    fig = plt.figure(figsize=(8.0, 4.8), dpi=150)
    ax = fig.add_subplot(111)

    for Lv in L_values:
        sub = curves[Lv]
        ax.plot(sub[x_col].values, sub["aA"].values, linewidth=1.2, label=f"L/2r0={Lv:g}")

    ax.plot(x_grid, a_grid, linestyle="--", linewidth=2.0, label=f"Interpolated L/2r0={L:g}")
    ax.plot(x_grid, a_reg, linestyle=":", linewidth=2.0, label=f"Regression deg {deg} (R²={r2:.3f})")

    ax.scatter([x_value], [res["aA"]], s=35)
    ax.text(x_value, res["aA"], f"  aA={res['aA']:.3f}", va="bottom")

    ax.set_xlabel(x_label)
    ax.set_ylabel("interaction factor aA")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"Mode={res['mode']}  |  L/2r0={res['L_over_2r0']:.3f}, S/2r0={res['S_over_2r0']:.3f}")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    return Response(content=buf.read(), media_type="image/png")


