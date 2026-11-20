"""Modelo entrenado con TUS operaciones reales de Hyperliquid (y con las decisiones
resueltas del propio agente).

Cada compra tuya en un mercado de predicción es un ejemplo:
  - p:  precio que pagaste (= probabilidad que le daba el mercado a ese lado),
  - pm: probabilidad del modelo estadístico EN ESE MOMENTO, recalculada con las velas
        de 15 min de Hyperliquid (mismo modelo log-normal que usa la nube),
  - h:  horas que faltaban para vencer,
  - y:  1 si ese lado ganó según el resultado OFICIAL (`settledOutcome`), 0 si perdió.

Con eso se ajusta una regresión logística pequeña,
    P(gana) = σ(b0 + b1·logit(p) + b2·(logit(pm) − logit(p)) + b3·ln(1+h)),
regularizada hacia «el mercado tiene razón» (b = 0, 1, 0, 0): con pocos datos casi no
se mueve del precio. Se valida por partes (5 particiones) y solo influye si acierta
mejor que el precio del mercado (Brier); su peso crece con los datos y la mejora, hasta
PESO_MAX. Sin dependencias: Newton con matrices 4×4 en Python puro.
"""
import math
import time

HORA = 3_600_000
DIA = 24 * HORA
MAX_EJEMPLOS = 400
MAX_RESULTADOS_CICLO = 25     # settledOutcome por ciclo (lo demás, en los siguientes)
MIN_ENTRENAR = 20             # ejemplos resueltos para ajustar algo
MIN_INFLUIR = 30              # y para que influya en las decisiones
PESO_MAX = 0.5
LAMBDA = 4.0                  # cuánto tira hacia «el mercado tiene razón»
PRIOR = [0.0, 1.0, 0.0, 0.0]
VELAS_HORAS = 26              # velas antes de cada compra para la volatilidad


# ---------------------------------------------------------------- utilidades
def _clip(p, a=0.001, b=0.999):
    return min(b, max(a, float(p)))


def logit(p):
    p = _clip(p)
    return math.log(p / (1 - p))


def sig(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def normal(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def prob_sobre(S, K, sigma_h, horas):
    """P(precio final ≥ K), log-normal (igual que la nube)."""
    if not (S and K and sigma_h and horas) or S <= 0 or K <= 0 or sigma_h <= 0 or horas <= 0:
        return None
    s = sigma_h * math.sqrt(horas)
    return normal((math.log(S / K) - s * s / 2) / s)


def _campos(desc):
    out = {}
    for parte in str(desc or "").split("|"):
        k, _, v = parte.partition(":")
        if v:
            out[k] = v
    return out


def _vence(t):
    import re
    m = re.match(r"^(\d{4})(\d\d)(\d\d)-(\d\d)(\d\d)", str(t or ""))
    if not m:
        return None
    import calendar
    return calendar.timegm((int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5]), 0)) * 1000


def mercado_de(settled):
    """Del `settledOutcome` saca qué se preguntaba: {sub, tipo, K | (K1, K2), vence, sf}.
    priceBinary: «≥ K». priceBucket: el índice i de umbrales [K1, K2, …] es el tramo i
    (0: < K1, 1: K1–K2, …, último: ≥ Kn)."""
    if not isinstance(settled, dict) or settled.get("settleFraction") is None:
        return None
    spec = settled.get("spec") or {}
    d = _campos(spec.get("description"))
    q = _campos((settled.get("question") or {}).get("description"))
    base = d if d.get("class") else q
    if not base.get("underlying"):
        return None
    out = {"sub": base["underlying"], "vence": _vence(base.get("expiry")), "sf": float(settled["settleFraction"])}
    if base.get("class") == "priceBinary" and base.get("targetPrice"):
        out.update(tipo="sobre", K=float(base["targetPrice"]))
    elif base.get("class") == "priceBucket" and base.get("priceThresholds") and "index" in d:
        ks = [float(x) for x in base["priceThresholds"].split(",") if x]
        i = int(d["index"])
        out.update(tipo="tramo", desde=ks[i - 1] if i >= 1 else None, hasta=ks[i] if i < len(ks) else None)
    else:
        return None
    return out if out["vence"] else None


def _sigma_hora(velas):
    """Volatilidad por hora con cierres de 15 min (desviación de los retornos × 2)."""
    cs = [float(v["c"]) for v in velas if float(v.get("c") or 0) > 0]
    rs = [math.log(b / a) for a, b in zip(cs, cs[1:])]
    if len(rs) < 8:
        return None
    m = sum(rs) / len(rs)
    var = sum((r - m) ** 2 for r in rs) / (len(rs) - 1)
    return math.sqrt(var) * 2


def prob_modelo_en(mk, velas, t):
    """Probabilidad del modelo para el lado «Sí» en el instante t (ms)."""
    antes = [v for v in velas if t - VELAS_HORAS * HORA <= int(v["t"]) <= t]
    if len(antes) < 10:
        return None
    S = float(antes[-1]["c"])
    sg = _sigma_hora(antes)
    horas = (mk["vence"] - t) / HORA
    if sg is None or horas <= 0:
        return None
    if mk["tipo"] == "sobre":
        return prob_sobre(S, mk["K"], sg, horas)
    a = 1.0 if mk.get("desde") is None else prob_sobre(S, mk["desde"], sg, horas)
    b = 0.0 if mk.get("hasta") is None else prob_sobre(S, mk["hasta"], sg, horas)
    return None if a is None or b is None else max(0.0, a - b)


# ---------------------------------------------------------------- datos
def compras_de(fills):
    """Tus compras en predicciones agrupadas por mercado+lado: precio medio y primera hora."""
    g = {}
    for f in fills or []:
        c = str(f.get("coin") or "")
        if c[:1] not in "#+" or f.get("dir") != "Buy":
            continue
        try:
            n = int(c[1:])
            px, sz, t = float(f["px"]), float(f["sz"]), int(f["time"])
        except (KeyError, TypeError, ValueError):
            continue
        k = (n // 10, n % 10)
        x = g.setdefault(k, {"m": k[0], "l": k[1], "t": t, "u": 0.0, "costo": 0.0})
        x["t"] = min(x["t"], t)
        x["u"] += sz
        x["costo"] += px * sz
    return [{**x, "p": x["costo"] / x["u"]} for x in g.values() if x["u"] > 0]


def actualizar(reg, fills, info, ahora=None, hasta=None, max_resultados=MAX_RESULTADOS_CICLO):
    """Suma al registro `reg` (dict) los ejemplos nuevos de tus compras ya liquidadas.
    `info(cuerpo)` = POST a la API pública de Hyperliquid. Devuelve (reg, nuevos)."""
    ahora = ahora or int(time.time() * 1000)
    reg = dict(reg or {})
    ej = list(reg.get("ejemplos") or [])
    vistos = {(e["m"], e["l"]) for e in ej}
    malos = set(tuple(x) for x in reg.get("descartados") or [])
    pend = [c for c in compras_de(fills) if (c["m"], c["l"]) not in vistos and (c["m"], c["l"]) not in malos]
    pend.sort(key=lambda c: -c["t"])
    resultados, velas_por, nuevos, consultas = {}, {}, 0, 0
    for c in pend:
        if consultas >= max_resultados or (hasta is not None and time.time() >= hasta):
            break
        if c["m"] not in resultados:
            consultas += 1
            try:
                resultados[c["m"]] = mercado_de(info({"type": "settledOutcome", "outcome": int(c["m"])}))
            except Exception:
                resultados[c["m"]] = None
        mk = resultados[c["m"]]
        if mk is None:
            # Sin liquidar todavía (o de un tipo que el modelo no entiende): se reintenta,
            # salvo que ya tenga más de 10 días.
            if ahora - c["t"] > 10 * DIA:
                malos.add((c["m"], c["l"]))
            continue
        if mk["sub"] not in velas_por:
            try:
                velas_por[mk["sub"]] = info({"type": "candleSnapshot", "req": {"coin": mk["sub"], "interval": "15m",
                                             "startTime": min(x["t"] for x in pend) - VELAS_HORAS * HORA, "endTime": ahora}}) or []
            except Exception:
                velas_por[mk["sub"]] = []
        pm_si = prob_modelo_en(mk, velas_por[mk["sub"]], c["t"])
        if pm_si is None:
            malos.add((c["m"], c["l"]))
            continue
        y = mk["sf"] if c["l"] == 0 else 1 - mk["sf"]
        ej.append({"m": c["m"], "l": c["l"], "t": c["t"], "sub": mk["sub"], "p": round(c["p"], 4),
                   "pm": round(pm_si if c["l"] == 0 else 1 - pm_si, 4), "h": round((mk["vence"] - c["t"]) / HORA, 2),
                   "y": round(y, 3), "u": round(c["u"], 2), "o": "tuyo"})
        nuevos += 1
    ej.sort(key=lambda e: -e["t"])
    reg.update(ejemplos=ej[:MAX_EJEMPLOS], descartados=[list(x) for x in list(malos)[-300:]], ts=ahora)
    return reg, nuevos


def ejemplos_del_agente(apr):
    """Decisiones resueltas del propio agente con precio y modelo: también entrenan."""
    out = []
    for d in (apr or {}).get("decisiones") or []:
        if d.get("origen") == "hyperliquid":
            continue  # tus operaciones ya entran como ejemplos propios: no contarlas dos veces
        if d.get("y") is None or d.get("p_mercado") is None or d.get("p_modelo") is None:
            continue
        out.append({"p": float(d["p_mercado"]), "pm": float(d["p_modelo"]), "h": float(d.get("horas") or 0), "y": float(d["y"]), "o": "agente"})
    return out


# ---------------------------------------------------------------- modelo
def _x(p, pm, h):
    lp = logit(p)
    return [1.0, lp, logit(pm) - lp, math.log1p(max(0.0, h))]


def _resolver4(A, b):
    """Gauss-Jordan para el sistema 4×4 de Newton."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-12:
            return None
        M[c], M[piv] = M[piv], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def ajustar(datos, lam=LAMBDA, iters=25):
    """Regresión logística con penalización λ·|w − PRIOR|² (Newton)."""
    w = list(PRIOR)
    X = [_x(d["p"], d["pm"], d["h"]) for d in datos]
    Y = [d["y"] for d in datos]
    for _ in range(iters):
        g = [lam * 2 * (w[j] - PRIOR[j]) for j in range(4)]
        H = [[(lam * 2 if i == j else 0.0) for j in range(4)] for i in range(4)]
        for x, y in zip(X, Y):
            p = sig(sum(a * b for a, b in zip(w, x)))
            for i in range(4):
                g[i] += (p - y) * x[i]
                for j in range(4):
                    H[i][j] += p * (1 - p) * x[i] * x[j]
        paso = _resolver4(H, g)
        if paso is None:
            break
        w = [a - b for a, b in zip(w, paso)]
        if max(abs(b) for b in paso) < 1e-7:
            break
    return w


def prob(w, p, pm, h):
    return sig(sum(a * b for a, b in zip(w, _x(p, pm, h))))


def _brier(pares):
    return sum((p - y) ** 2 for p, y in pares) / len(pares) if pares else None


EMBARGO_MS = 6 * HORA         # margen tras el bloque de prueba (autocorrelación)


def _psi(ref, cur, bins=10):
    """Population Stability Index entre dos muestras (ref = periodo de entrenamiento)."""
    if len(ref) < bins or not cur:
        return None
    s = sorted(ref)
    cortes = [s[min(len(s) - 1, (len(s) * i) // bins)] for i in range(1, bins)]
    def dist(xs):
        c = [0] * bins
        for x in xs:
            i = sum(1 for k in cortes if x > k)
            c[i] += 1
        return [max(v / len(xs), 1e-4) for v in c]
    r, c = dist(ref), dist(cur)
    return round(sum((cc - rr) * math.log(cc / rr) for rr, cc in zip(r, c)), 4)


def deriva(datos, corte=0.7):
    """PSI de las entradas (precio, modelo, horas) entre lo viejo (70 %) y lo reciente (30 %).
    > 0,20 = deriva fuerte: el mercado ya no se parece al que aprendió el modelo."""
    d = sorted(datos, key=lambda e: e.get("t") or 0)
    k = int(len(d) * corte)
    viejo, nuevo = d[:k], d[k:]
    if len(viejo) < 20 or len(nuevo) < 10:
        return {"estado": "sin datos", "psi": {}}
    psi = {nom: _psi([x[c] for x in viejo], [x[c] for x in nuevo]) for nom, c in (("precio", "p"), ("modelo", "pm"), ("horas", "h"))}
    mx = max(v for v in psi.values() if v is not None) if any(v is not None for v in psi.values()) else None
    estado = "sin datos" if mx is None else "fuerte" if mx > 0.20 else "moderada" if mx > 0.10 else "estable"
    return {"estado": estado, "psi": psi}


def entrenar(datos, k=5):
    """Ajusta y valida en bloques TEMPORALES (cada bloque es el futuro de lo que vio el modelo).
    Purga: quita de entrenamiento lo que se resolvió dentro del bloque de prueba.
    Embargo: quita lo que ocurrió justo después del bloque. Solo influye si mejora el mercado
    en la mayoría de los bloques (no basta con el promedio)."""
    datos = [d for d in datos if d.get("p") is not None and d.get("pm") is not None and d.get("y") is not None]
    n = len(datos)
    base = {"n": n, "tuyos": sum(1 for d in datos if d.get("o") == "tuyo"), "agente": sum(1 for d in datos if d.get("o") == "agente"),
            "peso": 0.0, "w": PRIOR, "ts": int(time.time() * 1000)}
    if n < MIN_ENTRENAR:
        return {**base, "motivo": f"Faltan datos: {n} de {MIN_ENTRENAR} operaciones resueltas"}
    datos = sorted(datos, key=lambda d: d.get("t") or 0)
    bloques = [list(range((f * n) // k, ((f + 1) * n) // k)) for f in range(k)]
    pares_cv, gana_bloques, purgados = [], 0, 0
    for idx in bloques:
        if not idx:
            continue
        ini = datos[idx[0]].get("t") or 0
        fin = datos[idx[-1]].get("t") or 0
        prueba = [datos[i] for i in idx]
        resto = []
        for i, d in enumerate(datos):
            if i in idx:
                continue
            res = (d.get("t") or 0) + (d.get("h") or 0) * HORA          # cuándo se resolvió
            antes = (d.get("t") or 0) < ini
            if antes and res >= ini:          # se resolvió dentro del bloque de prueba: purga
                purgados += 1
                continue
            if not antes and (d.get("t") or 0) <= fin + EMBARGO_MS:  # justo después: embargo
                purgados += 1
                continue
            resto.append(d)
        if len(resto) < 5:
            continue
        w = ajustar(resto)
        pb = [(prob(w, d["p"], d["pm"], d["h"]), d["y"]) for d in prueba]
        pares_cv += pb
        bm = _brier([(_clip(d["p"]), d["y"]) for d in prueba])
        bo = _brier(pb)
        if bo is not None and bm is not None and bo < bm:
            gana_bloques += 1
    if not pares_cv:
        return {**base, "motivo": "Faltan operaciones con tiempos suficientes para validar por bloques"}
    w = ajustar(datos)
    b_cv = _brier(pares_cv)
    b_mer = _brier([(_clip(d["p"]), d["y"]) for d in datos])
    b_mod = _brier([(_clip(d["pm"]), d["y"]) for d in datos])
    mejora = (b_mer - b_cv) / b_mer if b_mer and b_cv is not None else 0.0
    estables = gana_bloques >= -(-4 * k // 5)     # al menos 4 de 5 bloques
    peso = 0.0
    if n >= MIN_INFLUIR and mejora > 0 and estables:
        peso = round(min(PESO_MAX, mejora * 4) * min(1.0, n / 100), 3)
    ganadas = sum(1 for d in datos if d["y"] >= 0.5)
    ventaja = sum(d["y"] - d["p"] for d in datos) / n
    if peso > 0:
        motivo = f"Influye con peso {peso:.2f}: en validación acierta mejor que el precio en {gana_bloques} de {k} bloques de tiempo"
    elif n < MIN_INFLUIR:
        motivo = f"Aprendiendo: influye desde {MIN_INFLUIR} operaciones resueltas (hay {n})"
    elif mejora > 0 and not estables:
        motivo = f"No influye: mejora el precio en promedio, pero solo en {gana_bloques} de {k} bloques de tiempo (hace falta 4 de {k})"
    else:
        motivo = "No influye: todavía no acierta mejor que el precio del mercado"
    return {**base, "w": [round(x, 4) for x in w], "brier_cv": round(b_cv, 4), "brier_mercado": round(b_mer, 4),
            "brier_modelo": round(b_mod, 4), "peso": peso, "ganadas": ganadas, "ventaja_pts": round(ventaja * 100, 1),
            "bloques": k, "bloques_ok": gana_bloques, "purgados": purgados, "deriva": deriva(datos), "motivo": motivo}


def lecciones(datos, maximo=3):
    """Frases para la IA con lo que dicen tus operaciones (solo con muestra suficiente)."""
    out = []
    tuyos = [d for d in datos if d.get("o") == "tuyo"]
    for a, b, nombre in ((0.0, 0.3, "baratos (< 0,30)"), (0.3, 0.7, "intermedios (0,30–0,70)"), (0.7, 1.01, "caros (≥ 0,70)")):
        t = [d for d in tuyos if a <= d["p"] < b]
        if len(t) >= 8:
            gan = sum(1 for d in t if d["y"] >= 0.5) / len(t)
            pag = sum(d["p"] for d in t) / len(t)
            if abs(gan - pag) >= 0.08:
                out.append(f"En tus operaciones reales, los lados {nombre} pagados en promedio a {pag:.2f} ganaron {gan * 100:.0f}% de las veces ({len(t)} casos): "
                           + ("el mercado los sobrevaloraba." if gan < pag else "el mercado los subvaloraba."))
    return out[:maximo]


def resumen_para_ia(modelo):
    """Una frase para la IA con cómo te va y si el modelo entrenado ya influye."""
    if not modelo or not modelo.get("n"):
        return []
    m = modelo
    frase = (f"Historial real del dueño: {m.get('ganadas', 0)}/{m['n']} lados ganados, ventaja real {m.get('ventaja_pts', 0):+g} puntos "
             f"sobre lo pagado. Brier: mercado {m.get('brier_mercado')}, modelo estadístico {m.get('brier_modelo')}, modelo entrenado {m.get('brier_cv')}.")
    if m.get("peso"):
        frase += f" El modelo entrenado pesa {m['peso']:.2f} en la decisión: tenlo en cuenta (va en prob_modelo_entrenado_pct)."
    return [frase] if m.get("brier_mercado") is not None else []


def prob_lado(modelo, precio, p_modelo, horas):
    """Probabilidad del modelo entrenado para un lado (0..1) o None si aún no hay modelo."""
    if not modelo or (modelo.get("n") or 0) < MIN_ENTRENAR or precio is None or p_modelo is None:
        return None
    return prob(modelo.get("w") or PRIOR, precio, p_modelo, float(horas or 0))


def como_decisiones(ejemplos):
    """Tus operaciones reales ya resueltas, en el formato del registro de aprendizaje
    (cuentan en «compras ganadas», «ventaja real», calibración y lecciones)."""
    out = []
    for e in ejemplos or []:
        if e.get("o") != "tuyo" or e.get("y") is None:
            continue
        cod = int(e["m"]) * 10 + int(e["l"])
        out.append({"mercado": e["m"], "lado": e["l"], "coin": f"#{cod}", "etiqueta": f"{e.get('sub', '?')} · tu operación real"[:90],
                    "categoria": "cripto_hoy" if (e.get("h") or 0) <= 36 else "cripto", "horas": e.get("h"),
                    "precio": e["p"], "p_mercado": e["p"], "p_modelo": e["pm"], "accion": "comprar", "origen": "hyperliquid",
                    "ts": e["t"], "vence": int(e["t"] + (e.get("h") or 0) * HORA), "y": e["y"], "resuelto": e["t"]})
    return out
