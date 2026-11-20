import random
import unittest

import entrenado as T
import predicciones as PR
from test_pred_ia import CFG, OPS, libro, lista, opinar_fijo

H = 3_600_000


def ejemplos(n, senal, semilla=1):
    """Ejemplos sintéticos: con `senal` el modelo estadístico sabe más que el precio."""
    r = random.Random(semilla)
    out = []
    for i in range(n):
        verdad = r.uniform(0.1, 0.9)
        p = min(0.95, max(0.05, verdad + r.uniform(-0.35, 0.35)))      # el mercado se equivoca
        pm = min(0.97, max(0.03, verdad if senal else r.uniform(0.05, 0.95)))
        out.append({"p": p, "pm": pm, "h": r.uniform(1, 20), "y": 1.0 if r.random() < verdad else 0.0, "t": i * H, "o": "tuyo"})
    return out


class Mercados(unittest.TestCase):
    def test_lee_binario_y_tramos_de_settled_outcome(self):
        b = T.mercado_de({"spec": {"description": "class:priceBinary|underlying:SOL|expiry:20261112-0600|targetPrice:83.333|period:1d"}, "settleFraction": "0.0"})
        self.assertEqual((b["sub"], b["tipo"], b["K"], b["sf"]), ("SOL", "sobre", 83.333, 0.0))
        q = {"description": "class:priceBucket|underlying:BTC|expiry:20261119-0600|priceThresholds:63373,65960|period:1d"}
        t0 = T.mercado_de({"spec": {"description": "index:0"}, "settleFraction": "1.0", "question": q})
        t1 = T.mercado_de({"spec": {"description": "index:1"}, "settleFraction": "0.0", "question": q})
        t2 = T.mercado_de({"spec": {"description": "index:2"}, "settleFraction": "0.0", "question": q})
        self.assertEqual((t0["desde"], t0["hasta"]), (None, 63373.0))
        self.assertEqual((t1["desde"], t1["hasta"]), (63373.0, 65960.0))
        self.assertEqual((t2["desde"], t2["hasta"]), (65960.0, None))
        self.assertIsNone(T.mercado_de({"spec": {"description": "perp:BTC|seconds:90"}, "settleFraction": "1.0"}))
        self.assertIsNone(T.mercado_de(None))

    def test_agrupa_tus_compras_y_saca_ejemplos(self):
        fills = [{"coin": "#402640", "dir": "Buy", "side": "B", "px": "0.8", "sz": "2", "time": 10 * H},
                 {"coin": "#402640", "dir": "Buy", "side": "B", "px": "0.6", "sz": "2", "time": 11 * H},
                 {"coin": "#402640", "dir": "Settlement", "side": "A", "px": "1.0", "sz": "4", "time": 20 * H},
                 {"coin": "@107", "dir": "Buy", "side": "B", "px": "60", "sz": "1", "time": 10 * H}]
        c = T.compras_de(fills)
        self.assertEqual(len(c), 1)
        self.assertEqual((c[0]["m"], c[0]["l"], c[0]["t"], round(c[0]["p"], 3), c[0]["u"]), (40264, 0, 10 * H, 0.7, 4.0))
        velas = [{"t": 10 * H - k * 15 * 60_000, "c": 100 * (1 + 0.001 * ((-1) ** k))} for k in range(100)][::-1]

        def info(cuerpo):
            if cuerpo["type"] == "settledOutcome":
                return {"spec": {"description": "class:priceBinary|underlying:BTC|expiry:19700101-2000|targetPrice:100|period:1d"}, "settleFraction": "1.0"}
            return velas
        reg, n = T.actualizar({}, fills, info, ahora=30 * H)
        self.assertEqual(n, 1)
        e = reg["ejemplos"][0]
        self.assertEqual((e["y"], e["p"], e["h"]), (1.0, 0.7, 10.0))
        self.assertTrue(0.3 < e["pm"] < 0.7)
        # Ya visto: no lo vuelve a consultar.
        reg2, n2 = T.actualizar(reg, fills, lambda c: self.fail("no debería consultar"), ahora=31 * H)
        self.assertEqual((n2, len(reg2["ejemplos"])), (0, 1))


class Modelo(unittest.TestCase):
    def test_sin_datos_suficientes_no_influye(self):
        m = T.entrenar(ejemplos(10, True))
        self.assertEqual((m["peso"], m["w"]), (0.0, T.PRIOR))

    def test_sin_senal_no_influye(self):
        m = T.entrenar(ejemplos(120, False))
        self.assertEqual(m["peso"], 0.0)

    def test_con_senal_real_influye_y_corrige_hacia_el_modelo(self):
        m = T.entrenar(ejemplos(250, True))
        self.assertGreater(m["peso"], 0)
        self.assertLess(m["brier_cv"], m["brier_mercado"])
        self.assertGreater(m["w"][2], 0.2)  # aprende a creerle al modelo estadístico
        self.assertGreater(T.prob_lado(m, 0.4, 0.8, 5), 0.5)


class Validacion(unittest.TestCase):
    def test_no_entrena_con_el_futuro_ni_con_resultados_que_aun_no_existen(self):
        # Un ejemplo de hace 1 h que se resolvió en 20 h cae dentro del bloque de prueba: se purga.
        datos = ejemplos(250, True)
        for d in datos:
            d["h"] = 20.0
        m = T.entrenar(datos)
        self.assertGreater(m["purgados"], 0)
        self.assertEqual(m["bloques"], 5)

    def test_influye_solo_si_mejora_en_la_mayoria_de_bloques(self):
        m = T.entrenar(ejemplos(250, True))
        self.assertGreaterEqual(m["bloques_ok"], 4)

    def test_deriva_detecta_cambio_de_precios(self):
        viejo = [{"p": 0.3 + 0.001 * i % 0.2, "pm": 0.5, "h": 5.0, "t": i * H} for i in range(100)]
        nuevo = [{"p": 0.9, "pm": 0.5, "h": 5.0, "t": (100 + i) * H} for i in range(40)]
        d = T.deriva(viejo + nuevo)
        self.assertIn(d["estado"], ("moderada", "fuerte"))
        self.assertGreater(d["psi"]["precio"], 0.2)

    def test_deriva_estable_sin_cambios(self):
        datos = [{"p": 0.2 + 0.6 * ((i * 37) % 100) / 100, "pm": 0.5, "h": 5.0, "t": i * H} for i in range(200)]
        self.assertEqual(T.deriva(datos)["estado"], "estable")


class Decide(unittest.TestCase):
    def test_un_modelo_con_peso_frena_una_compra_sin_margen(self):
        malo = {"n": 100, "peso": 0.5, "w": [-3.0, 0.0, 0.0, 0.0]}  # le da ~5 % a todo
        nuevas, notas, res = PR.proponer(CFG, lista(), [], [], 50, opinar_fijo(OPS), libro(0.42), aprender={"modelo": malo})
        self.assertEqual(nuevas, [])
        self.assertTrue(any("Modelo entrenado" in n for n in notas) or res["candidatos"] == 0)

    def test_sin_peso_solo_informa_a_la_ia(self):
        neutro = {"n": 40, "peso": 0.0, "w": T.PRIOR}
        op = opinar_fijo(OPS)
        nuevas, _, res = PR.proponer(CFG, lista(), [], [], 50, op, libro(0.42), aprender={"modelo": neutro})
        self.assertEqual(len(nuevas), 1)
        self.assertIn("prob_modelo_entrenado_pct", op.llamadas[0][0][0])
        self.assertIsNotNone(res["decisiones"][0]["p_ent"])



class Aprende(unittest.TestCase):
    def test_tus_operaciones_entran_al_aprendizaje_sin_contarse_dos_veces(self):
        import aprendizaje as A
        ej = [{"m": 40264, "l": 0, "t": 5 * H, "sub": "BTC", "p": 0.7, "pm": 0.8, "h": 10, "y": 1.0, "o": "tuyo"},
              {"m": 39279, "l": 1, "t": 4 * H, "sub": "BTC", "p": 0.84, "pm": 0.76, "h": 4, "y": 0.0, "o": "tuyo"}]
        apr, n = A.importar({}, T.como_decisiones(ej))
        self.assertEqual(n, 2)
        apr, n = A.importar(apr, T.como_decisiones(ej))
        self.assertEqual(n, 0)
        p = A.puntaje(apr, ahora=6 * H)
        self.assertEqual((p["compras"], p["ganadas"]), (2, 1))
        self.assertEqual(T.ejemplos_del_agente(apr), [])  # no se duplican en el entrenamiento



if __name__ == "__main__":
    unittest.main()
