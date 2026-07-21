"""
threshold_review.py
====================

Herramienta de revisión manual del umbral de alerta (alert_h) por familia.

No es un script automático de reentreno: se ejecuta a mano cuando toca revisar
(ver criterio de disparo más abajo), usando los datos acumulados hasta ese
momento. Requiere criterio humano para la decisión final — el suelo operativo
mínimo por familia no sale de los datos, sale de mantenimiento real.

Uso típico:

    python3 threshold_review.py --predictions t2_predictions_log.csv \
                                 --faults turbine_2_fault_log.csv \
                                 --family yaw_cable

    python3 threshold_review.py --check-trigger \
                                 --faults turbine_2_fault_log.csv \
                                 --since-date 2026-06-13 \
                                 --min-new-events 2
"""

import argparse
import numpy as np
import pandas as pd

# Suelos operativos mínimos estimados (ver DECISIONS.md sección 11).
# Pendientes de validación con el equipo de mantenimiento real.
OPERATIONAL_FLOOR_H = {
    'yaw_cable':   24,   # rango estimado 24-36h, se usa el extremo conservador como suelo duro
    'generator':   72,
    'brake_hydro': 24,
    'pitch_bat':   120,
}

MATCH_WINDOW_H_DEFAULT = 120  # ventana hacia delante para asociar una alerta a un evento real
CLUSTER_GAP_H_DEFAULT = 24    # fallos de la misma familia a <24h se consideran el mismo evento


def cluster_events(timestamps, gap_hours=CLUSTER_GAP_H_DEFAULT):
    """Colapsa fallos consecutivos de una familia en eventos únicos."""
    ts = sorted(timestamps)
    if not ts:
        return []
    events = [ts[0]]
    cur_last = ts[0]
    for t in ts[1:]:
        if (t - cur_last).total_seconds() / 3600 > gap_hours:
            events.append(t)
        cur_last = t
    return events


def evaluate_threshold(pred_series, events, threshold, match_window_h=MATCH_WINDOW_H_DEFAULT):
    """
    pred_series: DataFrame con columnas ['last_data_ts', 'pred_h'] para UN modelo/familia.
    events: lista de timestamps de fallos reales (ya clusterizados).
    threshold: umbral candidato en horas.

    Devuelve recall a nivel de evento, precisión a nivel de alerta, y lead time medio.
    """
    p = pred_series.copy()
    p['alert'] = p['pred_h'] < threshold
    alert_days = p[p['alert']]['last_data_ts'].tolist()

    tp_events, lead_times = 0, []
    for ev in events:
        prior = [a for a in alert_days if a < ev and (ev - a).total_seconds() / 3600 <= match_window_h]
        if prior:
            tp_events += 1
            lead_times.append((ev - min(prior)).total_seconds() / 3600)
    recall = tp_events / len(events) if events else np.nan

    tp_alerts = sum(
        1 for a in alert_days
        if any(0 <= (ev - a).total_seconds() / 3600 <= match_window_h for ev in events)
    )
    precision = tp_alerts / len(alert_days) if alert_days else np.nan

    return dict(
        threshold=threshold, n_alert_days=len(alert_days),
        tp_events=tp_events, n_events=len(events),
        recall=recall, precision=precision,
        avg_lead_h=np.mean(lead_times) if lead_times else np.nan,
    )


def sweep(pred_series, events, floor_h, ceiling_h=250, step_h=2, match_window_h=MATCH_WINDOW_H_DEFAULT):
    """Barrido completo de umbrales candidatos, respetando el suelo operativo."""
    rows = [
        evaluate_threshold(pred_series, events, th, match_window_h)
        for th in range(floor_h, ceiling_h + 1, step_h)
    ]
    return pd.DataFrame(rows)


def recommend_threshold(sweep_df, floor_h):
    """
    Regla de decisión (ver DECISIONS.md sección 11):
    umbral_final = max( umbral mínimo con recall=100% , suelo operativo mínimo )
    Nunca se baja del suelo operativo aunque los datos lo permitan.
    """
    full_recall = sweep_df[sweep_df['recall'] >= 0.999]
    if full_recall.empty:
        return None, "Ningún umbral evaluado alcanza 100% de recall — no se puede recomendar un ajuste todavía."
    data_driven = int(full_recall['threshold'].min())
    final = max(data_driven, floor_h)
    reason = (
        f"umbral data-driven (recall 100%): {data_driven}h · "
        f"suelo operativo: {floor_h}h · resultado: {final}h"
    )
    return final, reason


def check_trigger(fault_log_path, family, since_date, min_new_events, gap_hours=CLUSTER_GAP_H_DEFAULT):
    """
    Criterio de disparo por volumen (no por calendario): ¿hay ya suficientes
    eventos nuevos desde la última revisión para justificar recalcular el umbral?
    """
    faults = pd.read_csv(fault_log_path, parse_dates=['timestamp'])
    fam_faults = faults[faults['family'] == family]
    events = cluster_events(fam_faults['timestamp'].tolist(), gap_hours)
    new_events = [e for e in events if e >= pd.Timestamp(since_date)]
    should_review = len(new_events) >= min_new_events
    return dict(
        family=family, new_events=len(new_events), min_required=min_new_events,
        should_review=should_review,
        event_dates=[e.strftime('%Y-%m-%d %H:%M') for e in new_events],
    )


def run_family_review(predictions_path, fault_log_path, family, match_window_h=MATCH_WINDOW_H_DEFAULT):
    faults = pd.read_csv(fault_log_path, parse_dates=['timestamp'])
    preds = pd.read_csv(predictions_path, parse_dates=['date', 'last_data_ts'])

    fam_faults = faults[faults['family'] == family]
    all_events = cluster_events(fam_faults['timestamp'].tolist())

    fam_preds = preds[preds['family'] == family].sort_values('last_data_ts').reset_index(drop=True)
    if fam_preds.empty:
        print(f"[{family}] Sin predicciones registradas todavía.")
        return

    pred_start = fam_preds['date'].min()
    evaluable_events = [e for e in all_events if e >= pred_start]

    print(f"=== Revisión de umbral — {family} ===")
    print(f"Ventana de predicciones: {pred_start.date()} -> {fam_preds['date'].max().date()}")
    print(f"Eventos reales evaluables en la ventana: {len(evaluable_events)}")
    if evaluable_events:
        print("  " + ", ".join(e.strftime('%Y-%m-%d %H:%M') for e in evaluable_events))

    if len(evaluable_events) < 2:
        print(
            "\nAVISO: menos de 2 eventos evaluables. Cualquier umbral recomendado aquí "
            "sería estadísticamente frágil (ver ML_DESIGN.md, advertencia sobre Event "
            "Recall con muestra pequeña). Se muestra el barrido igualmente, pero no se "
            "recomienda aplicarlo sin más eventos."
        )

    floor_h = OPERATIONAL_FLOOR_H.get(family)
    if floor_h is None:
        print(f"\nAVISO: no hay suelo operativo definido para '{family}' en OPERATIONAL_FLOOR_H.")
        floor_h = 0

    result = sweep(fam_preds[['last_data_ts', 'pred_h']], evaluable_events, floor_h, step_h=2,
                    match_window_h=match_window_h)
    pd.set_option('display.width', 120)
    print("\n" + result.to_string(index=False))

    final, reason = recommend_threshold(result, floor_h)
    print(f"\nRecomendación: {reason}")
    if final is not None:
        print(f"--> alert_h propuesto para '{family}': {final}h")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions', default='t2_predictions_log.csv')
    ap.add_argument('--faults', default='turbine_2_fault_log.csv')
    ap.add_argument('--family', choices=list(OPERATIONAL_FLOOR_H.keys()))
    ap.add_argument('--match-window-h', type=int, default=MATCH_WINDOW_H_DEFAULT)
    ap.add_argument('--check-trigger', action='store_true',
                     help="Solo comprueba si toca revisar (criterio por volumen), no hace el barrido.")
    ap.add_argument('--since-date', help="Fecha de la última revisión de umbral (YYYY-MM-DD).")
    ap.add_argument('--min-new-events', type=int, default=2)
    args = ap.parse_args()

    if args.check_trigger:
        families = [args.family] if args.family else list(OPERATIONAL_FLOOR_H.keys())
        for fam in families:
            r = check_trigger(args.faults, fam, args.since_date, args.min_new_events)
            flag = "SÍ" if r['should_review'] else "no"
            print(f"[{fam}] eventos nuevos desde {args.since_date}: {r['new_events']} "
                  f"(mínimo {r['min_required']}) -> ¿revisar? {flag}")
            if r['event_dates']:
                print("   " + ", ".join(r['event_dates']))
        return

    families = [args.family] if args.family else list(OPERATIONAL_FLOOR_H.keys())
    for fam in families:
        run_family_review(args.predictions, args.faults, fam, args.match_window_h)
        print()


if __name__ == '__main__':
    main()
