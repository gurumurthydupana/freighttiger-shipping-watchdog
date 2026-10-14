"""Core logic: weekly cost per tonne-km and the two baselines defined in the brief."""
import pandas as pd

HISTORY_WINDOW = 8  # trailing weeks, strictly before the current week


def load_shipments(path):
    df = pd.read_csv(path, parse_dates=["shipment_date"])
    df["route"] = df["origin"] + "-" + df["destination"]
    # Monday of the week containing the shipment (weeks run Mon-Sun)
    df["week_of"] = df["shipment_date"] - pd.to_timedelta(df["shipment_date"].dt.weekday, unit="D")
    df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]
    return df


def weekly_table(df):
    """One row per (route, route_type, week): cost per tonne-km = total cost / total tonne-km."""
    g = (df.groupby(["route", "route_type", "week_of"], as_index=False)
           .agg(cost=("freight_cost_inr", "sum"), tonne_km=("tonne_km", "sum"),
                n_shipments=("shipment_id", "count")))
    g["cost_per_tonne_km"] = g["cost"] / g["tonne_km"]
    return g.sort_values(["route", "week_of"]).reset_index(drop=True)


def add_baselines(w):
    w = w.copy()
    own, own_n = [], []
    for route, grp in w.groupby("route"):
        vals = grp["cost_per_tonne_km"].tolist()
        for i in range(len(vals)):
            prior = vals[max(0, i - HISTORY_WINDOW):i]  # strictly before current week
            own.append(sum(prior) / len(prior) if prior else float("nan"))
            own_n.append(len(prior))
    # groupby preserves the (route, week) sort order, so assignment lines up
    w["own_history_avg"] = own
    w["own_history_weeks_used"] = own_n

    # peers: same route_type, same week, excluding the route itself
    grp = w.groupby(["route_type", "week_of"])["cost_per_tonne_km"]
    s, n = grp.transform("sum"), grp.transform("count")
    w["peer_avg"] = (s - w["cost_per_tonne_km"]) / (n - 1)
    w.loc[n <= 1, "peer_avg"] = float("nan")

    w["vs_own_pct"] = (w["cost_per_tonne_km"] / w["own_history_avg"] - 1) * 100
    w["vs_peer_pct"] = (w["cost_per_tonne_km"] / w["peer_avg"] - 1) * 100
    return w
