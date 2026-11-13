"""Non-image baselines. The CNN only adds value if it beats these on the same samples."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


def hand_features(store, frame):
    """Simple numeric features computed from the same 78-bar input window (no future data)."""
    s = frame.security_index.to_numpy(np.int64)
    t = frame.end_index.to_numpy(np.int64)
    lc = store.log_close

    def ret(k):                                   # k-th most recent 1-bar log return
        return lc[s, t - k + 1] - lc[s, t - k]

    past = np.stack([ret(k) for k in range(1, 78)])
    cols = {f"r{k}": past[k - 1] for k in range(1, 13)}
    cols.update({f"cum{w}": lc[s, t] - lc[s, t - w] for w in (6, 12, 39, 77)})
    cols["realized_vol"] = np.nanstd(past, axis=0)
    cols["mean_abs_r12"] = np.nanmean(np.abs(past[:12]), axis=0)
    past_volume = np.stack([store.volume[s, t - k] for k in range(1, 78)])
    cols["rel_volume"] = np.log(store.volume[s, t] / np.nanmean(past_volume, axis=0))
    cols["time_of_day"] = t % 78
    return pd.DataFrame(cols)


def tree_size(model):
    """Parameter-count analogue for a boosted tree: number of nodes across all trees."""
    return int(sum(len(p.nodes) for stage in model._predictors for p in stage))


def fit_baselines(store, target, train, valid, seed=69, max_train=300_000):
    """Return ({name: validation scores}, training prior, fitted HGB)."""
    prior = float(train.label.mean())
    Xva = hand_features(store, valid)
    scores = {"training_prior": np.full(len(valid), prior)}
    if target == "vol":
        scores["rule_recent_abs_return"] = Xva.mean_abs_r12.to_numpy()
    else:
        scores["rule_last_bar_return"] = Xva.r1.to_numpy()
        scores["rule_78bar_momentum"] = Xva.cum77.to_numpy()

    sub = train.sample(min(max_train, len(train)), random_state=seed)
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=500,
                                         random_state=seed)
    hgb.fit(hand_features(store, sub), sub.label)
    print(f"HGB baseline: {tree_size(hgb):,} tree nodes ({hgb.n_iter_} trees, {Xva.shape[1]} features)")
    scores["hgb_hand_features"] = hgb.predict_proba(Xva)[:, 1]
    return scores, prior, hgb
