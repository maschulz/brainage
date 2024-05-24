from causalml.match import NearestNeighborMatch, create_table_one, MatchOptimizer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import QuantileTransformer
import pandas as pd
from shap import LinearExplainer
import shap
from typing import Optional, Literal, List


def propensity_score_matching(
    df: pd.DataFrame,
    treatment_col: List[str],
    match_cols: List[str],
    mask_col: Optional[str] = None,
    quantile_transform: Optional[Literal["normal", "uniform"]] = None,
    caliper: float = 0.25,
    random_state: int = 42,
):
    print(mask_col)
    assert (mask_col is None) or (df[mask_col].dtype == bool), "mask_col must be boolean"
    assert df[treatment_col].dtype == bool, "treatment_col must be boolean"
    
    if mask_col is not None:
        # Exclude samples that are masked, keep treated samples
        # df_ = df[(~df[mask_col]) ^ df[treatment_col]] # this is broken somehow
        df_=df[ ((df[mask_col] == False) & (df[treatment_col] == False)) | (df[treatment_col] == True) ]

        print(f"number of control samples: {len(df[df[treatment_col] == False])}, masked: {len(df_[df_[treatment_col] == False])}")
    else:
        df_ = df

    # print(f"number of control samples: {len(df[df[treatment_col] == False])}, masked: {len(df_[df_[treatment_col] == False])}")
    
    df_ = df_.copy()

    # Convert treatment column to binary
    df_[treatment_col] = (df_[treatment_col] == True).astype(int)
    # Convert match columns to float
    df_[match_cols] = df_[match_cols].astype(float)
    # Drop samples with missing values
    df_ = df_.dropna(subset=match_cols)

    X = df_[match_cols].values
    y = df_[treatment_col].values

    # Transform features
    if quantile_transform:
        qt = QuantileTransformer(
            output_distribution=quantile_transform, random_state=random_state
        )
        X = qt.fit_transform(X)

    # Fit linear model
    pm = LinearRegression()
    pm.fit(X, y)
    df_["_propensity-score"] = pm.predict(X)

    # Explain
    explainer = LinearExplainer(pm, X, feature_names=match_cols)
    shap_values = explainer(
        X,
    )

    # Set caliper at 0.25 standard deviations of the propensity score
    caliper = df_["_propensity-score"].std() * caliper

    # Match
    psm = NearestNeighborMatch(caliper=caliper, random_state=random_state)
    matched = psm.match(
        data=df_,
        treatment_col=treatment_col,
        score_cols=[
            "_propensity-score",
        ],
    )

    # Create pre and post matching statistics
    stats_post = create_table_one(
        data=matched,
        treatment_col=treatment_col,
        features=match_cols
        + [
            "_propensity-score",
        ],
    )
    stats_post["version"] = "post-matching"
    stats_post.loc[match_cols, "SHAP value"] = shap_values.abs.mean(axis=0).values
    stats_pre = create_table_one(
        data=df_,
        treatment_col=treatment_col,
        features=match_cols
        + [
            "_propensity-score",
        ],
    )
    stats_pre["version"] = "pre-matching"
    stats_pre.loc[match_cols, "SHAP value"] = shap_values.abs.mean(axis=0).values

    # Create MultiIndex for stats dataframe
    stats = pd.concat([stats_pre, stats_post])
    stats.set_index(["version", stats.index], inplace=True)
    
    return matched, stats
