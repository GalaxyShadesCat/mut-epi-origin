from scripts.grid_search.results import compute_derived_fields


def test_compute_derived_fields_example():
    row = {
        "correct_celltypes": "mela",
        "best_celltype_local_score": "mela",
        "mutations_pre_downsample": 4364,
        "mutations_post_downsample": 4364,
        "rf_perm_importances_mean_json": '{"dnase_mela": 0.9, "gc": 0.1}',
    }
    derived = compute_derived_fields(row)
    assert derived["is_correct_local_score"] is True
    assert derived["downsample_applied"] is False
    assert derived["downsample_ratio"] == 1.0
    assert derived["rf_top_feature_perm"] == "dnase_mela"
    assert derived["rf_top_feature_importance_perm"] == 0.9
    assert derived["rf_top_is_dnase"] is True
