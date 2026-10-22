"""The experiment runner expands grids correctly and routes model fields into the override."""

from experiments.run import argv_for, expand


def test_grid_expansion_and_routing():
    cfg = {"name": "t", "common": {"steps": 10, "override": {"d_model": 64}},
           "variants": {"mamba": {"model": "mamba", "override": {"layers": "MM"}},
                        "tf": {"model": "transformer", "override": {"layers": "A"}}},
           "grid": {"d_state": [4, 16], "lr": [1e-3], "seed": [0, 1]}}
    runs = dict(expand(cfg))
    assert len(runs) == 2 * 2 * 2
    a = runs["mamba_d_state4_seed1"]                      # only varying keys appear in the name
    assert a["override"] == {"d_model": 64, "layers": "MM", "d_state": 4}
    assert a["lr"] == 1e-3 and a["seed"] == 1 and a["steps"] == 10 and a["model"] == "mamba"
    argv = argv_for("x", a, "out")
    assert argv[:2] == ["-m", "training.train_lm"] and "--override" in argv and "--d_state" not in argv
