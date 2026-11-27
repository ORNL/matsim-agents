from matsim_agents.core.restart import KeyedResultLog


def test_saved_results_come_back_by_key(tmp_path):
    log = KeyedResultLog(tmp_path / "sub" / "done.jsonl")
    log.save("a", {"energy": 1.5})
    log.save("b", {"energy": 2.5})
    assert log.load() == {"a": {"energy": 1.5}, "b": {"energy": 2.5}}


def test_missing_file_has_no_results(tmp_path):
    assert KeyedResultLog(tmp_path / "none.jsonl").load() == {}


def test_a_later_entry_replaces_an_earlier_one(tmp_path):
    log = KeyedResultLog(tmp_path / "done.jsonl")
    log.save("a", 1)
    log.save("a", 2)
    assert log.load() == {"a": 2}


def test_a_write_cut_short_is_ignored_and_the_rest_kept(tmp_path):
    path = tmp_path / "done.jsonl"
    log = KeyedResultLog(path)
    log.save("a", 1)
    with path.open("a") as handle:
        handle.write('{"key": "trunc')
    assert log.load() == {"a": 1}


def test_revive_rebuilds_results_and_bad_ones_are_skipped(tmp_path):
    log = KeyedResultLog(tmp_path / "done.jsonl")
    log.save("a", {"n": 1})
    log.save("b", {"wrong": 1})
    assert log.load(lambda result: result["n"] * 10) == {"a": 10}
