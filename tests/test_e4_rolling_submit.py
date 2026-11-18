from benchmarks.e4.aurora_dft.rolling_submit import parse_qstat_full, parse_qstat_job_ids


def test_parse_qstat_job_ids_counts_only_current_users_jobs():
    text = """
Job ID Username Queue Jobname SessID NDS TSK Memory Time S Time
-------------------- -------- -------- ---------- ------ --- --- ------ ----- - -----
9090503.aurora-pbs-* mlupopa capacity e4-fcc* 738263 1 208 -- 101:0 R 15:48
9092467.aurora-pbs-* mlupopa capacity e4-phos* -- 2 416 -- 25:00 Q --
12345.aurora-pbs-* other capacity other -- 1 208 -- 01:00 Q --
"""
    assert parse_qstat_job_ids(text, "mlupopa") == {
        "9090503.aurora-pbs-*",
        "9092467.aurora-pbs-*",
    }


def test_parse_qstat_full_distinguishes_success_failure_and_active():
    assert parse_qstat_full("    job_state = F\n    Exit_status = 0\n") == ("F", 0)
    assert parse_qstat_full("    job_state = F\n    Exit_status = 1\n") == ("F", 1)
    assert parse_qstat_full("    job_state = R\n") == ("R", None)
