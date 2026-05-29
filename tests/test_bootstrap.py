def test_package_imports():
    import infineon_baseline
    assert hasattr(infineon_baseline, "__version__")

def test_validate_sequence_accessible_via_package():
    """sys.path nudge in __init__ must make generate_sequences importable AND callable."""
    from infineon_baseline import validate_sequence, Violation
    result = validate_sequence(["RECEIVE WAFER LOT", "SHIP LOT"])
    # This pair is missing every required block — must surface at least one violation.
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(v, Violation) for v in result)
