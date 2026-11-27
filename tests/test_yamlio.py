import pytest

from matsim_agents.core.yamlio import load_yaml_mapping, parse_yaml_mapping


def test_loads_a_mapping(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("a: 1\nb: [2, 3]\n")
    assert load_yaml_mapping(path) == {"a": 1, "b": [2, 3]}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("a: [unclosed\n", "invalid YAML"),
        ("- 1\n- 2\n", "expected a mapping"),
        ("", "expected a mapping"),
    ],
)
def test_bad_documents_raise_value_error_naming_the_source(text, message):
    with pytest.raises(ValueError, match=message) as error:
        parse_yaml_mapping(text, "where.yaml")
    assert "where.yaml" in str(error.value)
