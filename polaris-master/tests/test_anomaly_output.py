"""pytest framework for AnomalyOutput
"""

import json

from polaris.anomaly.anomaly_output import AnomalyOutput


def test_anomaly_output_json_serializable():
    """Test that anomaly Graph is JSON-serializable
    """

    output = AnomalyOutput()
    exported_from_json = json.loads(output.to_json())
    assert "metadata" in exported_from_json.keys()
    assert "data" in exported_from_json.keys()
    assert "timestamps" in exported_from_json['data'].keys()
    assert "values" in exported_from_json['data'].keys()
    assert "events" in exported_from_json['data'].keys()
