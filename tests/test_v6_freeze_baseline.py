import json
from pathlib import Path

def test_v6_freeze_report_exists():
    p=Path('outputs/repro/v6_freeze_medium_baseline.json')
    assert p.exists()
    data=json.loads(p.read_text(encoding='utf-8'))
    assert not data['failures']
    assert len(data['files'])>=3
