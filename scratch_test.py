import json
import pandas as pd
from src.llm.rationale_schema import RationaleOutput

df = pd.read_json('data/rationales/candidate_rationales_h1.jsonl', lines=True)

def parse_better(text):
    if not isinstance(text, str): return None
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        js = text[start:end+1]
        try:
            d = json.loads(js)
            d['action'] = d.get('action', '').lower()
            if d['action'] == 'buy': d['action'] = 'long'
            if d['action'] == 'sell': d['action'] = 'short'
            return d
        except:
            return None
    return None

errors = []
def validate(d):
    if not d: return False
    try:
        RationaleOutput(**d)
        return True
    except Exception as e:
        errors.append(str(e))
        return False

df['new_json'] = df['raw_text'].apply(parse_better)
df['new_schema_ok'] = df['new_json'].apply(validate)

error_counts = pd.Series(errors).value_counts()
print(error_counts.head(10))
