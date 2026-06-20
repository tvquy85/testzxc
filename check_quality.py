import json
p = 'data/rationales/raw/current_clean_train_qwen3_4b_v3.jsonl'
rows = []
with open(p, 'r', encoding='utf-8') as f:
    for line in f:
        rows.append(json.loads(line))
        
non_empty_news = []
non_empty_tech = []
for r in rows:
    try:
        out = json.loads(r['raw_output'])
        if len(out.get('news_rationale', [])) > 0:
            non_empty_news.append(r)
        if len(out.get('technical_rationale', [])) > 0:
            non_empty_tech.append(r)
    except:
        pass

print(f"Total processed: {len(rows)}")
print(f"Non-empty news rationale: {len(non_empty_news)}")
print(f"Non-empty technical rationale: {len(non_empty_tech)}")

if non_empty_news:
    print("\n--- Sample with news rationale ---")
    print(non_empty_news[0]['raw_output'])
    
if non_empty_tech:
    print("\n--- Sample with technical rationale ---")
    print(non_empty_tech[0]['raw_output'])
