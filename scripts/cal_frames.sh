python -c "import json; 
p='/path/to/meta/episodes.jsonl'; 
rows=[json.loads(l) for l in open(p) if l.strip()]; 
print('episodes=', len(rows), 'frames=', sum(r['length'] for r in rows), 'avg_len=', sum(r['length'] for r in rows)/len(rows))"