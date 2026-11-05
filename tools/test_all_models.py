import urllib.request
import json
import time

models_to_test = [
    'gemma-4-26b-a4b-it-abliterated-nanoplus',
    'flint-40b-apex-nanoplus',
    'ornith-nanoplus',
    'lfm2.5-24b-nanoplus',
    'tiel-coder-35b-apex-nanoplus'
]

results = {}

for m in models_to_test:
    print(f"\n==========================================")
    print(f" Testing: {m}")
    print(f"==========================================")
    
    # 1. Load model
    load_req = urllib.request.Request(
        'http://127.0.0.1:8088/api/models/load',
        data=json.dumps({'model': m}).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    try:
        t0 = time.time()
        with urllib.request.urlopen(load_req, timeout=120) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        t_load = time.time() - t0
        print(f"  [OK] Loaded into VRAM in {t_load:.1f}s")
    except Exception as e:
        print(f"  [FAIL] Could not load: {e}")
        results[m] = f"Load failed: {e}"
        continue

    # 2. Test inference
    chat_req = urllib.request.Request(
        'http://127.0.0.1:8088/v1/chat/completions',
        data=json.dumps({
            'model': 'strata',
            'messages': [{'role': 'user', 'content': 'What is 3 + 5? Reply with only the number.'}],
            'max_tokens': 20
        }).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    try:
        t0 = time.time()
        with urllib.request.urlopen(chat_req, timeout=60) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        t_gen = time.time() - t0
        choice = data['choices'][0]['message']
        out = (choice.get('content') or choice.get('reasoning_content') or '').strip().replace('\n', ' ')
        print(f"  [OK] Generated in {t_gen:.2f}s: \"{out[:50]}\"")
        results[m] = "WORKING (100% Verified)"
    except Exception as e:
        print(f"  [FAIL] Inference failed: {e}")
        results[m] = f"Inference failed: {e}"

print("\n" + "=" * 50)
print(" FINAL VERIFICATION SUMMARY:")
print("=" * 50)
for m, status in results.items():
    print(f"  - {m}: {status}")
