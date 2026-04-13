import requests
import time

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1)
session.mount('http://', adapter)

print('Testing persistent session (10 rapid requests)...')
for i in range(10):
    r = session.get('http://127.0.0.1:8080/state', timeout=5)
    s = r.json()['data']
    screen = s['screen']
    print(f'  [{i+1}] screen={screen} ok={r.ok}')
    time.sleep(0.02)

print()
print('Checking current screen state...')
s = session.get('http://127.0.0.1:8080/state', timeout=5).json()['data']
screen  = s['screen']
actions = s.get('available_actions')
print(f'  Screen:  {screen}')
print(f'  Actions: {actions}')
sel = s.get('selection') or {}
if sel:
    kind    = sel.get('kind')
    prompt  = sel.get('prompt', '')
    min_s   = sel.get('min_select')
    max_s   = sel.get('max_select')
    print(f'  Selection kind:   {kind}')
    print(f'  Prompt:           {prompt}')
    print(f'  min/max select:   {min_s} / {max_s}')

print()
print('No socket errors = connection fix is working')
