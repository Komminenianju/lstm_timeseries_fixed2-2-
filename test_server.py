"""
Run this FIRST to check if basic Flask POST works on your machine.
Open: http://localhost:5000/test
Then click the button — if it works, Flask is fine.
If it fails, the problem is Flask itself.
"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

from flask import Flask, jsonify, render_template_string
import time, threading

app = Flask(__name__)

HTML = """
<!DOCTYPE html><html><body style="font-family:monospace;padding:40px;background:#111;color:#0f0">
<h2>Flask POST Test</h2>
<button onclick="test1()" style="padding:10px 20px;margin:8px;background:#222;color:#0f0;border:1px solid #0f0;cursor:pointer">
  Test 1: Instant POST (should work)
</button>
<button onclick="test2()" style="padding:10px 20px;margin:8px;background:#222;color:#0f0;border:1px solid #0f0;cursor:pointer">
  Test 2: 10-second POST (simulates training)
</button>
<button onclick="test3()" style="padding:10px 20px;margin:8px;background:#222;color:#0f0;border:1px solid #0f0;cursor:pointer">
  Test 3: Background job + poll (new approach)
</button>
<div id="log" style="margin-top:20px;padding:16px;background:#0a0a0a;border:1px solid #333;min-height:100px"></div>
<script>
function log(msg, ok=true){
  const d = document.getElementById('log');
  const t = new Date().toLocaleTimeString();
  d.innerHTML += `<div style="color:${ok?'#0f0':'#f55'}">[${t}] ${msg}</div>`;
}

async function test1(){
  log('Sending instant POST...');
  try{
    const r = await fetch('/api/test-instant', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'});
    const d = await r.json();
    log('✓ Instant POST works: ' + JSON.stringify(d));
  }catch(e){ log('✗ FAILED: ' + e.message, false); }
}

async function test2(){
  log('Sending 10-second POST (will this fail?)...');
  try{
    const r = await fetch('/api/test-slow', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'});
    const d = await r.json();
    log('✓ Slow POST works: ' + JSON.stringify(d));
  }catch(e){ log('✗ FAILED on slow POST: ' + e.message, false); }
}

let pollTimer;
async function test3(){
  log('Starting background job...');
  try{
    const r = await fetch('/api/test-job', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'});
    const d = await r.json();
    log('Job started: ' + d.job_id);
    pollTimer = setInterval(async()=>{
      const s = await fetch('/api/test-poll/'+d.job_id);
      const sd = await s.json();
      log('Poll: ' + JSON.stringify(sd));
      if(sd.status === 'done'){ clearInterval(pollTimer); log('✓ Background job complete!'); }
    }, 2000);
  }catch(e){ log('✗ FAILED: '+e.message, false); }
}
</script>
</body></html>
"""

JOBS = {}

@app.route('/test')
def test_page():
    return render_template_string(HTML)

@app.route('/api/test-instant', methods=['POST'])
def test_instant():
    return jsonify({"ok": True, "msg": "instant response"})

@app.route('/api/test-slow', methods=['POST'])
def test_slow():
    time.sleep(10)
    return jsonify({"ok": True, "msg": "waited 10 seconds"})

@app.route('/api/test-job', methods=['POST'])
def test_job():
    import uuid
    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {"status": "running", "step": 0}
    def worker():
        for i in range(5):
            time.sleep(2)
            JOBS[job_id]["step"] = i+1
        JOBS[job_id]["status"] = "done"
    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"job_id": job_id})

@app.route('/api/test-poll/<job_id>')
def test_poll(job_id):
    job = JOBS.get(job_id, {})
    return jsonify({"status": job.get("status","unknown"), "step": job.get("step",0)})

if __name__ == '__main__':
    print("\n  Open: http://localhost:5000/test\n")
    app.run(debug=False, port=5000, threaded=True)
