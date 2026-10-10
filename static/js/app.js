function saveResult(d){fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)}).then(r=>r.json()).then(x=>alert(x.ok?'Recommendation saved to your prediction history.':'Could not save.')).catch(()=>alert('Could not save.'));}
setTimeout(()=>document.querySelectorAll('.flash').forEach(x=>x.remove()),5000);
