const ws = new WebSocket(`wss://${location.host}/ws`);
const log = document.getElementById('log');
const form = document.getElementById('form');
const input = document.getElementById('input');

function append(msg, cls='bot'){
  const div = document.createElement('div');
  div.className = cls;
  div.textContent = msg;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

ws.onmessage = (ev)=>{
  const data = JSON.parse(ev.data);
  append(`[${data.intent}] ${JSON.stringify(data)}`);
};

form.addEventListener('submit', e=>{
  e.preventDefault();
  const text = input.value.trim();
  if(!text) return;
  append(text, 'user');
  ws.send(text);
  input.value='';
});