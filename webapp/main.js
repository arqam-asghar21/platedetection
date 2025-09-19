const drop = document.getElementById('drop');
const fileInput = document.getElementById('file');
const btnChoose = document.getElementById('btnChoose');
const btnDetect = document.getElementById('btnDetect');
const spinner = document.getElementById('spinner');
const displayImage = document.getElementById('displayImage');
const toast = document.getElementById('toast');

let currentImage = null;

function setLoading(v){ document.body.classList.toggle('loading', v); spinner.style.display = v ? 'block' : 'none'; }

function readFileAsImage(file){
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = reader.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function prepareImageForUpload(file){
  const img = await readFileAsImage(file);
  const maxSide = Math.max(img.width, img.height);
  const maxTarget = 1200; // reduce further to be safe
  const scale = maxSide > maxTarget ? maxTarget / maxSide : 1;
  const cw = Math.round(img.width * scale);
  const ch = Math.round(img.height * scale);
  const off = document.createElement('canvas');
  off.width = cw; off.height = ch;
  const octx = off.getContext('2d');
  octx.drawImage(img, 0, 0, cw, ch);
  const dataUrl = off.toDataURL('image/jpeg', 0.82); // slightly lower quality
  return await (await fetch(dataUrl)).blob();
}

async function loadFileToCanvas(file){
  const img = await readFileAsImage(file);
  currentImage = img;
  // Show original image
  displayImage.src = img.src;
}

drop.addEventListener('click', ()=> fileInput.click());
btnChoose.addEventListener('click', (e)=>{ e.preventDefault(); fileInput.click(); });

;['dragenter','dragover'].forEach(evt=>drop.addEventListener(evt, (e)=>{ e.preventDefault(); drop.classList.add('dragover'); }));
;['dragleave','drop'].forEach(evt=>drop.addEventListener(evt, (e)=>{ e.preventDefault(); drop.classList.remove('dragover'); }));
drop.addEventListener('drop', async (e)=>{ const f = e.dataTransfer.files[0]; if(f) await loadFileToCanvas(f); });
fileInput.addEventListener('change', async ()=>{ const f = fileInput.files[0]; if(f) await loadFileToCanvas(f); });

btnDetect.addEventListener('click', async (e)=>{
  e.preventDefault();
  const f = fileInput.files[0];
  if(!f){ alert('Please choose an image first.'); return; }
  const blob = await prepareImageForUpload(f);
  const fd = new FormData(); fd.append('file', blob, 'upload.jpg');
  setLoading(true);
  try{
    const res = await fetch('/detect', { method:'POST', body: fd });
    if(!res.ok){
      const t = await res.text();
      throw new Error(t || 'Upload failed');
    }
    const data = await res.json();
    // Show blurred result in the same image container
    displayImage.src = `data:image/jpeg;base64,${data.image_base64}`;
    showToast('Detection complete!');
  } catch(err){
    console.error(err);
    showToast('Could not process this image. Try a different format or smaller image.');
  } finally {
    setLoading(false);
  }
});

function showToast(message){
  if(!toast) return;
  toast.textContent = message;
  toast.classList.add('show');
  setTimeout(()=> toast.classList.remove('show'), 2500);
}


