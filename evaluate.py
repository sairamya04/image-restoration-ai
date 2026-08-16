import argparse, os, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from models.integrated_model import IntegratedRestorationModel


def load_array(path):
    p=Path(path)
    if p.suffix.lower()=='.npy': return np.load(p).astype(np.float32)
    return np.asarray(Image.open(p).convert('L'),dtype=np.float32)/255.0

def save_array(arr,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.suffix.lower()=='.npy': np.save(path,arr.astype(np.float32))
    else: Image.fromarray(np.clip(arr,0,1)*255).convert('L').save(path)

def pair_name(p): return p.stem

def main():
    ap=argparse.ArgumentParser(description='KLA/SEMICON standalone image-restoration evaluator')
    ap.add_argument('--input-dir',required=True); ap.add_argument('--output-dir',required=True)
    ap.add_argument('--checkpoint',default='checkpoints/best_integrated_model.pth'); ap.add_argument('--gt-dir',default=None)
    args=ap.parse_args(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=IntegratedRestorationModel().to(device)
    ckpt=torch.load(args.checkpoint,map_location=device); model.load_state_dict(ckpt['model_state_dict']); model.eval()
    inp=Path(args.input_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    files=sorted([p for p in inp.iterdir() if p.suffix.lower() in {'.npy','.png','.jpg','.jpeg','.bmp','.tif','.tiff'}])
    if not files: raise RuntimeError('No supported test images found')
    metrics=[]; total=0
    for f in files:
        arr=load_array(f); original=arr.shape; x=torch.from_numpy(arr).float()
        if x.ndim==2: x=x[None,None]
        elif x.ndim==3: x=x[None]
        x=torch.nn.functional.interpolate(x,size=(128,128),mode='bicubic',align_corners=False).to(device)
        if device.type=='cuda': torch.cuda.synchronize()
        t=time.perf_counter()
        with torch.no_grad(): pred=model(x).clamp(0,1)
        if device.type=='cuda': torch.cuda.synchronize()
        total+=(time.perf_counter()-t)*1000
        pred_np=pred[0,0].cpu().numpy(); save_array(pred_np,out/f'{f.stem}.png')
        if args.gt_dir:
            gt_path=Path(args.gt_dir)/f'{f.stem}.npy'
            if not gt_path.exists():
                for ext in ['.png','.jpg','.jpeg','.bmp','.tif','.tiff']:
                    q=Path(args.gt_dir)/f'{f.stem}{ext}'
                    if q.exists(): gt_path=q; break
            if gt_path.exists():
                gt=load_array(gt_path); gt=torch.from_numpy(gt).float()[None,None]
                gt=torch.nn.functional.interpolate(gt,size=(256,256),mode='bilinear',align_corners=False)[0,0].numpy()
                mse=np.mean((pred_np-gt)**2); psnr=10*np.log10(1/(mse+1e-10)); ssim=structural_similarity(gt,pred_np,data_range=1.0); metrics.append((psnr,ssim))
    print(f'Processed: {len(files)} images'); print(f'Mean inference: {total/len(files):.3f} ms/image')
    if metrics: print(f'Mean PSNR: {np.mean([m[0] for m in metrics]):.6f} dB'); print(f'Mean SSIM: {np.mean([m[1] for m in metrics]):.6f}')

if __name__=='__main__': main()
