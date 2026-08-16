import argparse, os, time
import torch
import torch.nn.functional as F
from data.dataset import make_loaders
from models.integrated_model import IntegratedRestorationModel


def gradient_loss(pred,target):
    sx=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=pred.dtype,device=pred.device).view(1,1,3,3)
    sy=torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=pred.dtype,device=pred.device).view(1,1,3,3)
    return F.l1_loss(F.conv2d(pred,sx,padding=1),F.conv2d(target,sx,padding=1))+F.l1_loss(F.conv2d(pred,sy,padding=1),F.conv2d(target,sy,padding=1))

def frequency_loss(pred,target):
    return F.l1_loss(torch.abs(torch.fft.rfft2(pred,norm='ortho')),torch.abs(torch.fft.rfft2(target,norm='ortho')))

def restoration_loss(pred,target):
    l1=F.l1_loss(pred,target); grad=gradient_loss(pred,target); freq=frequency_loss(pred,target)
    return l1+0.1*grad+0.01*freq,l1,grad,freq


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--lr-dir',required=True); ap.add_argument('--gt-dir',required=True)
    ap.add_argument('--output',default='checkpoints/best_integrated_model.pth')
    ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--batch-size',type=int,default=8)
    args=ap.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader,val_loader=make_loaders(args.lr_dir,args.gt_dir,batch_size=args.batch_size)
    model=IntegratedRestorationModel().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,betas=(0.9,0.99),weight_decay=1e-6)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=1e-6)
    best=-float('inf'); os.makedirs(os.path.dirname(args.output) or '.',exist_ok=True)
    start=time.perf_counter()
    for epoch in range(args.epochs):
        model.train(); train_total=0
        for lr,gt in train_loader:
            lr,gt=lr.to(device,non_blocking=True),gt.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True); pred=model(lr); loss,_,_,_=restoration_loss(pred,gt)
            if not torch.isfinite(loss): raise RuntimeError(f'Non-finite loss at epoch {epoch+1}')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); train_total+=loss.item()
        model.eval(); psnr=[]; val_loss=0
        with torch.no_grad():
            for lr,gt in val_loader:
                lr,gt=lr.to(device,non_blocking=True),gt.to(device,non_blocking=True); pred=model(lr).clamp(0,1)
                val_loss+=restoration_loss(pred,gt)[0].item(); mse=((pred-gt)**2).mean((1,2,3)); psnr.extend((10*torch.log10(1/(mse+1e-10))).cpu().tolist())
        val_loss/=len(val_loader); mean_psnr=sum(psnr)/len(psnr); scheduler.step()
        print(f'Epoch [{epoch+1:02d}/{args.epochs}] Train: {train_total/len(train_loader):.6f} Val: {val_loss:.6f} PSNR: {mean_psnr:.5f} dB LR: {scheduler.get_last_lr()[0]:.2e}')
        if mean_psnr>best:
            best=mean_psnr; torch.save({'epoch':epoch+1,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict(),'psnr':best,'val_loss':val_loss},args.output); print(f'  NEW BEST: {best:.5f} dB')
    print(f'Best PSNR: {best:.5f} dB | time: {(time.perf_counter()-start)/60:.2f} min')

if __name__=='__main__': main()
