"""Offline replica of SegmentationMap.get_room_instance_by_point at a given resolution."""
import cv2, numpy as np
import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATS=[l.rstrip() for l in open(os.path.join(ROOT, "datasets", "behavior-1k-assets", "metadata", "room_categories.txt"))]
def build(scene, res):
    base=os.path.join(ROOT, "datasets", "behavior-1k-assets", "scenes", scene, "layout")
    i0=cv2.imread(f"{base}/floor_insseg_0.png", cv2.IMREAD_GRAYSCALE)
    s0=cv2.imread(f"{base}/floor_semseg_0.png", cv2.IMREAD_GRAYSCALE)
    H=i0.shape[0]; ms=int(H*0.01/res)
    ins=cv2.resize(i0,(ms,ms),interpolation=cv2.INTER_NEAREST)
    sem=cv2.resize(s0,(ms,ms),interpolation=cv2.INTER_NEAREST)
    s2i={}
    for i in np.unique(ins):
        if i==0: continue
        x,y=np.where(ins==i); s2i.setdefault(sem[x[0],y[0]],[]).append(i)
    names={}
    for s,ids in s2i.items():
        for k,i in enumerate(ids): names[i]=f"{CATS[s-1]}_{k}"
    return dict(ins=ins,names=names,ms=ms,res=res,scene=scene)
def room(m, xy):
    q=np.array(xy,dtype=float)/m["res"]+m["ms"]/2.0; x,y=int(q[1]),int(q[0])
    if not (0<=x<m["ms"] and 0<=y<m["ms"]): return None
    v=m["ins"][x,y]; return m["names"].get(int(v)) if v!=0 else None
def free_after_erosion(scene, rooms, radius=0.62, mapname="floor_trav_no_obj_0.png"):
    base=os.path.join(ROOT, "datasets", "behavior-1k-assets", "scenes", scene, "layout")
    m=build(scene,0.01); trav=cv2.imread(f"{base}/{mapname}", cv2.IMREAD_GRAYSCALE)
    inv={v:k for k,v in m["names"].items()}
    rp=int(np.ceil(radius/0.01)); k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*rp+1,2*rp+1))
    out={}
    for r in rooms:
        if r not in inv: out[r]=None; continue
        mask=((m["ins"]==inv[r])&(trav>0)).astype(np.uint8)
        out[r]=(round(mask.sum()*1e-4,2), round(cv2.erode(mask,k).sum()*1e-4,2))
    return out
