"""Audit COOHAVIOR's S1 robot stations against Merom_1_int, then write the shared-spawn layouts.

Two things, in order, both CPU-only (baked maps + scene json + object metadata):

1. For every one of V4's fifteen staged robot poses (audit_index.json, all set
   counts are prefixes of one route), test the robot's footprint against the
   oriented footprint of every object V4's filter keeps. Found four inside
   furniture: kitchen jackal and drone in furniture_sink_czyfhq_0, dining jackal
   and ridgeback in breakfast_table_skczfi_0.
2. Measure the most open room -- no-object traversability minus kept-object
   footprints, eroded by the largest robot (a 0.7 m Jackal, square kernel) --
   and pack all fifteen ground/air robots around the centroid of its largest free
   region, nearest-first in route order, so sets_<k> is the first 3k of sets_5.
   Crazyflies spawn at z = 1.2 m directly above their own Jackal.

Writes coop2/team_layouts/s1/sets_{1..5}.json. Re-running reproduces them.

    python feasibility_verify/make_s1_shared_spawn_layouts.py
"""
import json, math, os, sys, collections
import numpy as np, cv2
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); SCENE="Merom_1_int"
audit=json.load(open("/home/zixuanwe/Desktop/COOHAVIOR/benchmark_audit/audit_index.json"))
# ---- stations from audit: check all set counts agree with the sets=5 record
recs={r["robot_set_count"]:r for r in audit["figures"] if r["task_id"]=="S1-V4-HH"}
ROUTE=recs[5]["route_station_order"]                       # ['M5','M3','M2','M1','M4']
def mobile(rec): return {r["prim"]:(round(r["xyz_world_m"][0],4),round(r["xyz_world_m"][1],4)) for r in rec["robot_positions_world_m"] if r["kind"]!="crx10ial"}
full=mobile(recs[5]); consistent=True
for k in range(1,6):
    sub=mobile(recs[k]); consistent &= all(full[p]==xy for p,xy in sub.items()) and set(sub)=={p for p in full if p.rsplit('_',1)[-1].upper() in ROUTE[:k]}
for tid in ("S1-V4-LL","S1-V4-LH","S1-V4-HL"):
    r5=[r for r in audit["figures"] if r["task_id"]==tid and r["robot_set_count"]==5][0]; consistent &= mobile(r5)==full
print("audit: sets 1..5 are prefixes of the route and identical across the four S1 tasks:", consistent)
# ---- scene objects that survive V4's filter, with world xy AABBs
scene=json.load(open(f"{ROOT}/datasets/behavior-1k-assets/scenes/{SCENE}/json/{SCENE}_best.json"))
info=scene["objects_info"]["init_info"]; reg=scene["state"]["registry"]["object_registry"]
edits=json.load(open(f"{ROOT}/coop2/team_layouts/v4_s1_v4_ll_scene.json"))
gone=set(edits["deactivate"]); moves=edits.get("move",{})
def yaw(q): x,y,z,w=q; return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
boxes=[]
for n,i in info.items():
    a=i["args"]; cat=a.get("category")
    if n in gone or cat in ("floors","walls","ceilings","window","door") or n not in reg: continue
    meta=f"{ROOT}/datasets/behavior-1k-assets/objects/{cat}/{a['model']}/misc/metadata.json"
    if not os.path.exists(meta): continue
    m=json.load(open(meta)); bb=np.array(m["bbox_size"])*np.array(a.get("scale",[1,1,1])); off=np.array(m.get("base_link_offset",[0,0,0]))*np.array(a.get("scale",[1,1,1]))
    pos=np.array(moves.get(n, reg[n]["root_link"]["pos"])); th=yaw(reg[n]["root_link"]["ori"])
    c,s=math.cos(th),math.sin(th); R=np.array([[c,-s],[s,c]])
    ctr=pos[:2]+R@off[:2]; hx,hy=bb[0]/2,bb[1]/2
    corners=np.array([R@np.array([sx*hx,sy*hy]) for sx in(-1,1) for sy in(-1,1)])+ctr
    zmin=pos[2]+off[2]-bb[2]/2; zmax=pos[2]+off[2]+bb[2]/2
    boxes.append((n,cat,corners[:,0].min(),corners[:,0].max(),corners[:,1].min(),corners[:,1].max(),zmin,zmax,a.get("in_rooms")))
print(f"kept objects with footprints: {len(boxes)}")
# ---- robots: footprint half-extents at layout scale (from inspect_scene's WxD)
HALF={"ridgeback_franka":0.25,"jackal":0.35,"crazyflie":0.12}; HEIGHT={"ridgeback_franka":0.57,"jackal":0.70,"crazyflie":0.24}
def overlaps(xy,h,kind):
    hits=[]
    for b in boxes:
        n,cat,x0,x1,y0,y1,z0,z1,rooms=b
        if x0-h<xy[0]<x1+h and y0-h<xy[1]<y1+h and z0<HEIGHT[kind]+0.05 and z1>0.02:
            hits.append((n,cat,round(min(xy[0]-x0,x1-xy[0]),2),round(min(xy[1]-y0,y1-xy[1]),2)))
    return hits
kinds={p:p.split("_s1_")[0] for p in full}
print("\n=== station robots vs kept furniture (xy AABB overlap, robot footprint inflated) ===")
problems={}
for p,xy in sorted(full.items(), key=lambda kv:(kv[0].rsplit('_',1)[-1],kv[0])):
    hits=overlaps(xy,HALF[kinds[p]],kinds[p])
    flag="  <-- INSIDE/OVERLAPPING" if hits else ""
    print(f"  {p:26} {xy}  {[h[0] for h in hits]}{flag}")
    if hits: problems[p]=hits

# ================= relocation =================
from feasibility_verify.segroom import build, room
seg=build(SCENE,0.1)
base=f"{ROOT}/datasets/behavior-1k-assets/scenes/{SCENE}/layout"
trav=cv2.imread(f"{base}/floor_trav_no_obj_0.png",cv2.IMREAD_GRAYSCALE); H=trav.shape[0]; RES=0.01
def to_px(xy): return int(xy[1]/RES+H/2), int(xy[0]/RES+H/2)   # (row, col) -- same flip as world_to_map
def to_xy(rc): return ((rc[1]-H/2)*RES,(rc[0]-H/2)*RES)
occ=np.zeros_like(trav,dtype=np.uint8)                    # kept furniture, painted at 1 cm
for n,cat,x0,x1,y0,y1,z0,z1,rooms in boxes:
    if z1<0.02: continue
    r0,c0=to_px((x0,y0)); r1,c1=to_px((x1,y1)); occ[max(r0,0):r1+1,max(c0,0):c1+1]=1
MARGIN=0.05; SEP={"ridgeback_franka":0.25,"jackal":0.35,"crazyflie":0.12}
def free_mask(kind):
    h=HALF[kind]+MARGIN; k=cv2.getStructuringElement(cv2.MORPH_RECT,(2*int(h/RES)+1,)*2)   # square footprint: a circle under-protects the corners
    floor=cv2.erode((trav>0).astype(np.uint8),k); blocked=cv2.dilate(occ,k)
    return (floor==1)&(blocked==0)
# ================= spawn everyone together in the most open room =================
ROOMS=["living_room_0","bedroom_0","childs_room_0","kitchen_0","dining_room_0","corridor_0"]
inv={v:k for k,v in seg["names"].items()}
seg1=build(SCENE,0.01)                      # room ids at map resolution, for masking
inv1={v:k for k,v in seg1["names"].items()}
big=free_mask("jackal")                    # floor eroded by the largest footprint, minus kept furniture
print("\n=== admissible floor per room for a 0.7 m robot (m2) ===")
areas={}
for rname in ROOMS:
    if rname not in inv1: continue
    m=big&(seg1["ins"]==inv1[rname]); areas[rname]=m.sum()*RES*RES
    print(f"  {rname:16} {areas[rname]:6.2f}")
spawn_room=max(areas,key=areas.get); print("most open:", spawn_room)
mask=big&(seg1["ins"]==inv1[spawn_room])
num,lab,stats,cents=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
comp=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA])); mask=(lab==comp)
cx,cy=cents[comp]; centre=to_xy((cy,cx)); print(f"largest free component {stats[comp,cv2.CC_STAT_AREA]*RES*RES:.2f} m2, centroid {tuple(round(c,2) for c in centre)}")
rr,cc=np.where(mask); cells=np.stack([rr,cc],1)
GAP=0.15
order=[f"{kind}_s1_{st.lower()}" for st in ROUTE for kind in ("ridgeback_franka","jackal","crazyflie")]
placed={}
HOVER_Z=1.2   # above a 0.70 m Jackal and a 0.57 m Ridgeback, well under any ceiling
dist_c=np.hypot(*(cells-np.array([cy,cx])).T)*RES
idx=np.argsort(dist_c)
ground=[p for p in order if kinds[p]!="crazyflie"]
for p in ground:
    h=HALF[kinds[p]]
    for i in idx:
        cxy=to_xy(cells[i])
        if all(max(abs(cxy[0]-q[0]),abs(cxy[1]-q[1]))>=h+HALF[kinds[o]]+GAP for o,q in placed.items()):
            placed[p]=(round(cxy[0],3),round(cxy[1],3)); break
    else: raise SystemExit(f"could not place {p}")
# each drone hovers directly above its own team's Jackal
for p in order:
    if kinds[p]=="crazyflie":
        st=p.rsplit("_",1)[-1]; placed[p]=placed[f"jackal_s1_{st}"]
print("\n=== packed spawn (prefix order = route order; drones share their Jackal's x,y at z=%.1f) ===" % HOVER_Z)
for p in order: print(f"  {p:26} {placed[p]}  {math.hypot(placed[p][0]-centre[0],placed[p][1]-centre[1]):.2f} m from centroid")
bad=[(p,overlaps(xy,HALF[kinds[p]],kinds[p])) for p,xy in placed.items() if overlaps(xy,HALF[kinds[p]],kinds[p])]
print("furniture overlap after packing:", bad if bad else "none")
# ================= layouts =================
MODEL={"ridgeback_franka":("v4_ridgeback_ur5",0.5,True),"jackal":("v4_jackal",0.7,False),"crazyflie":("v4_crazyflie_cf2x",2.0,False)}
outdir=f"{ROOT}/coop2/team_layouts/s1"; os.makedirs(outdir,exist_ok=True)
for k in range(1,6):
    robots=[]
    for i,p in enumerate(order[:3*k]):
        kind=kinds[p]; model,scale,lock=MODEL[kind]
        pos=list(placed[p])+([HOVER_Z] if kind=="crazyflie" else [])
        # Teams are team_1..team_k in order of appearance and robots are
        # <type>_<team ordinal>, the same in every S1 layout (user,
        # 2026-09-13). The V4 set (m5, m3, ...) survives only in _v4_prim.
        ordinal=(i//3)+1
        e={"name":f"{ {'ridgeback_franka':'ridgeback','jackal':'jackal','crazyflie':'drone'}[kind] }_{ordinal}",
           "model":model,"position":pos,"team":f"team_{ordinal}",
           "_v4_prim":p,"_v4_staged_position":list(full[p]),"scale":scale}
        if lock: e["base_locked_while_holding"]=True
        robots.append(e)
    doc={"_comment":[
      f"COOHAVIOR S1 robots, {k} set{'s' if k>1 else ''} ({3*k} robots), for Merom_1_int -- all spawned together.",
      "",
      f"Everyone starts in {spawn_room}, packed around the centroid of its largest free",
      "region (decision: user, 2026-09-12). The room is the most open one measured as",
      "traversable floor (no-object map) minus the footprints of every object V4 keeps,",
      "eroded by the largest robot footprint; admissible area per room, m2: "
      + ", ".join(f"{r} {a:.1f}" for r,a in sorted(areas.items(), key=lambda kv:-kv[1])) + ".",
      "Ground vehicles are placed nearest-to-centroid first in V4's route order M5, M3,",
      "M2, M1, M4 with pairwise clearance = sum of half-footprints + 0.15 m, so sets_<k>",
      "is exactly the first 3k robots of sets_5 and every smaller team stays compact.",
      "Each Crazyflie spawns hovering at z = 1.2 m directly above its own team's Jackal",
      "-- [x, y, z] in position -- so a trio is born in one spot; a holonomic base keeps",
      "its z through every navigate, so it stays airborne.",
      "",
      "Teams are still V4's trios -- one Ridgeback+UR5 arm (base-locked while holding),",
      "one Jackal carrier, one Crazyflie -- named by the station they came from",
      "(_v4_prim, _v4_staged_position). Those staged poses are NOT used: four of the",
      "fifteen sit inside furniture V4 keeps (kitchen jackal and drone in",
      "furniture_sink_czyfhq_0, dining jackal and ridgeback in breakfast_table_skczfi_0),",
      "and the plan changed to a shared spawn before that was worth fixing per station.",
      "",
      "Scales: ridgeback 0.5 and jackal 0.7 are the values v4_s1_v4_ll was verified",
      "with (the staging USD says 1.0 / 0.5, unresolved); crazyflie 2.0 by decision."],
      "robots":robots}
    json.dump(doc,open(f"{outdir}/sets_{k}.json","w"),indent=2); open(f"{outdir}/sets_{k}.json","a").write("\n")
    print(f"wrote s1/sets_{k}.json  ({len(robots)} robots)")
