# provenance: original. Masterwork 21.8 tables (DERIVED). Run from this folder: python3 gen.py nv|dc|amd|intel|old|sm|cls
exec(open('model.py').read().split("print('| Card")[0])
BPT_T = 22.43/35.0   # bytes per total parameter in urm_TG2 (DERIVED: file / nominal 35B)
BPT_A = 2.27/3.0     # bytes per active parameter per token (DERIVED: R3 token table / nominal 3B active)
DSH = 1.74/2.27      # dense share of the active bytes (ASSUMPTION: carried over from the 35B)
RES2 = 0.6           # VRAM reserve for a small model (ASSUMPTION)
# hit-rate curve as a share of the expert bytes held (35B: expert bytes = 22.43 - 1.74 = 20.69 GB)
EXPB = 22.43-1.74
fcurve=[(0,0),(2*1.0737/EXPB,0.43),(4*1.0737/EXPB,0.63),(6*1.0737/EXPB,0.76)]
def hitf(f):
    for (a,ha),(b,hb) in zip(fcurve,fcurve[1:]):
        if f<=b: return ha+(hb-ha)*(f-a)/(b-a)
    return 0.76
classes=[('7B-A1.3B',6.9,1.3),('16B-A2.4B',15.7,2.4),('21B-A3.6B',21.0,3.6),('30B-A3.3B',30.5,3.3)]
def run(v,b,T,A):
    WT=BPT_T*T; WA=BPT_A*A; D=DSH*WA; X=WA-D
    if WT+RES2<=v:
        return 'all',1.0,1/(WA/(E_LO*b)+O_LO),1/(WA/(E_HI*b)+O_HI)
    bank=v-D-RES2
    if bank<0.3: return 'no',0,0,0
    h=hitf(bank/(WT-D))
    t=lambda e,o: D/(e*b)+h*X/(e*b)+(1-h)*X/BCPU+o
    return '%.1f GB'%bank,h,1/t(E_LO,O_LO),1/t(E_HI,O_HI)
sel=['GTX 970','GTX 980 Ti','GTX 1050 Ti','GTX 1060 6GB','GTX 1070','GTX 1080 Ti','RX 570 4GB','RX 580 8GB','RTX 2060','RTX 3070','RTX 4060','RTX 5060','RX 6600','RX 7600','Arc A380','Arc A750','Tesla P40','Tesla V100 16GB']
print('| Card | VRAM | GB/s |'+'|'.join(c[0] for c in classes)+'|')
for n,v,b,ven,note in cards:
    if n not in sel: continue
    cells=[]
    for cn,T,A in classes:
        p,h,lo,hi=run(v,b,T,A)
        cells.append('—' if p=='no' else ('%d-%d (all in VRAM)'%(round(lo),round(hi)) if p=='all' else '%d-%d (h %.2f)'%(round(lo),round(hi),h)))
    print('|%s|%s|%s|%s|'%(n,v,b,'|'.join(cells)))
for cn,T,A in classes: print(cn,'file %.1f GB, active %.2f GB, dense %.2f'%(BPT_T*T,BPT_A*A,DSH*BPT_A*A))
# old-card generic: tok/s for a dense model of S GB, and the largest that fits
print()
for n,v,b,ven,note in cards:
    if n not in ['GTX 960 4GB','GTX 970','GTX 980 Ti','GTX 1050 Ti','GTX 1060 6GB','GTX 1070','GTX 1080 Ti','RX 570 4GB','RX 580 8GB','Arc A380','Tesla P40','Tesla V100 32GB']: continue
    S=v-0.6
    print('|%s|%s|%s|%.1f GB|%d-%d|%d-%d|'%(n,v,b,S,round(E_LO*b/S),round(E_HI*b/S),round(E_LO*b/1),round(E_HI*b/1)))
