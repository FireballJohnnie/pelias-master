# provenance: original. Masterwork 21.8 tables (DERIVED). Run from this folder: python3 gen.py nv|dc|amd|intel|old|sm|cls
# Older/smaller cards: DERIVED projections from Pelias's measured numbers (see the proposal for each input's source)
E_LO, E_HI = 259/552, 494/552     # R3 v1 streaming / read ceiling, as fractions of the 4070's peak (MEASURED on the 4070)
O_LO, O_HI = 3.2e-3, 0.5e-3      # per-token overhead: today (DERIVED, game plan) / after the R3 sync fixes (DERIVED, Fable: 160 boundaries at a few us)
DENSE, EXP = 1.74, 0.532          # GB read per token: dense + head, routed experts (DERIVED, R3 token table)
FILE = 22.43                      # urm_TG2 file, GB (MEASURED)
BCPU = 42.5                       # GB/s, CPU cold-expert path on DDR4-3200 dual channel (40-45, game plan)
RES = 1.0                         # GB of VRAM kept for KV cache, workspace, desktop (ASSUMPTION, stated)
curve = [(0,0),(2,0.43),(4,0.63),(6,0.76)]  # hot-bank GiB -> hit rate (DERIVED, simulated on the router trace)
def hit(g):
    g=min(g,6.0)
    for (a,ha),(b,hb) in zip(curve,curve[1:]):
        if g<=b: return ha+(hb-ha)*(g-a)/(b-a)
    return 0.76
def moe(vram,bw):
    if vram >= FILE + RES:  # whole file resident
        lo = 1/(DENSE+EXP)/1 if False else 1/((DENSE+EXP)/(E_LO*bw)+O_LO)
        hi = 1/((DENSE+EXP)/(E_HI*bw)+O_HI)
        return 'all in VRAM',None,lo,hi
    bank=(vram-DENSE-RES)/1.0737  # GB -> GiB
    if bank<0.5: return 'does not fit',None,0,0
    h=hit(bank)
    def t(e,o): return DENSE/(e*bw)+h*EXP/(e*bw)+(1-h)*EXP/BCPU+o
    return 'bank %.1f GiB'%bank,h,1/t(E_LO,O_LO),1/t(E_HI,O_HI)
cards = [
 # name, VRAM GB, GB/s, vendor, note
 ('GTX 960 4GB',4,112,'N','sm_52, no dp4a'),('GTX 970',3.5,224,'N','sm_52, no dp4a; 4 GB card, 3.5 GB fast segment used here'),
 ('GTX 980 Ti',6,336.5,'N','sm_52, no dp4a'),('GTX 1050 Ti',4,112,'N','sm_61'),('GTX 1060 6GB',6,192,'N','sm_61'),
 ('GTX 1070',8,256,'N','sm_61'),('GTX 1080 Ti',11,484,'N','sm_61'),('RTX 2060',6,336,'N','sm_75'),('RTX 2080 Ti',11,616,'N','sm_75'),
 ('RTX 3060 12GB',12,360,'N','sm_86'),('RTX 3070',8,448,'N','sm_86'),('RTX 3080 10GB',10,760,'N','sm_86'),('RTX 3090',24,936,'N','sm_86'),
 ('RTX 4060',8,272,'N','sm_89'),('RTX 4060 Ti 16GB',16,288,'N','sm_89'),('RTX 4070 (this PC, stock memory)',12,504,'N','sm_89'),('RTX 4090',24,1008,'N','sm_89'),
 ('RTX 5060',8,448,'N','sm_120'),('RTX 5060 Ti 16GB',16,448,'N','sm_120'),('RTX 5070',12,672,'N','sm_120'),('RTX 5090',32,1792,'N','sm_120'),
 ('Tesla P40',24,346,'N','sm_61, data centre'),('Tesla P100 16GB',16,732,'N','sm_60, no dp4a, data centre'),('Tesla T4',16,320,'N','sm_75, data centre'),
 ('Tesla V100 16GB',16,900,'N','sm_70, data centre'),('Tesla V100 32GB',32,900,'N','sm_70, data centre'),
 ('RX 570 4GB',4,224,'A','GCN 4'),('RX 580 8GB',8,256,'A','GCN 4'),('RX 5700 XT',8,448,'A','RDNA 1'),('RX 6600',8,224,'A','RDNA 2'),
 ('RX 6700 XT',12,384,'A','RDNA 2'),('RX 6800',16,512,'A','RDNA 2'),('RX 7600',8,288,'A','RDNA 3'),('RX 7800 XT',16,624,'A','RDNA 3'),
 ('RX 7900 XTX',24,960,'A','RDNA 3'),('RX 9060 XT 16GB',16,320,'A','RDNA 4'),('RX 9070 XT',16,640,'A','RDNA 4'),
 ('Instinct MI25',16,484,'A','Vega 10, data centre'),('Instinct MI50 16GB',16,1024,'A','Vega 20, data centre'),('Instinct MI50 32GB / MI60',32,1024,'A','Vega 20, data centre'),
 ('TITAN Xp',12,547.6,'N','sm_61'),('TITAN V',12,651.3,'N','sm_70, HBM2'),('TITAN RTX',24,672,'N','sm_75'),
 ('Quadro P6000',24,432.8,'N','sm_61'),('Quadro RTX 8000',48,672,'N','sm_75'),('RTX A4000',16,448,'N','sm_86'),('RTX A5000',24,768,'N','sm_86'),
 ('RTX A6000',48,768,'N','sm_86'),('RTX 6000 Ada',48,960,'N','sm_89'),('Tesla M40 24GB',24,288,'N','sm_52, no dp4a'),
 ('Arc A380',6,186,'I','Alchemist'),('Arc A750',8,512,'I','Alchemist'),('Arc A770 16GB',16,560,'I','Alchemist'),('Arc B570',10,380,'I','Battlemage'),('Arc B580',12,456,'I','Battlemage'),
]
print('| Card | VRAM | GB/s | 35B-A3B (urm_TG2) placement | hit | tok/s (DERIVED) |')
for n,v,b,ven,note in cards:
    p,h,lo,hi=moe(v,b)
    hs='%.2f'%h if h is not None else '—'
    rng='%d-%d'%(round(lo),round(hi)) if hi>0 else '—'
    print('|%s|%s|%s|%s|%s|%s|%s|'%(n,v,b,p,hs,rng,note))
# tiny dense models, fully resident: bytes = params * bpw / 8
print()
models=[('0.6B',0.6),('1.7B',1.7),('4B',4.0),('8B',8.0),('14B',14.0)]
for bpw,lab in [(7.53,'Q7_ZT'),(4.5,'4.5-bit (Q4_TG2 class)')]:
    print('##',lab)
    for n,v,b,ven,note in cards:
        row=[]
        for mn,p in models:
            gb=p*bpw/8
            if gb+0.6<=v:
                row.append('%s %d-%d'%(mn,round(E_LO*b/gb),round(E_HI*b/gb)))
        print(n,'|',' ; '.join(row[-2:]))
# check: the 4070 at 552 with the game plan's rung 1
p,h,lo,hi=moe(12,552); print('4070 @552:',p,h,round(lo),round(hi))
t=DENSE/(E_HI*552)+0.765*EXP/(E_HI*552)+(1-0.765)*EXP/BCPU+3.2e-3; print('rung-1 check (e hi, o 3.2ms, h .765): %.1f tok/s'%(1/t))
