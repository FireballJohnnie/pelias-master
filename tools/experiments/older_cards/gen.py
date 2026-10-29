# provenance: original. Masterwork 21.8 tables (DERIVED). Run from this folder: python3 gen.py nv|dc|amd|intel|old|sm|cls
exec(open('model.py').read().split("print('| Card")[0])
exec(open('smallmoe.py').read().split("sel=[")[0].split("exec(open('model.py')")[1].split("\n",1)[1])
C={n:(v,b,ven,note) for n,v,b,ven,note in cards}
def moe_row(n):
    v,b,ven,note=C[n]; p,h,lo,hi=moe(v,b)
    return '| %s | %s | %s | %s | %s | %d-%d |'%(n,v,b,p,'—' if h is None else '%.2f'%h,round(lo),round(hi))
def dense_row(n):
    v,b,ven,note=C[n]; S=v-0.6
    # a dense model filling the card, and a 1 GB model; overhead for a ~28-layer model (o x 28/40)
    f=lambda gb:(1/(gb/(E_LO*b)+O_LO*28/40),1/(gb/(E_HI*b)+O_HI*28/40))
    a=f(S); c=f(1.0)
    return '| %s | %s | %s | %.1f GB | %d-%d | %d-%d |'%(n,v,b,S,a[0],a[1],c[0],c[1])
def sm_row(n):
    v,b,ven,note=C[n]; cells=[]
    for cn,T,A in classes:
        p,h,lo,hi=run(v,b,T,A)
        cells.append('—' if p=='no' else ('%d-%d (all)'%(round(lo),round(hi)) if p=='all' else '%d-%d (h %.2f)'%(round(lo),round(hi),h)))
    return '| %s | %s | %s | %s |'%(n,v,b,' | '.join(cells))
import sys
sec=sys.argv[1]
H='| Card | VRAM GB | GB/s | Placement | h | tok/s |\n|---|---|---|---|---|---|'
if sec=='nv': print(H); [print(moe_row(n)) for n in ['RTX 2060','RTX 2080 Ti','RTX 3060 12GB','RTX 3070','RTX 3080 10GB','RTX 3090','RTX 4060','RTX 4060 Ti 16GB','RTX 4070 (this PC, stock memory)','RTX 4090','RTX 5060','RTX 5060 Ti 16GB','RTX 5070','RTX 5090']]
if sec=='dc': print(H); [print(moe_row(n)) for n in ['Tesla P40','Tesla P100 16GB','Tesla T4','Tesla V100 16GB','Tesla V100 32GB','Instinct MI25','Instinct MI50 16GB','Instinct MI50 32GB / MI60']]
if sec=='pro': print(H); [print(moe_row(n)) for n in ['TITAN Xp','TITAN V','TITAN RTX','Quadro P6000','Quadro RTX 8000','RTX A4000','RTX A5000','RTX A6000','RTX 6000 Ada','Tesla M40 24GB']]
if sec=='amd': print(H); [print(moe_row(n)) for n in ['RX 5700 XT','RX 6600','RX 6700 XT','RX 6800','RX 7600','RX 7800 XT','RX 7900 XTX','RX 9060 XT 16GB','RX 9070 XT']]
if sec=='intel': print(H); [print(moe_row(n)) for n in ['Arc A380','Arc A750','Arc A770 16GB','Arc B570','Arc B580']]
if sec=='old':
    print('| Card | VRAM GB | GB/s | Largest dense model that fits | tok/s at that size | tok/s, a 1 GB model |\n|---|---|---|---|---|---|')
    [print(dense_row(n)) for n in ['GTX 960 4GB','GTX 970','GTX 980 Ti','GTX 1050 Ti','GTX 1060 6GB','GTX 1070','GTX 1080 Ti','RX 570 4GB','RX 580 8GB','Arc A380']]
if sec=='sm':
    print('| Card | VRAM GB | GB/s | '+' | '.join(c[0]+' class' for c in classes)+' |\n|---|---|---|---|---|---|---|')
    [print(sm_row(n)) for n in ['GTX 970','GTX 1050 Ti','RX 570 4GB','GTX 980 Ti','GTX 1060 6GB','RTX 2060','Arc A380','GTX 1070','RX 580 8GB','RX 6600','RX 7600','RTX 4060','RTX 3070','RTX 5060','Arc A750','GTX 1080 Ti','Tesla P40','Tesla V100 16GB']]
if sec=='cls':
    for cn,T,A in classes: print('| %s | %.1f | %.2f | %.2f |'%(cn,BPT_T*T,BPT_A*A,DSH*BPT_A*A))
if sec=='chk':
    t=DENSE/(E_HI*552)+0.765*EXP/(E_HI*552)+(1-0.765)*EXP/BCPU+3.2e-3; print('%.1f'%(1/t))
    print('igpu %.0f-%.0f'%(1/(2.27/(E_LO*89.6)+O_LO),1/(2.27/(E_HI*89.6)+O_HI)))
