# experiment record

## baseline model
cnn2 lr:1e-3 epoch:200 \
Train metrics, acc:  0.6997, loss:  0.7611 \
Eval metrics, acc:  0.4553, loss:  4.0620

mcldnn lr:1e-3 epoch:200 \
Train metrics, acc:  0.7119, loss:  0.8353 \
Eval metrics, acc:  0.6048, loss:  1.5642

petcgdnn lr:1e-3 epoch:200 \
Train metrics, acc:  0.9878, loss:  0.0251 \
Eval metrics, acc:  0.5369, loss:  4.1933

## vanilla KD(T=4, a=0.9)
cnn2 lr:1e-3 epoch:200 \
learn from:petcgdnn \
Train metrics, acc:  0.5430, loss:  1.2382 \
Eval metrics, acc:  0.5056, loss:  1.7962 

learn from:mcldnn \
Train metrics, acc:  0.5420, loss:  0.4417 \
Eval metrics, acc:  0.5044, loss:  1.4805 

## DIST(a=1, b=2, y=2) 
cnn2 lr:1e-3 epoch:200 \

learn from:petcgdnn \
Train metrics, acc:  0.6558, loss:  0.6858 \
Eval metrics, acc:  0.4907, loss:  1.5319

learn from:mcldnn \
Train metrics, acc:  0.6117, loss:  0.5229 \
Eval metrics, acc:  0.5062, loss:  1.3935

## DIST(a=1, b=1, y=1) 

learn from:mcldnn \
Train metrics, acc:  0.6105, loss:  0.6717 \
Eval metrics, acc:  0.5067, loss:  1.3979

(a=2, b=1, y=1) 
- Train metrics, acc:  0.4666, loss:  1.1003
- Eval metrics, acc:  0.4640, loss:  1.4001

## DIST (normalized KD div)
cnn2 lr:1e-3 epoch:200 \

learn from:mcldnn \ lr = 1e-3
Train metrics, acc:  0.5927, loss:  0.1994 \
Eval metrics, acc:  0.4902, loss:  1.4801

## Sim-KD
cnn2 lr:1e-3 epoch:200 \

learn from:petcgdnn \
Train metrics, loss:  0.0911 \
student - Eval metrics, acc:  0.4414, loss:  3.1710

learn from:mcldnn \ lr = 1e-3
Train metrics, loss:  0.0731 \
Eval metrics, acc:  0.2698, loss:  4.8282

\ lr = 2e-3
Train metrics, loss:  0.0882 \
Eval metrics, acc:  0.2425, loss:  5.2740