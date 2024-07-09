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

## Sim-KD
cnn2 lr:1e-3 epoch:200 \

learn from:petcgdnn \
Train metrics, loss:  0.0911 \
student - Eval metrics, acc:  0.4414, loss:  3.1710


