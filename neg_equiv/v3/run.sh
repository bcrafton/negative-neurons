
# python36 t4.py --lr 0.1    --eps 0.00001 --gpu 0 > r1 &
# python36 t4.py --lr 0.01   --eps 0.00001 --gpu 1 > r2 &
# python36 t4.py --lr 0.001  --eps 0.00001 --gpu 2 > r3 &
# python36 t4.py --lr 0.0001 --eps 0.00001 --gpu 3 > r4 &

wait

# python36 t4.py --lr 0.1    --eps 0.000001 --gpu 0 > r5 &
# python36 t4.py --lr 0.01   --eps 0.000001 --gpu 1 > r6 &
# python36 t4.py --lr 0.001  --eps 0.000001 --gpu 2 > r7 &
# python36 t4.py --lr 0.0001 --eps 0.000001 --gpu 3 > r8 &

wait 

python36 t4.py --lr 0.001  --eps 0.000001 --gpu 0 > r9 &
python36 t4.py --lr 0.001  --eps 0.000001 --gpu 1 > r10 &

