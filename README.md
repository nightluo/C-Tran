<!--
 * @Author: 1872194982@qq.com 1872194982@qq.com
 * @Date: 2022-12-05 15:51:03
 * @LastEditors: 1872194982@qq.com 1872194982@qq.com
 * @LastEditTime: 2022-12-07 11:15:41
 * @FilePath: \C-Tran\project\C-Tran\README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
**General Multi-label Image Classification with Transformers**<br/>
Jack Lanchantin, Tianlu Wang, Vicente Ordóñez Román, Yanjun Qi<br/>
Conference on Computer Vision and Pattern Recognition (CVPR) 2021<br/>
[[paper]](https://arxiv.org/abs/2011.14027) [[poster]](https://github.com/QData/C-Tran/blob/main/supplemental/ctran_poster.pdf) [[slides]](https://github.com/QData/C-Tran/blob/main/supplemental/ctran_slides.pdf)
<br/>


## Training and Running C-Tran ##

Python version 3.7 is required and all major packages used and their versions are listed in `requirements.txt`.

### C-Tran on COCO80 Dataset ###
Download COCO data (19G)
```
wget http://cs.virginia.edu/yanjun/jack/vision/coco.tar.gz
mkdir -p data/
tar -xvf coco.tar.gz -C data/
```

Train New Model
```
python3 main.py  --batch_size 16  --lr 0.00001 --optim 'adam' --layers 3  --dataset 'coco' --use_lmt --dataroot /mnt/data/luoyan/coco/

CUDA_VISIBLE_DEVICES=5,6 python3 main.py  --batch_size 16  --lr 0.00001 --optim 'adam' --layers 3  --dataset 'coco' --use_lmt --dataroot /mnt/data/luoyan/coco/

python3 main.py  --batch_size 16  --lr 0.00001 --optim 'adam' --layers 3  --dataset 'coco' --use_lmt --dataroot /home/yuez/night/dataset/

python3 main.py  --batch_size 16  --lr 0.00001 --optim 'adam' --layers 3  --dataset 'coco' --use_lmt --dataroot E:/Datasets/coco/
```


### C-Tran on VOC20 Dataset ###
Download VOC2007 data (1.7G)
```
wget http://cs.virginia.edu/yanjun/jack/vision/voc.tar.gz
mkdir -p data/
tar -xvf voc.tar.gz -C data/
```

Train New Model
```
python main.py  --batch_size 16  --lr 0.00001 --optim 'adam' --layers 3  --dataset 'voc' --use_lmt --grad_ac_step 2 --dataroot data/

python main.py  --batch_size 16  --lr 0.00001 --optim 'adam' --layers 3  --dataset 'voc' --use_lmt --grad_ac_step 2 --dataroot E:/Datasets/

python main.py  --batch_size 16  --lr 0.00001 --optim 'adam' --layers 3  --dataset 'voc' --use_lmt --grad_ac_step 2 --dataroot /mnt/data/luoyan/

python main.py  --batch_size 16  --lr 0.00001 --optim 'adam' --layers 3  --dataset 'voc' --use_lmt --grad_ac_step 2 --dataroot /home/yuez/night/dataset
```


## Citing ##

```bibtex
@article{lanchantin2020general,
  title={General Multi-label Image Classification with Transformers},
  author={Lanchantin, Jack and Wang, Tianlu and Ordonez, Vicente and Qi, Yanjun},
  journal={arXiv preprint arXiv:2011.14027},
  year={2020}
}
```
