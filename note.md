## 4.26

多少mse loss对应可行的logits loss？

一个step到底能有多低的logits loss？

直接给答案的训练：

logits loss:1.1524810791015625; mse loss: 0.08501794934272766

logits loss:0.9525314569473267;mse loss: 0.07577991485595703

说明mse和logits之间没有必然联系，mse很大的时候logits loss可以变得很小。

尝试VQ-VAE? 多层codebook正好和mamba的结构一样。

64*64第一次测试：

logits_loss: 4.232490539550781; mse_loss: 0.06567761301994324

进一步证明mse与logits没有多少联系。

尝试了不同的采样步数，100的mse低但是1000的logits更低一点。

直接预测target mse卡在0.12，logits loss是2.7左右。

同上但是logits loss来训练，1.7左右，eval 2.7，老是梯度爆炸，原因不明。

## 4.27

v_prediction并不是预测x0，之前的实验都错了。

小t loss大是很常见的现象，尝试用min snr来解决。

加入min snr后mse要到0.04左右才可以，但是鉴于mse和logits的关系，这也不一定。

logits_loss: 4.160210609436035; mse_loss: 0.040410395711660385

snr没有帮助，logits loss会随着mse下降而上升。

测试sample prediction直接给答案：logits_loss: 1.22; mse_loss: 0.08

用mamba mse训练预测sample：logits_loss: 4.18; mse_loss: 0.054

用mamba一步预测target，mse：logits_loss: 2.75; mse_loss: 0.12

## 4.29

训练一个mse的sample diffuser，研究一下loss关于时间步的分布，是不是扩散模型加噪的问题。

sample：logits_loss: 3.619229316711426; mse_loss: 0.025557005777955055 感觉进步了一点点。

logits_loss: 4.0986647605896; mse_loss: 0.09623508155345917

发现并没有所谓的突然变容易，loss和noise占比的关系是比较连续的。

v_prediction: logits_loss: 1.0539296865463257; mse_loss: 0.1725289523601532

v_prediction的完整训练：logits_loss: 3.836989402770996; mse_loss: 0.12368296831846237

## 5.5