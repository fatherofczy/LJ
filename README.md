# tfm_baseline
此段实习主要尝试了模型结构，noisy label，VAE，截面模型等系列。
模型结构：
1.	模型结构系列
a)	首先尝试了MTSMixer, TSMixer, DLinear等，大体思路都是用全连接层替换transformer里的注意力层，但是都无法超过基准线baseline [https://www.notion.so/baseline-1332027a6f308038829ac1e24c741b69]
b)	后来尝试了很多结构上的探索，主要是针对之前实习proj里LSTM相关的扩展，比如参考Multiplicative LSTM，Mogrifier LSTM里面针对lstm结构的改动，都没什么显著稳定的提升，只能说在cherry pick上有更好的结果了，LSTM的结果意向都很volatile。后面也尝试了一些lstm和transformer的结合，也没有显著的提升。
c)	随后尝试了WITRAN[https://www.notion.so/WITRAN-4d5c6ff915644733ad2149a19fd927bf]，WITRAN结构是RNN结构的改进，显示地结合了长短时序信息WITRAN，显著打败了Baseline，后续就展开了很多对WITRAN的研究
i.	尝试将之前尝试过的LSTM系列里的Mogrifier LSTM，Multiplicative LSTM的结构去实验，发现并没有什么提升[https://www.notion.so/witran_lstmtrick-d6f2ed35fbe44109b1c494732471069a]
ii.	尝试结合WITRAN和tfm [https://www.notion.so/witran_tfm-1212027a6f30809b86eaed6f4fc56074]，在进入时序模型前，先对截面做注意力。也没有提升；尝试把witran里的卷积换成Conv结构，参数量相比naïve
的witran下降了90%，表现和原先的witran类似Conv_witran
iii.	将State Frequency model和WITARN结合，即在witran结构里面加入一些频率的模块，从趋势和频率两个角度进行信息提取，也没有提升SFM_witran，SFM_lstm
iv.	尝试了stacking的方法，即WITRAN_posneg，有比较好的效果，这里显示地将模型对涨/跌进行了学习
v.	Follow up上一个，探索了很多MOE,MMOE的策略，但是都没有很好的结果，目标是用更快的效率和更少的参数量达到类似的效果。witran_stack_single，witran_2_stack，这里需要很多精力去设计不同的模型结构MOE/MMOE
2.	Noisy label系列（以下模型主体结构都是WITRAN）
a)	这里首先尝试了使用一个小网络去给出样本权重，来对损失函数做加权计算，具体的实现方法模仿这篇论文：Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting。尝试过的网络结构主要是MLP+relu+MLP, 
i.	最naïve的方法是输入当前的loss，模型从loss里学习信息出来，比baseline略有提升。 WITRAN_meta
ii.	参考mentornet的思路，我发现在网络的输入里同时加入y_true, loss有更好的效果,比只加入当前训练的loss好一些。 witran_mentornet_arch
iii.	尝试了TRA: Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport，这里思路是利用不同的头的loss去对样本做路由，但是我尝试后发现并没有什么提升
b)	随后尝试了很多权重策略，除了DE基本都没什么提升：
i.	double ensemble,利用历史损失函数去衡量每个样本的权重，DE的优点是稳定训练曲线，但是没有什么结果上更大的提升；witran_expert3_double_ensemble，witran_expert3_double_ensemble_last_first
ii.	尝试每次训练drop掉损失最大的100个样本（每轮一共3759个样本），表现下降很大，说明loss大的样本可能携带很多信息；
iii.	每次训练drop掉loss最大的c%的样本，（c随着epoch增大而增大）；没什么提升 self-paced learning(WITRAN)
iv.	Mixing_label,将损失小的样本和损失大的样本混合 witran_expert3_mixing_label
v.	根据训练结果观察模型在行业，时间上的表现，对损失达的人为赋予更大的权重Industry_weight_WITRAN，Industry_weight_WITRAN
3.	VAE系列
a)	VAE系列的灵感来源于 LEARNING TO (LEARN AT TEST TIME)，模仿这篇文章的思路，尝试了在训练过程中交替进行VAE重构任务和预测任务的方法VAE_witran。这篇实验验证了VAE任务确实有额外的价值vae_baseline。此外，我在做交替优化的时候，做VAE重构任务时候会重新使用一个优化器；我发现如果是和预测任务共用一个优化器表现会降低很多VAE_share_optimizer；或者是在做回归任务时候，冻结VAE模型的参数不产生梯度，表现下降更多VAE_param_freeze
b)	后来尝试修改VAE模块的结构，比如使用TFM作为特征提取层VAE_channel_mix_tfm_witran，并没有提升；或者使用VAE对输入进行分类VAE_stack，将不同类的x路由到不同的结构，没什么提升；或者不进行路由VAE_single_arch，均没有看到提升
c)	后来尝试drop掉VAE里的KL loss项（这样就退化为了autoencoder），可以看到有明显的提升，说明KL的存在限制了模型的表达能力autoencoder，进一步尝试了分类版本的autoencoder，但是没看到显著的提升。尝试调整kl_loss项的系数（系数反映了对先验方差的估计），但是也没得到更好的结果

4.	截面模型系列
a)	模仿阿里巴巴的论文（CAN）CAN(L=64),输入时序数据，将特征的时序数据视作当前的嵌入，让不同特征进行交互后和原始特征concat起来送进TFM模块，表现不佳
b)	Fieldy Transformer(L=1)，FTTransformer(L=1)探究不同维度做注意力对结果的影响，表现都非常差，截面模型在这个时序任务上表现差很多

[https://www.notion.so/baseline-1332027a6f308038829ac1e24c741b69]
