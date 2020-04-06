## Architecture Variants

In this work, we provide a set of variants to validate our design choices employed in DAVO. 
There are version names of these variants:

| Version                   | Description                                              |
| ------------------------- | -------------------------------------------------------- |
| v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_flow-abs_flow-fc_tanh | DAVO |
| v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-no_segmask | Ours w/o attention |
| v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-static | Ours w/ static_attention | 
| v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-no_segmask-se_insert | Ours w/ feature attention |
| v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_seg_wo_tgt-fc_tanh | DAVO (segmentation source) |
| v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_depth_wo_tgt_to_seg-fc_tanh | DAVO (depth source) |
| v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_rgb_wo_tgt_to_seg-fc_tanh | DAVO (rgb source) |

you can quickly switch the variant by choosing the version name from them, or make your own variant by selecting 
components listed below while setting the command `--version` in the training phase and the testing phase.


## Architecture Components

#### Choose the input of PoseNN:

* v0 : only RGB frames.
* v1 : the concatenation of RGB frames and optical flows.

    #### [additonally concatenate segmentation id maps]
    * -seglabelid

#### Choose learning rate decay:

* -decay50k
* -decay100k
* -decay50k-staircase
* -decay100k-staircase

#### Choose PoseNN: 

* -dilatedPoseNN
* -dilatedCouplePoseNN
* -sharedNN-dilatedPoseNN
* -sharedNN-dilatedCouplePoseNN

    #### [adjust conv6 channels in PoseNN]
    * -cnv6_<num_channels>
  
    #### [adjust se_block between conv5 and conv6 in PoseNN]
    * -se_insert
    * -se_replace
    * -se_skipadd
    
#### Choose Attention mode:

* -segmask_all
* -segmask_rgb
* -no_segmask

#### Choose AttentionNN source and layer types:

* -se_flow
* -se_spp_flow
* -se_seg
* -se_spp_seg
* -se_mixSegFlow
* -se_spp_mixSegFlow
* -se_SegFlow_to_seg
* -se_app_mixSegFlow
* -se_seg_wo_tgt
* -se_depth_wo_tgt_to_seg
* -se_rgb_wo_tgt_to_seg

    #### [adjust the activated function in AttentionNN]
    * -fc_relu
    * -fc_tanh
    * -fc_lrelu
