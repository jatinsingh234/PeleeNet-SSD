
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?W8???"@???"H???"Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2??8???@???H???XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?š@?šH?šXbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?ĝ@?ĝH?ĝXb*model/bbn_features_transition3_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8?ܱ@?ܱH?ܱXbHgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropInputhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2@B8???@???H???Xb*model/bbn_features_transition2_conv/Conv2Dhu  ?A
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2@B8???@???H???XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2 ?8???
@???
H???
Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?A
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2 ?8???
@???
H???
XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@??
H??
Xb.model/bbn_features_stemblock_stem1_conv/Conv2Dhdu  HB
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2??8š?@š?Hš?XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2!?8???@???H???XbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2??8?ĝ@?ĝH?ĝXb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2??8???@???H???XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???XbMgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8???@???H???XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8???@???H???XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8???@???H???Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???XbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8???@???H???XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8׾?@׾?H׾?Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb*model/bbn_features_transition4_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???XbLgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputh$u  HB
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2??8???@???H???XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu ??B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2??8???@???H???Xb*model/bbn_features_transition2_conv/Conv2Dhu ??B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?+8̌?@̌?Ȟ?Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2??8???@???H???XbHgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2??8???@???H???Xb*model/bbn_features_transition3_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2??8???@???H???XbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2??8???@???H???XbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2??8ɿ?@ɿ?Hɿ?XbMgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8?ʸ@?ʸH?ʸXb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, true, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2d8???@???H???bmodel/re_lu_113/Reluhu  ?B
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 6, 7, 5, 4, 5, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)V??* 28???@???H???XbQgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputhu  HB
c
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28???@??H??Xbmodel/ssd_cls4conv2/Conv2Dh?u  HB
c
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28ȋ?@??H??Xbmodel/ssd_cls1conv2/Conv2Dh?u  HB
c
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28???@??H??Xbmodel/ssd_cls3conv2/Conv2Dh?u  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xbmodel/ssd_cls1conv2/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8???@???H???Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8???@???H???Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 2?8???@?ؔH???XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???XbMgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2??8???@???H???XbMgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8ۯ?@ۯ?Hۯ?Xb?model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock3_denselayer1_branch2c_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??HXb?model/bbn_features_denseblock3_denselayer1_branch1b_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb)model/ssd_res_block2_branch1a_conv/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2,?8???@???H???XbLgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2,?8ڒ?@ڒ?Hڒ?Xb*model/bbn_features_transition4_conv/Conv2Dhu  ?B
m
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@???H???bmodel/re_lu_113/Reluhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2Dhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 6, 7, 3, 3, 5, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)M?2* 2?
8???@??wH??xXb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch1b_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch2c_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch1b_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock4_denselayer1_branch2c_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock4_denselayer1_branch1b_conv/Conv2Dhu  ?B
I
redzone_checker*?2?@8???@??H??bmodel/re_lu_114/Reluhu  ?B
`
redzone_checker*?2?@8???@??HXb)model/ssd_res_block4_branch1b_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block1_branch1b_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block5_branch1b_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block3_branch1b_conv/Conv2Dhu  ?B
I
redzone_checker*?2?@8???@??H??bmodel/re_lu_115/Reluhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
?void implicit_convolve_sgemm<float, float, 1024, 6, 7, 3, 3, 5, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)@?2* 2?8ؔ?@??oH??qXb*model/bbn_features_transition2_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@??H??Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
0cudnn_convolve_sgemm_sm35_ldg_nn_128x8x128x16x16??A*2?8???@???H???Xb.model/bbn_features_stemblock_stem1_conv/Conv2Dhu  B
?
redzone_checker*?2?@8???@??H??XbIgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbIgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropFilterhu  ?B
v
redzone_checker*?2?@8ո?@??H??Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?B
a
redzone_checker*?2?@8???@??H??Xb*model/bbn_features_transition3_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block1_branch1c_conv/Conv2Dhu  ?B
_
redzone_checker*?2?@8???@??H??Xb(model/ssd_res_block2_branch2_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2Dhu  ?B
_
redzone_checker*?2?@8Օ?@??H??Xb(model/ssd_res_block1_branch2_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock4_denselayer1_branch1a_conv/Conv2Dhu  ?B
a
redzone_checker*?2?@8֌?@??H??Xb*model/bbn_features_transition4_conv/Conv2Dhu  ?B
_
redzone_checker*?2?@8???@??H??Xb(model/ssd_res_block4_branch2_conv/Conv2Dhu  ?B
_
redzone_checker*?2?@8???@??H??Xb(model/ssd_res_block3_branch2_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block4_branch1c_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block3_branch1c_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 208???@??iH??kXbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???XbJgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void wgrad2d_grouped_direct_kernel<float, float, float, (cudnnTensorFormat_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnConvolutionStruct, cudnnFilterStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int)*?2?8???@???H???Xbbgradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
Q
redzone_checker*?2?@8?Π@??H??Xbmodel/ssd_cls4conv2/Conv2Dhu  ?B
Q
redzone_checker*?2?@8???@??H??Xbmodel/ssd_cls1conv2/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8Ԩ?@Ԩ?HԨ?Xb(model/ssd_res_block2_branch2_conv/Conv2Dhu  HB
Q
redzone_checker*?2?@8???@??H??Xbmodel/ssd_cls3conv2/Conv2Dhu  ?B
Q
redzone_checker*?2?@8???@??H??Xbmodel/ssd_box4conv2/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8Ԛ?@Ԛ?HԚ?Xb?model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2Dhu  HB
Q
redzone_checker*?2?@8???@??H??Xbmodel/ssd_cls5conv2/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu  ?B
]
redzone_checker*?2?@8???@??H??Xb&model/ssd_feature_extend2_conv1/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8?ڟ@??H??Xb)model/ssd_res_block1_branch1a_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8?ڟ@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch2c_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8?ן@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
v
redzone_checker*?2?@8?͟@??H??Xb?model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8?ɟ@??H??Xb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8?ǟ@??H??Xb?model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8?Ɵ@??H??XbKgradient_tape/model/ssd_res_block4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
v
redzone_checker*?2?@8?ş@??H??Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8?ß@??H??XbLgradient_tape/model/ssd_res_block4_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8ӿ?@??H??XbKgradient_tape/model/ssd_res_block5_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer6_branch2c_conv/Conv2D/Conv2DBackpropInputhu  ?B
v
redzone_checker*?2?@8???@??HXb?model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2Dhu  ?B
a
redzone_checker*?2?@8???@??H??Xb*model/bbn_features_transition2_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2Dhu  ?B

redzone_checker*?2?@8ղ?@??H??XbHgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropInputhu  ?B
v
redzone_checker*?2?@8׮?@??H??Xb?model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8Ԭ?@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropInputhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8ի?@??HXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8Ҫ?@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer6_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8ե?@??H??XbKgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2Dhu  ?B

redzone_checker*?2?@8???@??H??XbHgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropInputhu  ?B
Q
redzone_checker*?2?@8???@??HXbmodel/ssd_box1conv2/Conv2Dhu  ?B
v
redzone_checker*?2?@8Ԓ?@??H??Xb?model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block2_branch1a_conv/Conv2Dhu  ?B
e
redzone_checker*?2?@8???@??H??Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
_
redzone_checker*?2?@8???@??H??Xb(model/ssd_res_block5_branch2_conv/Conv2Dhu  ?B
Q
redzone_checker*?2?@8???@??H??Xbmodel/ssd_box5conv2/Conv2Dhu  ?B
Q
redzone_checker*?2?@8???@??H??Xbmodel/ssd_box3conv2/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block5_branch1a_conv/Conv2Dhu  ?B
]
redzone_checker*?2?@8???@??HXb&model/ssd_feature_extend3_conv1/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?
8???@???H???Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?
8???@???H???Xb?model/bbn_features_denseblock1_denselayer1_branch1b_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8Ө?@Ө?HӨ?Xb)model/ssd_res_block1_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 8, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)u?R* 28???@??[H??]XbMgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2?8???@?̉H???XbQgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)]?*28???@???H???XbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8?؉@?؉H?؉Xb?model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2Dhu  ?B
?
?void wgrad2d_grouped_direct_kernel<float, float, float, (cudnnTensorFormat_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnConvolutionStruct, cudnnFilterStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int)*?2?8???@???H???Xbbgradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2?8???@???H???XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8???@??TH??UXbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
redzone_checker*?2?@8???@??HXbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
f
redzone_checker*?2?@8???@??H??Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block5_branch1c_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  ?B
a
redzone_checker*?2?@8???@??H??Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8???@???H???XbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28???@???H???Xb*model/bbn_features_transition4_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28???@???H???XbLgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*2?8???@??QH??QXbQgradient_tape/model/bbn_features_stemblock_stem1_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2? 8???@ɊwH??wXbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8???@??tH??uXb*model/bbn_features_transition2_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8???@țtHȱtXbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8???@???H???XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
cvoid DSE::vector_fft<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)8 ??*2? 8???@??rH??rXb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8???@??qH??qXbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2 ?8???@??pH??pXbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28???@??oH??oXbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28???@???H???Xb*model/bbn_features_transition3_conv/Conv2Dhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8???@???H???Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?A
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28???@???H???XbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???XbJgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8???@??mH??mXbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
u
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xbmodel/ssd_cls5conv2/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xbagradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8ϟ?@ϟ?Hϟ?Xbagradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
redzone_checker*?2?@8???@??H¾Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8ϕ?@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8ѐ?@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8ѐ?@??H??XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8Ό?@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
s
redzone_checker*?2?@8???@??H??Xb<gradient_tape/model/ssd_box2conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
e
redzone_checker*?2?@8???@??HªXb.model/bbn_features_stemblock_stem1_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??HXbbgradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbJgradient_tape/model/ssd_res_block5_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
t
redzone_checker*?2?@8???@??H??Xb=gradient_tape/model/ssd_cls5conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H§Xbagradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block3_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
s
redzone_checker*?2?@8???@??H??Xb<gradient_tape/model/ssd_cls4conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbMgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block5_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
t
redzone_checker*?2?@8???@??H??Xb=gradient_tape/model/ssd_cls3conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbJgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??HXbLgradient_tape/model/ssd_res_block3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block5_branch1c_conv/Conv2D/Conv2DBackpropInputhu  ?B
t
redzone_checker*?2?@8???@??H??Xb=gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropFilterhu  ?B
f
redzone_checker*?2?@8???@??H??Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block4_branch1c_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??HXbJgradient_tape/model/ssd_res_block4_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block4_branch1c_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block4_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block5_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
s
redzone_checker*?2?@8???@??H??Xb<gradient_tape/model/ssd_box4conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block2_branch1c_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block2_branch1c_conv/Conv2D/Conv2DBackpropFilterhu  ?B
s
redzone_checker*?2?@8???@??H??Xb<gradient_tape/model/ssd_box5conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block3_branch1c_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
t
redzone_checker*?2?@8???@??H??Xb=gradient_tape/model/ssd_cls4conv2/Conv2D/Conv2DBackpropFilterhu  ?B
s
redzone_checker*?2?@8???@??H??Xb<gradient_tape/model/ssd_cls3conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block5_branch1c_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
s
redzone_checker*?2?@8???@??H??Xb<gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropInputhu  ?B
s
redzone_checker*?2?@8???@??H??Xb<gradient_tape/model/ssd_box3conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbMgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
s
redzone_checker*?2?@8???@??H??Xb<gradient_tape/model/ssd_cls5conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8ν?@??H??XbKgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbJgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block3_branch1c_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8ζ?@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbJgradient_tape/model/ssd_res_block3_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??HXbbgradient_tape/model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb(model/ssd_res_block1_branch2_conv/Conv2Dhu  HB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 8, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)|?R* 2X8???@??kH??kXbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2B8???@???H???Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
:cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_64x8x256x8x32}?R* 2Q8???@??iH??jXb*model/bbn_features_transition3_conv/Conv2Dhu  HB
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2? 8ϳ?@ϳ?Hϳ?XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2? 8???@???H???Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8???@???H???XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28???@??H??Xb=gradient_tape/model/ssd_cls3conv2/Conv2D/Conv2DBackpropFilterh?u  HB
?
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28Ϝ?@??H??Xb=gradient_tape/model/ssd_cls4conv2/Conv2D/Conv2DBackpropFilterh?u  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb<gradient_tape/model/ssd_cls3conv2/Conv2D/Conv2DBackpropInputhu  HB
?
?void sgemm_largek_lds64<false, true, 5, 5, 4, 4, 4, 32>(float*, float const*, float const*, int, int, int, int, int, int, float const*, float const*, float, float, int, int, int*, int*)*?!*28???@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb<gradient_tape/model/ssd_cls4conv2/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb<gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_c2r_32x32<float, true, false, 0u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)8 ??*?2 8???@??H??Xb.model/bbn_features_stemblock_stem1_conv/Conv2Dhdu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb<gradient_tape/model/ssd_cls5conv2/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8???@???H???Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
m
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@???H???bmodel/re_lu_114/Reluhu  HB
m
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@???H???bmodel/re_lu_115/Reluhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2Q8???@???H???Xb?model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2Dhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2? 8κ?@κ?Hκ?XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 28???@??AH??AXbMgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2Q8???@???H???Xb?model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb?model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2Q8???@???H???Xb?model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?κ@?κH?κXbagradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xbagradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8???@???H???Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8?ӹ@?ӹH?ӹXbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2B8???@???H???Xbbgradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8ͻ?@ͻ?Hͻ?XbHgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropInputhu  HB
?
?void sgemm_largek_lds64<false, true, 5, 5, 4, 4, 4, 32>(float*, float const*, float const*, int, int, int, int, int, int, float const*, float const*, float, float, int, int, int*, int*)*?!*28???@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@???H???XbHgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropInputhu  HB
?
?void sgemm_largek_lds64<false, true, 5, 5, 4, 4, 4, 32>(float*, float const*, float const*, int, int, int, int, int, int, float const*, float const*, float, float, int, int, int*, int*)*?!*28?޷@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?W8쭵@ı<H??<Xb.model/bbn_features_stemblock_stem1_conv/Conv2Dhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, true, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?28???@???H???bmodel/re_lu_115/Reluhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28́?@??<H??<XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28???@??;H??<Xb*model/bbn_features_transition1_conv/Conv2Dhu  HB
t
redzone_checker*?2?@8툴@??H??Xb=gradient_tape/model/ssd_box2conv2/Conv2D/Conv2DBackpropFilterh
u  ?B
?
redzone_checker*?2?@8???@??H??XbIgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropFilterh
u  ?B
t
redzone_checker*?2?@8?߳@??H??Xb=gradient_tape/model/ssd_box4conv2/Conv2D/Conv2DBackpropFilterh
u  ?B
?
redzone_checker*?2?@8?۳@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterh
u  ?B
?
redzone_checker*?2?@8?ճ@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterh
u  ?B
t
redzone_checker*?2?@8?ѳ@??H??Xb=gradient_tape/model/ssd_box3conv2/Conv2D/Conv2DBackpropFilterh
u  ?B
t
redzone_checker*?2?@8?ǳ@??H??Xb=gradient_tape/model/ssd_box5conv2/Conv2D/Conv2DBackpropFilterh
u  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2Q8콳@콳H콳Xb?model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?W8???@??;H??;Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  HB
I
redzone_checker*?2?@8???@??H??bmodel/re_lu_113/Reluh
u  ?B
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 5, 6, 4, 3, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)X?A*28???@???H???XbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8?ޱ@??XH??YXb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb?model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2Dhu  HB
?
?void DSE::regular_fft_pad<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??*2? 8???@??SHƬVXb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 28B8?Ȥ@?ȤH?ȤXbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2B8윤@윤H윤Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu  ?A
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb?model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???XbKgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?a8?ߣ@??QHŅRXb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)P?2* 28???@???H???XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xbmodel/ssd_cls3conv2/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???XbKgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2,8???@ƬOHźOXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8???@???H???Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, true, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2$8?ם@?םH?םbmodel/re_lu_114/Reluhu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8???@???H???Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)P?2* 28?ќ@?ќH?ќXbMgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?˛@?˛H?˛Xbagradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?ɛ@?ɛH?ɛXbagradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8?ǚ@??2H??4Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8릚@릚H릚Xb?model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2Dhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2B8?ę@?ęH?ęXbbgradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?
8???@???H???Xb?model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb)model/ssd_res_block2_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8꽗@??2H??2XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28̨?@??H??XbKgradient_tape/model/ssd_res_block4_branch2_conv/Conv2D/Conv2DBackpropFilterh?u  HB
?
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28ͧ?@??H??XbKgradient_tape/model/ssd_res_block3_branch2_conv/Conv2D/Conv2DBackpropFilterh?u  HB
?
&cudnn_dgrad_sm35_ldg_nt_64x16x128x8x32{?`* 2$8???@??JH??JXbHgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropInputhu  HB
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8뒕@뒕H뒕Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8???@???H???Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?A
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8ꐕ@??1H??1XbRgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?X8???@???H???XbKgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8?ӑ@??0H??0Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
redzone_checker*?2?@8???@??H??XbQgradient_tape/model/bbn_features_stemblock_stem1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)P?2* 2?
8???@???H???XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B

redzone_checker*?2?@8???@??H??XbHgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbRgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8˽?@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2B8???@???H???Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu  ?A
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)P?2* 2?
8?΍@?΍H?΍Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb?model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2Dhu  HB
?
.cudnn_convolve_sgemm_sm35_ldg_nn_32x16x64x8x16S?2*2?8???@???H???Xb*model/bbn_features_transition2_conv/Conv2Dhu ??B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8ʵ?@ʵ?Hʵ?Xb?model/bbn_features_denseblock2_denselayer1_branch1b_conv/Conv2Dhu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 20B8驋@驋H驋Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void DSE::regular_fft_clip<1, 2, 256, 16, 16, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??*2? 8꺊@꺊H꺊XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8?܈@?܈H?܈XbJgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2,?8?׈@?׈H?׈Xb(model/ssd_res_block2_branch2_conv/Conv2Dhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2@B8???@??CH??CXbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2? 8???@???H???bmodel/re_lu_113/Reluhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28Ȟ?@??AH??AXbLgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8ɲ?@ɲ?Hɲ?Xb?model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)?*?2?8??@??H??bFgradient_tape/model/bbn_features_transition1_norm/FusedBatchNormGradV3hu  ?B
?
?void DSE::regular_fft_clip<1, 2, 256, 16, 16, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??*2? 8??}@??}H??}XbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??|@??|H??|Xbagradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
:cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_64x8x256x8x32}?R* 28??|@??<H??@Xb*model/bbn_features_transition4_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*2X8??|@??|H??|XbQgradient_tape/model/bbn_features_stemblock_stem1_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
cvoid DSE::vector_fft<1, 2, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)5 ??*2? 8??|@??|H??|XbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
cvoid DSE::vector_fft<1, 2, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)5 ??*2? 8??|@??|H??|XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8??{@??{H??{Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 4, 6, 3, 2, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)O? *2,8??z@??zH??zXbLgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??z@??zH??zXb=gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*2X8??z@??zH??zXbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??y@??yH??yXb=gradient_tape/model/ssd_cls5conv2/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8ɚy@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch1b_conv/Conv2Dh	u  HB
?
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28??y@?hH??XbLgradient_tape/model/ssd_res_block4_branch1c_conv/Conv2D/Conv2DBackpropFilterh?u  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)H?*2?W8??x@??xH??xXbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)P?2* 2?8??x@??xH??xXbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??x@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputh	u  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2B8??x@??xH??xXbbgradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28??x@?hH??XbLgradient_tape/model/ssd_res_block3_branch1c_conv/Conv2D/Conv2DBackpropFilterh?u  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??w@??;H??<Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??w@??wH??wXbKgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2? 8??w@??wH??wXb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2? 8??w@??wH??wXbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??v@??vH??vXb?model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2,8Ȧv@??;H??;XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2? 8ȋu@ȋuHȋuXbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2? 8Ɇu@ɆuHɆuXb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??t@??tH??tXbKgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2B8??t@??tH??tXb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu  ?A
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)?*?2 8??t@??tH??tbJgradient_tape/model/bbn_features_stemblock_stem1_norm/FusedBatchNormGradV3hu  ?B
?
cvoid DSE::vector_fft<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)8 ??*2? 8ȝs@ȝsHȝsXbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
cvoid DSE::vector_fft<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)8 ??*2? 8??r@??rH??rXbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_r2c_32x32<float, false, 0u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?28??q@?}H??Xb.model/bbn_features_stemblock_stem1_conv/Conv2Dhdu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2(B8??q@??qH??qXbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?@8??q@??qH??qXbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??p@??pH??pXbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??p@??pH??pXbJgradient_tape/model/ssd_res_block3_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2 ?8??p@??pH??pXb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??p@??pH??pXbJgradient_tape/model/ssd_res_block4_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??p@??pH??pXb(model/ssd_res_block4_branch2_conv/Conv2Dhu  HB
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??p@??7H??8Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
0cudnn_convolve_sgemm_sm35_ldg_nn_128x8x128x16x16??A*2?8??p@??pH??pXb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??p@??pH??pXb(model/ssd_res_block3_branch2_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??o@??oH??oXbJgradient_tape/model/ssd_res_block5_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2,8??o@??7H??7Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void implicit_convolve_sgemm<float, float, 512, 6, 8, 3, 3, 5, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)u?R* 2Q8??o@??oH??oXb*model/bbn_features_transition3_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??o@??oH??oXb(model/ssd_res_block5_branch2_conv/Conv2Dhu  HB
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8??n@??nH??nXbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8ǻn@ǻnHǻnXb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2??8??n@??nH??nXb*model/bbn_features_transition4_conv/Conv2Dhu  H?
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8??m@??mH??mXbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8??m@??mH??mXbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??m@??$H??$Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8ǰm@ǰmHǰmXb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8??m@??mH??mXbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?A8??m@??6H??6Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8??l@??lH??lXb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2&8??l@??H??Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??l@??lH??lXb)model/ssd_res_block1_branch1a_conv/Conv2Dhu  HB
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*  2?8??k@??kH??kXbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8??j@??jH??jXbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void cudnn::cnn::wgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)]?*28??j@??jH??jXbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?@8ȥj@ȥjHȥjXbKgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cudnn_convolve_sgemm_sm35_ldg_nn_64x16x64x16x16O?B*2Q8??j@??jH??jXb*model/bbn_features_transition3_conv/Conv2Dhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??j@??jH??jXbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??i@??iH??iXbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??i@??iH??iXb?model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2Dhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?%8??f@?? H??#Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?A
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?%8??f@?? H??"XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??d@??dH??dXbKgradient_tape/model/ssd_res_block5_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?<8??d@??dH??dXb=gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??d@??dH??dXbKgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8Ǹd@ǸdHǸdXb)model/ssd_res_block1_branch1b_conv/Conv2Dhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??d@??dH??dXb<gradient_tape/model/ssd_cls4conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??d@??dH??dXb<gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??d@??dH??dXb<gradient_tape/model/ssd_cls3conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8Ǳd@ǱdHǱdXbKgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??d@??dH??dXb)model/ssd_res_block5_branch1b_conv/Conv2Dhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??d@??dH??dXb<gradient_tape/model/ssd_cls5conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*28??d@?? H??!bmodel/re_lu_113/Reluhu  HB
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??d@??dH??dXbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??d@??dH??dXb)model/ssd_res_block4_branch1b_conv/Conv2Dhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?<8??d@??dH??dXb=gradient_tape/model/ssd_cls3conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??d@??dH??dXb)model/ssd_res_block3_branch1b_conv/Conv2Dhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?<8??c@??cH??cXb=gradient_tape/model/ssd_cls4conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?<8??c@??cH??cXb=gradient_tape/model/ssd_cls5conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??c@??cH??cXbbgradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??c@??cH??cXbJgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2 ?8??c@??cH??cXb(model/ssd_res_block1_branch2_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??c@??cH??cXbKgradient_tape/model/ssd_res_block4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??b@??bH??bXbbgradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??`@??`H??`XbJgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2B8??`@??`H??`Xb?model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2Dhu  ?A
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8Ƿ`@??0Hâ0Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*208??`@??H?? Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??_@??/H??/Xb?model/bbn_features_denseblock1_denselayer1_branch1b_conv/Conv2Dhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8Ƿ^@Ƿ^HǷ^Xbmodel/ssd_cls5conv2/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??^@??^H??^Xbmodel/ssd_cls1conv2/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8ǰ^@ǰ^Hǰ^Xbmodel/ssd_cls4conv2/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??^@??^H??^Xbmodel/ssd_cls3conv2/Conv2Dhu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2 B8??]@??]H??]Xbagradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)P?2* 28??]@??]H??]XbHgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropInputhu  ?B

/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??\@??\H??\Xb(model/ssd_res_block2_branch2_conv/Conv2Dhu  HB
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?%8??\@??H?? XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?%8??[@??H?? Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?A
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8Ƙ[@??-H??-Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??Z@??ZH??ZXbagradient_tape/model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??Z@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1b_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??Y@??YH??YXbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??X@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch2c_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8??X@??H??Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?A
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??W@??WH??WXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??*2? 8??V@??VH??VXbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??U@??UH??UXbbgradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??T@??TH??TXbbgradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8??S@??)H??*XbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
?void DSE::regular_fft_pad<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??*2? 8??S@??SH??SXbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??S@??SH??SXbRgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?a8??S@??SH??SXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2
?8??S@??SH??SXbagradient_tape/model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2
?8??R@??RH??RXbagradient_tape/model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?a8??R@??RH??RXbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??R@??RH??RXbLgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2?8??R@??)H??)Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  HB
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?a8ƎR@ƎRHƎRXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?a8??Q@??QH??QXbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?8??Q@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void cudnn::cnn::wgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)\?*28ƚQ@ƚQHƚQXbMgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?8??P@??PH??PXbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8??O@??'H??(Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2,8??O@??OH??OXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2&8??O@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*208??O@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_r2c_32x32<float, true, 0u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)? ??*?2 8??M@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8ćM@??H??Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?A
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*208??L@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??L@??LH??LXb?model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2Dhu  ?B
?
?void fft2d_c2r_32x32<float, false, false, 0u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)8 ??*?28??K@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2	?8??J@??JH??JXbagradient_tape/model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??J@??JH??JXb?model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2	?8??J@??JH??JXbagradient_tape/model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??J@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch1b_conv/Conv2Dhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)P?2* 2?
8??J@??JH??JXbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?,8??I@??IH??IXbLgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??I@??IH??IXbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??H@??H??Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??H@??HH??HXbJgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8??G@??GH??GXb?model/bbn_features_denseblock2_denselayer1_branch2c_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??G@??GH??GXbbgradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2)8??G@??H??Xb)model/ssd_res_block1_branch1b_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??F@??FH??FXbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??F@??FH??FXb?model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2Dhu  HB
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2? 8??F@??FH??FXb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu ??B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??F@??FH??FXbbgradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2 ?8??F@??FH??FXbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*28??F@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
q
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??F@??FH??FXbmodel/ssd_cls1conv2/Conv2Dhu  HB
?
?void DSE::regular_fft_clip<1, 2, 256, 16, 16, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??*2?8??F@??FH??FXb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 28??E@??H??XbKgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 512, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float) ?*?2?8??E@??EH??Eb4model/bbn_features_transition1_norm/FusedBatchNormV3hu  ?B
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 8, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)|?R* 28??D@??DH??DXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2?88??D@??DH??DXbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??D@??DH??DXbKgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2,?8??D@??DH??DXb)model/ssd_res_block2_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*2$8??D@??DH??DXbIgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropFilterhu  HB

/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??D@??DH??DXb(model/ssd_res_block1_branch2_conv/Conv2Dhu  HB
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?8ĻC@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
hvoid fft1d_r2c_32<float, float, float2, true, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?y8ģC@ģCHģCXb*model/bbn_features_transition4_conv/Conv2Dhu  ?A
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?y8??B@??BH??BXbLgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??B@??BH??BXbagradient_tape/model/bbn_features_denseblock4_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??B@??BH??BXb?model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*208??B@??HXbbgradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??B@??BH??Bbmodel/re_lu_115/Reluhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8āB@āBHāBbmodel/re_lu_114/Reluhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??A@??AH??AXbHgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8??A@?? H??!Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??A@??AH??AXbHgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??A@??H??XbHgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropInputh$u  ?B
?
?void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 512, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float) ?*?2 8??A@??AH??Ab8model/bbn_features_stemblock_stem1_norm/FusedBatchNormV3hu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 6, 7, 3, 3, 5, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)M?2* 28??A@??AH??AXb*model/bbn_features_transition4_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28ıA@ıAHıAXb<gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??A@??AH??AXbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??@@??@H??@XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)P?2* 2?
8??@@??@H??@XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??@@??H??XbHgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropInputh$u  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?%8???@??H??!Xb*model/bbn_features_transition2_conv/Conv2Dhu  ?A
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?%8???@H??!XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2E8į?@į?Hį?Xbmodel/ssd_cls4conv2/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*28???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??>@??>H??>Xb?model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2Dhu  HB
?
cvoid DSE::vector_fft<1, 2, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)5 ??*2?8ů>@ů>Hů>Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?B
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 5, 6, 4, 3, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)X?A*2+8??>@??H??XbHgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropInputhu ??B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8??=@??=H??=Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)?*?28??=@??=H??=bKgradient_tape/model/bbn_features_stemblock_stem2a_norm/FusedBatchNormGradV3hu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??=@ñH??Xb?model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2Dhu  ?B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2 ?8??<@??<H??<Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu ??B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2? 8??<@??<H??<Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu ??B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??<@??<H??<Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??<@??<H??<XbLgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??<@??<H??<XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*28ś<@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??<@??<H??<Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8Ň<@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??;@??;H??;XbKgradient_tape/model/ssd_res_block5_branch2_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??;@??;H??;Xb?model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2Dhu  HB
j
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??;@??;H??;bmodel/re_lu_114/Reluhu  HB
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2?08??;@??;H??;Xbbgradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 4, 6, 3, 2, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)O? *2,8??;@??;H??;XbJgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??;@??;H??;Xb)model/ssd_res_block1_branch1c_conv/Conv2Dhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8ğ;@ğ;Hğ;Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
j
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8ě;@ě;Hě;bmodel/re_lu_115/Reluhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??;@??;H??;Xb)model/ssd_res_block4_branch1c_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??;@??;H??;Xb)model/ssd_res_block3_branch1c_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??;@??;H??;Xb)model/ssd_res_block5_branch1c_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)M?2* 2$8??;@??H??XbIgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??:@??:H??:XbKgradient_tape/model/ssd_res_block3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?28??:@??:H??:b"gradient_tape/model/re_lu/ReluGradhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??:@??:H??:XbKgradient_tape/model/ssd_res_block4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??:@??:H??:XbKgradient_tape/model/ssd_res_block5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??:@??:H??:Xbbgradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?28??:@??:H??:b%gradient_tape/model/re_lu_19/ReluGradhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8ľ:@ľ:Hľ:XbHgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropInputhu  HB
~
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8ļ:@ļ:Hļ:Xb&model/ssd_feature_extend3_conv1/Conv2Dhu  HB
~
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??:@??:H??:Xb&model/ssd_feature_extend2_conv1/Conv2Dhu  HB
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2??8??:@??:H??:Xb*model/bbn_features_transition3_conv/Conv2Dhu  H?
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2,8??:@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2?8??:@H??Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??:@??:H??:XbKgradient_tape/model/ssd_res_block4_branch1c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??:@??:H??:XbKgradient_tape/model/ssd_res_block2_branch1c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??9@??9H??9XbHgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??9@??9H??9XbKgradient_tape/model/ssd_res_block3_branch1c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??9@??9H??9XbKgradient_tape/model/ssd_res_block5_branch1c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
cvoid DSE::vector_fft<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)8 ??*2?8??9@??9H??9XbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??9@??9H??9Xb?model/bbn_features_denseblock4_denselayer1_branch1a_conv/Conv2Dhu  HB
?
cvoid DSE::vector_fft<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)8 ??*2?8??9@??9H??9XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8Ħ9@Ħ9HĦ9Xbbgradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??9@??9H??9Xb?model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2Dhu  ?B
?
?void fft2d_c2r_32x32<float, true, false, 0u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)8 ??*?2 8ĕ9@??H??Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??9@??9H??9Xbagradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8??8@??8H??8Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
.cudnn_convolve_sgemm_sm35_ldg_nn_32x16x64x8x16S?2*2)8??8@??8H??8Xb*model/bbn_features_transition4_conv/Conv2Dhu ??B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??8@??8H??8Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??8@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2Dhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??8@??8H??8XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??8@??8H??8Xb)model/ssd_res_block5_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*2	8??7@??7H??7Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?A8??7@??7H??7XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?	
?	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??7@??7H??7bAdam/gradients/AddN_30hu  ?B
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?A8??7@??7H??7Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2,8ĳ7@ĳ7Hĳ7Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?A8??7@??7H??7Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2? 8??7@H??Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu  ?A
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??7@??7H??7Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8ą7@??H??Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu  HB
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?A8??6@??6H??6Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2? 8??6@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2Dhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?A8??6@??6H??6XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?A8Ľ6@Ľ6HĽ6Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??6@??6H??6bAdam/gradients/AddN_33hu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?A8Ī6@Ī6HĪ6Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
?void cudnn::pooling_bw_kernel_max<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor) ? *?2,8??6@??6H??6b5gradient_tape/model/max_pooling2d/MaxPool/MaxPoolGradhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)H?*2?8??6@??6H??6Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??5@??5H??5Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??5@??5H??5Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??5@??5H??5Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??5@??H??Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8??5@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??5@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2? 8??5@??5H??5XbLgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??5@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??5@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch2c_conv/Conv2Dhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2? 8??5@??5H??5XbKgradient_tape/model/ssd_res_block3_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?8??5@??5H??5Xbbgradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??5@??H??Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??5@??5H??5Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??5@??5H??5Xb?model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8ę5@ę5Hę5Xbagradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)?*?2?8??5@??5H??5bFgradient_tape/model/bbn_features_transition2_norm/FusedBatchNormGradV3hu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2? 8??5@??5H??5XbKgradient_tape/model/ssd_res_block4_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2? 8??5@??5H??5XbKgradient_tape/model/ssd_res_block5_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??4@??4H??4XbKgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??4@??4H??4Xbbgradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28İ4@İ4Hİ4Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8??4@??H??Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?A
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8Ė4@Ė4HĖ4Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch2a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_r2c_32x32<float, false, 0u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?28??3@??H??Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??3@??3H??3Xbbgradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8??3@??H??Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 28??3@??H??XbLgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8??3@??H??Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?A
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28â3@â3Hâ3Xbbgradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*2[8??3@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch1b_conv/Conv2Dh$u  ?B
?
[void cudnn::ops::scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*?2??8??2@??H??XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*2[8??2@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2? 8??2@??2H??2XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2? 8??2@??2H??2Xb*model/bbn_features_transition2_conv/Conv2Dhu ??B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2 ?8??2@??2H??2Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*28??2@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2? 8??2@??2H??2Xb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu ??B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??2@??2H??2Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?%8ĵ2@??H??XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?%8??2@??H??Xb*model/bbn_features_transition2_conv/Conv2Dhu  ?A
?
[void cudnn::ops::scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*?2??8??2@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??2@??2H??2Xb(model/ssd_res_block4_branch2_conv/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8Ĕ2@Ĕ2HĔ2XbJgradient_tape/model/ssd_res_block3_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??2@??2H??2XbJgradient_tape/model/ssd_res_block4_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??2@??2H??2XbKgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8Ñ2@Ñ2HÑ2Xb(model/ssd_res_block5_branch2_conv/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2 ?8??2@??2H??2Xb)model/ssd_res_block1_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*2
8??2@??2H??2Xbbgradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??2@??2H??2Xb(model/ssd_res_block3_branch2_conv/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??2@??2H??2XbJgradient_tape/model/ssd_res_block5_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
&cudnn_dgrad_sm35_ldg_nt_64x16x128x8x32{?`* 28??2@??HXbKgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8Ą2@Ą2HĄ2Xb?model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??2@??2H??2Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??1@??1H??1Xbbgradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2?(8??1@??1H??1Xbbgradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??1@??1H??1Xbagradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*208??1@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8??1@??H??Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu  ?A
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8??1@??H??Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28??1@??H??XbJgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?8??1@??1H??1Xbbgradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
q
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??0@??0H??0Xbmodel/ssd_cls3conv2/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??0@??0H??0Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch2b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??0@??0H??0Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??0@??0H??0Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch2b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??0@??0H??0Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?8í0@í0Hí0Xbbgradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??0@??0H??0Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??0@??0H??0Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch2b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??0@??0H??0Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch2b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??0@??0H??0Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 28??0@??H??XbKgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??0@??0H??0Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch2c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8Ù0@Ù0HÙ0Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch2b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??0@??0H??0Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8Ė0@??H??Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8Ė0@Ė0HĖ0Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??0@??0H??0Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??0@??0H??0Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch2b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??0@??0H??0Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch2c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??0@??0H??0Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch2c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??/@??/H??/Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2)8??/@??H??Xb(model/ssd_res_block2_branch2_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 28??/@??H??Xb=gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??/@??/H??/Xbbgradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??/@??/H??/Xb?model/bbn_features_denseblock1_denselayer2_branch1b_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??/@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??/@??/H??/Xb?model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??/@??/H??/Xb?model/bbn_features_denseblock1_denselayer3_branch2b_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??/@??/H??/Xb?model/bbn_features_denseblock1_denselayer2_branch2b_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??/@??/H??/Xb?model/bbn_features_denseblock1_denselayer1_branch2c_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??/@??/H??/Xb?model/bbn_features_denseblock1_denselayer1_branch2b_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??/@??/H??/Xb?model/bbn_features_denseblock1_denselayer2_branch2c_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??/@??/H??/Xbbgradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??/@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch1b_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??/@??/H??/Xb?model/bbn_features_denseblock1_denselayer3_branch2c_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*28??/@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??.@??.H??.Xb)model/ssd_res_block2_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8??.@??HµXbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)H?*2?8??.@??.H??.Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?8??.@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?8đ.@??H??XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void cudnn::pooling_bw_kernel_avg<float, float, cudnn::averpooling_func<float, true>, 2, false>(cudnnTensorStruct, float const*, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)( ? *?2 8Ć.@Ć.HĆ.b1gradient_tape/model/average_pooling2d/AvgPoolGradhu  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??.@??H??XbLgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8??-@??-H??-Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch2a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*208??-@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8ä-@ä-Hä-Xb?model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??-@??-H??-Xbagradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 28??-@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_r2c_32x32<float, false, 0u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??,@??H??XbMgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??,@??,H??,Xb<gradient_tape/model/ssd_cls3conv2/Conv2D/Conv2DBackpropInputhu  HB
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??,@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch1b_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?	8??,@??H??	XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2? 8??,@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2?8??,@??,H??,Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??,@??,H??,Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 4, 6, 3, 2, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)O? *2 8??,@??,H??,XbJgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
q
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28??,@??H??Xb)model/ssd_res_block1_branch1b_conv/Conv2Dh$u  ?B
q
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28ç,@??H??Xb)model/ssd_res_block3_branch1b_conv/Conv2Dh$u  ?B
?
?void DSE::regular_fft_pad<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??*2?8??,@??,H??,XbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??+@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch2c_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
?void DSE::regular_fft_pad<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??*2?8??+@??+H??+XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
Z
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28??+@??H??bmodel/re_lu_114/Reluh$u  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*28??+@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
Z
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28??+@??H??bmodel/re_lu_115/Reluh$u  ?B
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 28??+@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??+@??H??XbKgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
q
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28??+@??H??Xb)model/ssd_res_block5_branch1b_conv/Conv2Dh$u  ?B
?
?void tensorflow::functor::PadInputCustomKernelNCHW<float, 4>(int, float const*, tensorflow::functor::Dimension<4>, float*, tensorflow::functor::Dimension<4>, tensorflow::functor::Dimension<(4)-(2)>, float)*?28??+@??+H??+XbRgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void tensorflow::functor::PadInputCustomKernelNCHW<float, 4>(int, float const*, tensorflow::functor::Dimension<4>, float*, tensorflow::functor::Dimension<4>, tensorflow::functor::Dimension<(4)-(2)>, float)*?28??+@??+H??+Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  ?B
q
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28Ñ+@??H??Xb)model/ssd_res_block4_branch1b_conv/Conv2Dh$u  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??*@??H??XbKgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
?void tensorflow::functor::PadInputCustomKernelNCHW<float, 4>(int, float const*, tensorflow::functor::Dimension<4>, float*, tensorflow::functor::Dimension<4>, tensorflow::functor::Dimension<(4)-(2)>, float)*?28??*@??*H??*XbQgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*28??*@??H??bmodel/re_lu_114/Reluhu  HB
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28¡*@??H??XbKgradient_tape/model/ssd_res_block4_branch1b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?8??*@??*H??*Xbbgradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??)@??H??XbKgradient_tape/model/ssd_res_block5_branch1b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??)@??)H??)Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8??)@??)H??)XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu ??B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8??)@??)H??)Xb*model/bbn_features_transition2_conv/Conv2Dhu ??B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?228??)@??)H??)Xb&model/ssd_feature_extend2_conv1/Conv2Dhu  ?B
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?	8??)@??H??Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?A
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*28??)@??H??Xb)model/ssd_res_block3_branch1b_conv/Conv2Dhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?
8??)@??)H??)Xbagradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 4, 6, 3, 2, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)O? *28??)@??)H??)Xb<gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8??)@§HXb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8??)@??H??Xb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu  ?A
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?
8??)@??)H??)Xb?model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2 ?8??(@??(H??(Xbagradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu ??B
?
hvoid fft1d_r2c_32<float, float, float2, true, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?@8??(@??(H??(Xb*model/bbn_features_transition3_conv/Conv2Dhu  ?A
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2? 8??(@??(H??(Xb?model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2Dhu ??B
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2? 8??(@??(H??(Xbbgradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2??8??'@??'H??'Xb(model/ssd_res_block2_branch2_conv/Conv2Dhu  H?
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2	8??'@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 28??'@??H¤Xbagradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8??'@??'H??'Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch2a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?
8??'@??'H??'Xbbgradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*?28??'@??H??bmodel/concatenate_3/concathu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)H?*2?8??'@??'H??'Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, false, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)P?*28??'@??H??Xb)model/ssd_res_block5_branch1b_conv/Conv2Dhu  HB
?
?void wgrad2d_grouped_direct_kernel<float, float, float, (cudnnTensorFormat_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnConvolutionStruct, cudnnFilterStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int)*?2?	8??&@??&H??&Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*?28??&@??&H??&bmodel/re_lu/Reluhu  ?B
?
&cudnn_dgrad_sm35_ldg_nt_64x16x128x8x32{?`* 28??&@??&H??&XbHgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropInputhu  HB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*?28??&@??&H??&bmodel/re_lu_19/Reluhu  ?B
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2?8??&@??&H??&Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu ??B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2
?8??&@??&H??&Xbbgradient_tape/model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??&@??&H??&XbKgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2
?8??&@??&H??&Xbbgradient_tape/model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2?8??&@??&H??&Xb?model/bbn_features_denseblock1_denselayer1_branch1b_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8??&@??H??Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu  HB
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?8??%@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 28??%@??H??XbLgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2Q8??%@??%H??%Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??%@??%H??%Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch2a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
[void cudnn::ops::scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*?2և8??%@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void wgrad2d_grouped_direct_kernel<float, float, float, (cudnnTensorFormat_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnConvolutionStruct, cudnnFilterStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int)*?2?8??%@??	H??	Xb=gradient_tape/model/ssd_box3conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28??%@??H??XbJgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?	8§%@§%H§%Xbagradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?	8%@%H%Xb?model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28Ô%@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::im2col4d_kernel<float, long>(cudnn::cnn::im2col4d_params, cudnnConvolutionStruct, cudnnTensor4dStruct, float const*, float*) *?2?8??$@??$H??$Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??$@??$H??$Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 5, 6, 4, 3, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)X?A*2+8??$@??$H??$XbKgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*28??$@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*28??$@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*28??$@??H??bmodel/re_lu_115/Reluhu  HB
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8??$@??$H??$Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu ??B
?	
?	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??$@??$H??$bAdam/gradients/AddN_31hu  ?B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8??$@??$H??$Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu ??B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?@8??$@??$H??$XbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?	8??#@??#H??#Xbbgradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?8??#@??#H??#Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8??#@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*28??#@??H??Xb)model/ssd_res_block4_branch1b_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8??#@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?28??#@??#H??#Xbmodel/ssd_box1conv2/Conv2Dhu  ?B
?
?void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 512, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float) ?*?28#@#H#b9model/bbn_features_stemblock_stem2a_norm/FusedBatchNormV3hu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28#@??H??XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2)8#@??H??Xbmodel/ssd_cls1conv2/Conv2Dhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28??"@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 5, 6, 4, 3, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)X?A*2+8??"@«H??XbKgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8??"@??H??Xb?model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??"@??"H??"Xb)model/ssd_res_block1_branch1a_conv/Conv2Dhu  HB
?
9cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_32x16x32x8x8D?$*2?8??"@??H??Xb?model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2	?8??"@??"H??"Xbbgradient_tape/model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2	?8??"@??"H??"Xbbgradient_tape/model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??"@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28??"@??H??Xb<gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 8, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)|?R* 28"@"H"Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8"@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2Dhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?8??!@??!H??!Xbbgradient_tape/model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8??!@??!H??!Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)\?*28??!@??!H??!XbKgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 5, 6, 4, 3, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)X?A*2+8??!@??H??XbKgradient_tape/model/ssd_res_block4_branch1b_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??!@??!H??!XbKgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8??!@??H??Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu  HB
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2?8??!@??H??Xb?model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2Dhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??!@??!H??!XbKgradient_tape/model/ssd_res_block5_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??!@??!H??!Xb)model/ssd_res_block5_branch1b_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??!@??!H??!Xb)model/ssd_res_block4_branch1b_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??!@??!H??!XbKgradient_tape/model/ssd_res_block4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8º!@º!Hº!Xb)model/ssd_res_block1_branch1b_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8ó!@ó!Hó!Xb)model/ssd_res_block3_branch1b_conv/Conv2Dhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2Q8¯!@¯!H¯!Xbagradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8¯!@¯!H¯!XbKgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8??!@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2Dhu  ?A
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??!@??!H??!Xb?model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8Ü!@Ü!HÜ!Xbagradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2*@8??!@??!H??!Xb?model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??!@??!H??!Xbagradient_tape/model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)P?2* 2?
8??!@??!H??!Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8?? @?? H?? Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch2a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
9cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_32x16x32x8x8D?$*2?8?? @?? H?? Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  HB
?
?void tensorflow::functor::PadInputCustomKernelNCHW<float, 4>(int, float const*, tensorflow::functor::Dimension<4>, float*, tensorflow::functor::Dimension<4>, tensorflow::functor::Dimension<(4)-(2)>, float)*?28?? @?? H?? Xb.model/bbn_features_stemblock_stem1_conv/Conv2Dhu  ?B
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?8û @??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void tensorflow::functor::PadInputCustomKernelNCHW<float, 4>(int, float const*, tensorflow::functor::Dimension<4>, float*, tensorflow::functor::Dimension<4>, tensorflow::functor::Dimension<(4)-(2)>, float)*?28º @º Hº XbQgradient_tape/model/bbn_features_stemblock_stem1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8?? @??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28?? @??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28?? @?? H?? Xbagradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?8?? @?? H?? Xbbgradient_tape/model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2?8« @« H« Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu ??B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28¨ @¨ H¨ Xbagradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28¨ @¨ H¨ b'gradient_tape/model/concatenate_3/Slicehu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8?? @??H??Xb?model/bbn_features_denseblock1_denselayer1_branch1b_conv/Conv2Dhu  ?B
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2?8?? @??H??Xb?model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*208?? @?? H?? Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch2a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?8?? @?? H?? XbHgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
9cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_32x16x32x8x8D?$*2?8 @ H Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  HB
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28 @?lH??XbLgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28??@?mH??Xb?model/bbn_features_denseblock3_denselayer1_branch1b_conv/Conv2Dh$u  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xb?model/bbn_features_denseblock3_denselayer1_branch1b_conv/Conv2Dhu  HB
?
9cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_32x16x32x8x8D?$*2?8??@??H??Xb?model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2Dhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2(@8??@??H??Xb?model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2Dhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8??@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2Dhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cub::DeviceSegmentedRadixSortKernel<cub::DeviceRadixSortPolicy<float, int, int>::Policy700, true, true, float, int, cub::TransformInputIterator<int, tensorflow::impl::SegmentOffsetCreator, cub::CountingInputIterator<int, long>, long>, int>(float const*, float*, int const*, int*, cub::TransformInputIterator<int, tensorflow::impl::SegmentOffsetCreator, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::impl::SegmentOffsetCreator, cub::CountingInputIterator<int, long>, long>, int, int, int)`?D*?28??@??H??b2compute_loss/cond/else/_1/compute_loss/cond/TopKV2hu  zB
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8??@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28ñ@?kH??Xb?model/bbn_features_denseblock4_denselayer1_branch1b_conv/Conv2Dh$u  ?B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8¨@¨H¨Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu ??B
?	
?	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@??H??bAdam/gradients/AddN_26hu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8??@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch1b_conv/Conv2Dhu  ?A
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)H?*2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??@??H??XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28@?jH??Xb?model/bbn_features_denseblock2_denselayer1_branch1b_conv/Conv2Dh$u  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8@HXb)model/ssd_res_block4_branch1b_conv/Conv2Dhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?$8@HXb*model/bbn_features_transition3_conv/Conv2Dhu  ?A
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?8??@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbLgradient_tape/model/ssd_res_block2_branch1c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbKgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbIgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropFilterhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?$8??@??H??XbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2Q8??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2Q8??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbIgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbLgradient_tape/model/ssd_res_block3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??@??H??XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbKgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbLgradient_tape/model/ssd_res_block5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?8??@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbLgradient_tape/model/ssd_res_block4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbLgradient_tape/model/ssd_res_block5_branch1c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
9cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_32x16x32x8x8D?$*2)8??@??H??Xb)model/ssd_res_block2_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xb)model/ssd_res_block3_branch1b_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbKgradient_tape/model/ssd_res_block5_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xb)model/ssd_res_block5_branch1b_conv/Conv2Dhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xb)model/ssd_res_block1_branch1b_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbKgradient_tape/model/ssd_res_block4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2&@8??@??H??Xb?model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)\?*28??@??H??Xb=gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 5, 6, 4, 3, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)X?A*2+8??@??H??XbHgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropInputhu ??B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)\?*28??@??H??XbKgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??@??H??Xb?model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??@??H??Xb?model/bbn_features_denseblock4_denselayer1_branch1b_conv/Conv2Dhu  HB
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 1024, 2, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?`*?2?8??@??H??bwgradient_tape/model/bbn_features_stemblock_stem1_conv/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??@??H??Xb?model/bbn_features_denseblock3_denselayer1_branch1b_conv/Conv2Dhu  HB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*?28??@??H??bmodel/concatenate_2/concathu  ?B
r
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xbmodel/ssd_box4conv2/Conv2Dhu  HB
r
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xbmodel/ssd_box5conv2/Conv2Dhu  HB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?28??@??H??b$gradient_tape/model/re_lu_1/ReluGradhu  ?B
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2??8??@??H??Xb(model/ssd_res_block1_branch2_conv/Conv2Dhu  H?
r
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xbmodel/ssd_box3conv2/Conv2Dhu  HB
?
:cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_64x8x256x8x32}?R* 28@??H??Xb(model/ssd_res_block1_branch2_conv/Conv2Dhu  HB
r
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xbmodel/ssd_box1conv2/Conv2Dhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?8??@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??@??H??Xb?model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2,8??@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2?8??@??H??Xb?model/bbn_features_denseblock1_denselayer3_branch2a_conv/Conv2Dhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28??@??H??XbKgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2$@8??@??H??Xb?model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2	8??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
hvoid fft1d_r2c_32<float, float, float2, true, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?,8??@??H??Xb(model/ssd_res_block2_branch2_conv/Conv2Dhu  ?A
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)?*?2 8??@??H??bKgradient_tape/model/bbn_features_stemblock_stem2b_norm/FusedBatchNormGradV3hu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8??@??H??Xb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2Q8??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?!8??@??H??Xb?model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2Dhu  ?A
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)?*?2 8??@??H??bJgradient_tape/model/bbn_features_stemblock_stem3_norm/FusedBatchNormGradV3hu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8­@­H­Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??@??H??XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?28??@??H??b%gradient_tape/model/re_lu_40/ReluGradhu  ?B
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??@??H??Xb?model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 28??@??	H??	XbLgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??@??H??Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 5, 6, 4, 3, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)X?A*28??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1b_conv/Conv2D/Conv2DBackpropInputhu ??B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8??@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??@??H??Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
?void pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)( ? *?2,8??@??H??bmodel/max_pooling2d/MaxPoolhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8??@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2? 8??@??H??Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2? 8??@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2? 8??@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void pooling_fw_4d_kernel<float, float, cudnn::averpooling_func<float, true>, (cudnnPoolingMode_t)2, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)0 ?*?2 8??@??H??bmodel/average_pooling2d/AvgPoolhu?O?B
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2??8??@??H??Xbmodel/ssd_cls3conv2/Conv2Dhu  H?
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2? 8??@??H??XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2??8??@??H??Xbmodel/ssd_cls5conv2/Conv2Dhu  H?
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2??8??@??H??Xbmodel/ssd_cls1conv2/Conv2Dhu  H?
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2??8??@??H??Xbmodel/ssd_cls4conv2/Conv2Dhu  H?
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch1b_conv/Conv2Dhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?8??@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2Dhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?8??@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch1b_conv/Conv2Dhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2? 8??@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?8??@??H??XbIgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)G?*2$8??@??H??XbLgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?8??@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B