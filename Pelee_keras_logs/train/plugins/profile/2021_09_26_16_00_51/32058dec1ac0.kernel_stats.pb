
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?W8ߧ?"@ߧ?"Hߧ?"Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2??8蟎@蟎H蟎XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?ػ@?ػH?ػXbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb*model/bbn_features_transition3_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@???H???XbHgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropInputhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2@B8???@???H???XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2@B8???@???H???Xb*model/bbn_features_transition2_conv/Conv2Dhu  ?A
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2 ?8?֕
@?֕
H?֕
XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2 ?8׏?	@׏?	H׏?	Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?A
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@??
H??Xb.model/bbn_features_stemblock_stem1_conv/Conv2Dhdu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2??8???@???H???XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2??8???@???H???XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2!?8???@???H???XbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???XbMgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2??8?Ť@?ŤH?ŤXb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8舑@舑H舑XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???XbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8???@???H???XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8???@???H???Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8???@???H???XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb*model/bbn_features_transition4_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?ȕ@?ȕH?ȕXbLgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputh$u  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?+8???@???H???Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2??8???@???H???XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2??8ݨ?@ݨ?Hݨ?XbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2??8???@???H???Xb*model/bbn_features_transition2_conv/Conv2Dhu ??B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2??8???@???H???XbHgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2??8???@???H???Xb*model/bbn_features_transition3_conv/Conv2Dhu  ?B
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2??8???@???H???XbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2??8???@???H???XbMgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, true, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2d8???@???H???bmodel/re_lu_113/Reluhu  ?B
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 6, 7, 5, 4, 5, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)V??* 28ٶ?@ٶ?Hٶ?XbQgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputhu  HB
c
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28???@??H??Xbmodel/ssd_cls3conv2/Conv2Dh?u  HB
c
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28???@??H??Xbmodel/ssd_cls1conv2/Conv2Dh?u  HB
c
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28???@??H??Xbmodel/ssd_cls4conv2/Conv2Dh?u  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xbmodel/ssd_cls1conv2/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8???@???H???Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8???@???H???Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 2?8Ԫ?@Ǹ?H???XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???XbMgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2??8???@???H???XbMgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropFilterhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock3_denselayer1_branch1b_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8?@??H??Xb?model/bbn_features_denseblock3_denselayer1_branch2c_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb)model/ssd_res_block2_branch1a_conv/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2,?8???@???H???XbLgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2,?8???@???H???Xb*model/bbn_features_transition4_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8ѿ?@??H??Xb?model/bbn_features_denseblock4_denselayer1_branch2b_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block1_branch1b_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock4_denselayer1_branch2c_conv/Conv2Dhu  ?B
m
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@???H???bmodel/re_lu_113/Reluhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2Dhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 6, 7, 3, 3, 5, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)M?2* 2?
8???@??xH??{Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block5_branch1b_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block4_branch1b_conv/Conv2Dhu  ?B
I
redzone_checker*?2?@8???@??H??bmodel/re_lu_115/Reluhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch2b_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block3_branch1b_conv/Conv2Dhu  ?B
I
redzone_checker*?2?@8???@??H??bmodel/re_lu_114/Reluhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch2c_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch1b_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8ϵ?@??H??Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
?void implicit_convolve_sgemm<float, float, 1024, 6, 7, 3, 3, 5, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)@?2* 2?8???@??sH??tXb*model/bbn_features_transition2_conv/Conv2Dhu  ?B
?
0cudnn_convolve_sgemm_sm35_ldg_nn_128x8x128x16x16??A*2?8???@???H???Xb.model/bbn_features_stemblock_stem1_conv/Conv2Dhu  B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8͕?@??H??Xb?model/bbn_features_denseblock4_denselayer1_branch2a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2Dhu  ?B
_
redzone_checker*?2?@8???@??H??Xb(model/ssd_res_block1_branch2_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block1_branch1c_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 208???@??mH??pXbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2Dhu  ?B
_
redzone_checker*?2?@8???@??H??Xb(model/ssd_res_block3_branch2_conv/Conv2Dhu  ?B
_
redzone_checker*?2?@8???@??H??Xb(model/ssd_res_block4_branch2_conv/Conv2Dhu  ?B
_
redzone_checker*?2?@8???@??H??Xb(model/ssd_res_block2_branch2_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8͈?@??H??Xb)model/ssd_res_block4_branch1c_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??H??XbIgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropFilterhu  ?B
a
redzone_checker*?2?@8???@??H??Xb*model/bbn_features_transition3_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??H??XbIgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropFilterhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block3_branch1c_conv/Conv2Dhu  ?B
a
redzone_checker*?2?@8???@??H??Xb*model/bbn_features_transition4_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8???@???H???Xb?model/bbn_features_denseblock1_denselayer1_branch2a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?î@?îH?îXbJgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?
8?ج@?جH?جXb?model/bbn_features_denseblock1_denselayer1_branch2b_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void wgrad2d_grouped_direct_kernel<float, float, float, (cudnnTensorFormat_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnConvolutionStruct, cudnnFilterStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int)*?2?8???@???H???Xbbgradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
e
redzone_checker*?2?@8???@??H??Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8Δ?@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch2a_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer6_branch2c_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer6_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
Q
redzone_checker*?2?@8͂?@??H??Xbmodel/ssd_box1conv2/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropInputhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block1_branch1a_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8̡?@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  ?B
Q
redzone_checker*?2?@8ˁ?@??H??Xbmodel/ssd_cls1conv2/Conv2Dhu  ?B
v
redzone_checker*?2?@8?Ԥ@??H??Xb?model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8?Ӥ@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?Ƥ@?ƤH?ƤXb(model/ssd_res_block2_branch2_conv/Conv2Dhu  HB
?
redzone_checker*?2?@8?¤@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch2c_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2Dhu  ?B
a
redzone_checker*?2?@8???@??H??Xb*model/bbn_features_transition2_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8?ܣ@??H??Xb?model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2Dhu  ?B
Q
redzone_checker*?2?@8?ˣ@??H??Xbmodel/ssd_cls3conv2/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block5_branch1a_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8画@??H??Xb)model/ssd_res_block3_branch1a_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
`
redzone_checker*?2?@8˛?@??H??Xb)model/ssd_res_block2_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8ɐ?@??H??Xb?model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8ʆ?@??H??Xb?model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8@??H??Xb?model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2Dhu  ?B
Q
redzone_checker*?2?@8???@??H??Xbmodel/ssd_box3conv2/Conv2Dhu  ?B
]
redzone_checker*?2?@8???@??H??Xb&model/ssd_feature_extend3_conv1/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2Dhu  ?B

redzone_checker*?2?@8???@??H??XbHgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8?ݢ@??H??Xb?model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8?Ң@??H??XbKgradient_tape/model/ssd_res_block5_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
_
redzone_checker*?2?@8?͢@??H??Xb(model/ssd_res_block5_branch2_conv/Conv2Dhu  ?B
Q
redzone_checker*?2?@8?ʢ@??H??Xbmodel/ssd_box4conv2/Conv2Dhu  ?B
v
redzone_checker*?2?@8?Ƣ@??H??Xb?model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8?¢@??H??Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb)model/ssd_res_block1_branch1a_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8;?@??H??XbLgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
v
redzone_checker*?2?@8м?@??H??Xb?model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8躢@??H??XbKgradient_tape/model/ssd_res_block4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B

redzone_checker*?2?@8촢@??H??XbHgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8뮢@??H??XbLgradient_tape/model/ssd_res_block4_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
v
redzone_checker*?2?@8έ?@??H??Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu  ?B
Q
redzone_checker*?2?@8ͨ?@??H??Xbmodel/ssd_box5conv2/Conv2Dhu  ?B
Q
redzone_checker*?2?@8???@??H??Xbmodel/ssd_cls5conv2/Conv2Dhu  ?B
Q
redzone_checker*?2?@8濫@??H??Xbmodel/ssd_cls4conv2/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8̳?@??H??Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?
8̭?@̭?H̭?Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 8, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)u?R* 28@??_H??`XbMgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2?8?՟@?ӏH恐XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2?8̚?@???H???XbQgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8̘?@̘?H̘?Xb?model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)]?*28?@?H?XbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void wgrad2d_grouped_direct_kernel<float, float, float, (cudnnTensorFormat_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnConvolutionStruct, cudnnFilterStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int)*?2?8???@???H???Xbbgradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  ?B
f
redzone_checker*?2?@8???@??H??Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8???@ÇUH??WXbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28?ހ@?ހH?ހXbLgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2? 8˺?@??H??XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block5_branch1c_conv/Conv2Dhu  ?B
a
redzone_checker*?2?@8???@??H??Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28???@???H???Xb*model/bbn_features_transition4_conv/Conv2Dhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8???@???H???XbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*2?8???@??RH??SXbQgradient_tape/model/bbn_features_stemblock_stem1_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8???@??yH??zXbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8???@??wH??xXb*model/bbn_features_transition2_conv/Conv2Dhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8???@??vH??wXbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2? 8???@???H???XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8???@???H???XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
cvoid DSE::vector_fft<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)8 ??*2? 8???@ůrH??rXb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28???@???H???Xb*model/bbn_features_transition3_conv/Conv2Dhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28???@??qH??qXbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8???@???H???XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28???@???H???XbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu  HB
?
redzone_checker*?2?@8ɚ?@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8ʂ?@ʂ?Hʂ?Xbagradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2B8???@???H???Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xbagradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8ʹ?@ʹ?Hʹ?Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?A
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2 ?8???@??oH??oXbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8Ɍ?@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8Ɋ?@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
f
redzone_checker*?2?@8ʇ?@??H??Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8ɇ?@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  ?B
e
redzone_checker*?2?@8???@??H??Xb.model/bbn_features_stemblock_stem1_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??H??XbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbMgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void sgemm_largek_lds64<false, true, 5, 5, 4, 4, 4, 32>(float*, float const*, float const*, int, int, int, int, int, int, float const*, float const*, float, float, int, int, int*, int*)*?!*28???@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
redzone_checker*?2?@8???@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbMgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb(model/ssd_res_block1_branch2_conv/Conv2Dhu  HB
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
u
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xbmodel/ssd_cls5conv2/Conv2Dhu  HB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 8, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)|?R* 2X8???@??mH??nXbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???XbJgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8???@???H???Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu  ?B
s
redzone_checker*?2?@8???@??H??Xb<gradient_tape/model/ssd_cls5conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
s
redzone_checker*?2?@8???@??H??Xb<gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
s
redzone_checker*?2?@8???@??H??Xb<gradient_tape/model/ssd_cls4conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8ɜ?@??H??XbJgradient_tape/model/ssd_res_block3_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
s
redzone_checker*?2?@8ʘ?@??H??Xb<gradient_tape/model/ssd_cls3conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block5_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
s
redzone_checker*?2?@8???@??H??Xb<gradient_tape/model/ssd_box4conv2/Conv2D/Conv2DBackpropInputhu  ?B
t
redzone_checker*?2?@8???@??H??Xb=gradient_tape/model/ssd_cls5conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
t
redzone_checker*?2?@8???@??H??Xb=gradient_tape/model/ssd_cls4conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8ʂ?@??H??XbKgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block4_branch1c_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block2_branch1c_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbJgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
s
redzone_checker*?2?@8???@??H??Xb<gradient_tape/model/ssd_box2conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
t
redzone_checker*?2?@8???@??H??Xb=gradient_tape/model/ssd_cls3conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block5_branch1c_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block5_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
s
redzone_checker*?2?@8???@??H??Xb<gradient_tape/model/ssd_box5conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block4_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
t
redzone_checker*?2?@8???@??H??Xb=gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropFilterhu  ?B
s
redzone_checker*?2?@8???@??H??Xb<gradient_tape/model/ssd_box3conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block3_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbJgradient_tape/model/ssd_res_block5_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbJgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block3_branch1c_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbJgradient_tape/model/ssd_res_block4_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block5_branch1c_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block2_branch1c_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8???@??lH??lXbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block3_branch1c_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block4_branch1c_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2? 8???@???H???Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28???@??H??Xb=gradient_tape/model/ssd_cls4conv2/Conv2D/Conv2DBackpropFilterh?u  HB
?
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28???@??H??Xb=gradient_tape/model/ssd_cls3conv2/Conv2D/Conv2DBackpropFilterh?u  HB
?
?void implicit_convolve_sgemm<float, float, 512, 6, 8, 3, 3, 5, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)u?R* 2Q8ɢ?@??hH??hXb*model/bbn_features_transition3_conv/Conv2Dhu  HB
?
?void fft2d_c2r_32x32<float, true, false, 0u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)8 ??*?2 8˚?@??H??Xb.model/bbn_features_stemblock_stem1_conv/Conv2Dhdu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2? 8???@???H???XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2Q8???@???H???Xb?model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb<gradient_tape/model/ssd_cls3conv2/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb<gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb<gradient_tape/model/ssd_cls4conv2/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb<gradient_tape/model/ssd_cls5conv2/Conv2D/Conv2DBackpropInputhu  HB
m
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@???H???bmodel/re_lu_114/Reluhu  HB
m
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@???H???bmodel/re_lu_115/Reluhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28???@??BHÞCXbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 28???@âBH??BXbMgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?W8?޿@???H???Xb.model/bbn_features_stemblock_stem1_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?W8?˿@???H??@Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  HB
?
?void sgemm_largek_lds64<false, true, 5, 5, 4, 4, 4, 32>(float*, float const*, float const*, int, int, int, int, int, int, float const*, float const*, float, float, int, int, int*, int*)*?!*28???@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
?void sgemm_largek_lds64<false, true, 5, 5, 4, 4, 4, 32>(float*, float const*, float const*, int, int, int, int, int, int, float const*, float const*, float, float, int, int, int*, int*)*?!*28ǒ?@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8?ٽ@?ٽH?ٽXbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8ț?@ț?Hț?Xb?model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8Ɂ?@Ɂ?HɁ?Xbagradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xbagradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 5, 6, 4, 3, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)X?A*28?˼@?˼H?˼XbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu ??B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2B8?Ǽ@?ǼH?ǼXbbgradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8Ș?@Ș?HȘ?XbHgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropInputhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8?˻@?˻H?˻XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2Q8ȸ?@ȸ?Hȸ?Xb?model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@???H???XbHgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28ɦ?@í=H??>Xb*model/bbn_features_transition1_conv/Conv2Dhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2Q8ȹ?@ȹ?Hȹ?Xb?model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, true, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?28苸@苸H苸bmodel/re_lu_115/Reluhu  ?B
t
redzone_checker*?2?@8邷@??H??Xb=gradient_tape/model/ssd_box3conv2/Conv2D/Conv2DBackpropFilterh
u  ?B
?
redzone_checker*?2?@8ǵ?@??H??XbIgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropFilterh
u  ?B
I
redzone_checker*?2?@8紶@??H??bmodel/re_lu_113/Reluh
u  ?B
t
redzone_checker*?2?@8???@??H??Xb=gradient_tape/model/ssd_box5conv2/Conv2D/Conv2DBackpropFilterh
u  ?B
t
redzone_checker*?2?@8?ϵ@??H??Xb=gradient_tape/model/ssd_box4conv2/Conv2D/Conv2DBackpropFilterh
u  ?B
t
redzone_checker*?2?@8???@??H??Xb=gradient_tape/model/ssd_box2conv2/Conv2D/Conv2DBackpropFilterh
u  ?B
?
?void DSE::regular_fft_pad<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??*2? 8ȁ?@??XH??\Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterh
u  ?B
?
redzone_checker*?2?@8?Գ@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterh
u  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8罳@罳H罳Xb?model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2Dhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8???@??XH??ZXb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8?ծ@?ծH?ծXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb?model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2Q8???@???H???Xb?model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2B8ǂ?@ǂ?Hǂ?Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu  ?A
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 28B8???@???H???Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?˥@?˥H?˥XbKgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8???@???H???Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?a8?ͣ@??QH??RXb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)P?2* 28???@???H???XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)P?2* 28???@???H???XbMgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???XbKgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8???@??5H??6Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?
8?Ѡ@?ѠH?ѠXb?model/bbn_features_denseblock1_denselayer1_branch2a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8?Р@??4H??6Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xbmodel/ssd_cls3conv2/Conv2Dhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2,8桟@??OH??OXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8?ޞ@?ޞH?ޞXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, true, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2$8Ǜ?@Ǜ?HǛ?bmodel/re_lu_114/Reluhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8洝@洝H洝Xbagradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xbagradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28???@??H??XbKgradient_tape/model/ssd_res_block3_branch2_conv/Conv2D/Conv2DBackpropFilterh?u  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8???@??4H??4XbRgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?ܛ@?ܛH?ܛXb?model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2Dhu  HB
?
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28?ӛ@??H??XbKgradient_tape/model/ssd_res_block4_branch2_conv/Conv2D/Conv2DBackpropFilterh?u  HB
?
&cudnn_dgrad_sm35_ldg_nt_64x16x128x8x32{?`* 2$8???@??LH??LXbHgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8ǟ?@??2H??3XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2B8???@???H???Xbbgradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8???@???H???Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?A
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)P?2* 2?
8???@???H???XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?@?H?Xb)model/ssd_res_block2_branch1a_conv/Conv2Dhu  HB
?
redzone_checker*?2?@8???@??H??XbRgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbQgradient_tape/model/bbn_features_stemblock_stem1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8?ȓ@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?X8???@???H???XbKgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B

redzone_checker*?2?@8???@??H??XbHgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
.cudnn_convolve_sgemm_sm35_ldg_nn_32x16x64x8x16S?2*2?8???@???H???Xb*model/bbn_features_transition2_conv/Conv2Dhu ??B
?
?void DSE::regular_fft_clip<1, 2, 256, 16, 16, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??*2? 8???@???H???XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb?model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)P?2* 2?
8?@?H?Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2B8ư?@ư?Hư?Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu  ?A
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock2_denselayer1_branch1b_conv/Conv2Dhu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 20B8損@損H損Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2,?8???@???H???Xb(model/ssd_res_block2_branch2_conv/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8???@???H???XbJgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2@B8?ʇ@??CH??CXbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)?*?2?8?݄@?݄H?݄bFgradient_tape/model/bbn_features_transition1_norm/FusedBatchNormGradV3hu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8捄@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputh	u  HB
?
?void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2? 8???@???H???bmodel/re_lu_113/Reluhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 256, 16, 16, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??*2? 8???@???H???XbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28???@??AH??AXbLgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb?model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2Dhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2? 8???@???H???XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8?؀@?؀H?؀Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 4, 6, 3, 2, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)O? *2,8??@??H??XbLgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2? 8??~@??~H??~XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*2X8??~@??~H??~XbQgradient_tape/model/bbn_features_stemblock_stem1_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??~@??~H??~Xbagradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28??}@?jH??XbLgradient_tape/model/ssd_res_block3_branch1c_conv/Conv2D/Conv2DBackpropFilterh?u  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)P?2* 2?8??}@??}H??}XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*2X8??|@??|H??|XbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
cvoid DSE::vector_fft<1, 2, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)5 ??*2? 8??|@??|H??|XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
cvoid DSE::vector_fft<1, 2, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)5 ??*2? 8??|@??|H??|XbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2? 8??|@??|H??|Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??|@??|H??|Xb=gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropFilterhu  HB
?
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28Ɔ|@?jH??XbLgradient_tape/model/ssd_res_block4_branch1c_conv/Conv2D/Conv2DBackpropFilterh?u  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??z@??zH??zXb=gradient_tape/model/ssd_cls5conv2/Conv2D/Conv2DBackpropFilterhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2B8??z@??zH??zXbbgradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??y@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch2b_conv/Conv2Dh	u  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)H?*2?W8??y@??yH??yXbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8Ŋy@ŊyHŊyXbKgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??x@??xH??xXb?model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2Dhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??x@??;H??<Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
:cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_64x8x256x8x32}?R* 28??x@??;H??<Xb*model/bbn_features_transition4_conv/Conv2Dhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2? 8??x@??xH??xXb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 0u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?28??v@??H??Xb.model/bbn_features_stemblock_stem1_conv/Conv2Dhdu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)P?2* 2?
8??v@??;H??;XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??u@??uH??uXbKgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2B8??u@??uH??uXb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu  ?A
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??u@??uH??uXbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??t@??tH??tXbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)?*?2 8??t@??tH??tbJgradient_tape/model/bbn_features_stemblock_stem1_norm/FusedBatchNormGradV3hu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2&8Ųt@??H??Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??t@??tH??tXbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)]?*28??s@??sH??sXbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
cvoid DSE::vector_fft<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)8 ??*2? 8??r@??rH??rXbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2(B8??r@??rH??rXbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
cvoid DSE::vector_fft<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)8 ??*2? 8??r@??rH??rXbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
0cudnn_convolve_sgemm_sm35_ldg_nn_128x8x128x16x16??A*2?8??r@??rH??rXb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8Ńr@ŃrHŃrXb)model/ssd_res_block1_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??q@??%H??&Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?@8Ěq@ĚqHĚqXbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??q@??qH??qXbJgradient_tape/model/ssd_res_block4_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??q@??qH??qXbJgradient_tape/model/ssd_res_block5_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??p@??pH??pXb(model/ssd_res_block3_branch2_conv/Conv2Dhu  HB
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??p@??8H¸8Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??p@??pH??pXb(model/ssd_res_block5_branch2_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??p@??pH??pXb(model/ssd_res_block4_branch2_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??p@??pH??pXbJgradient_tape/model/ssd_res_block3_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2 ?8??p@??pH??pXb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2??8??o@??oH??oXb*model/bbn_features_transition4_conv/Conv2Dhu  H?
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2,8ċo@??7H??7Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8??n@??nH??nXb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8Ĝn@ĜnHĜnXbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8??n@??nH??nXbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8??n@??nH??nXbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8??m@??mH??mXbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?A8??m@??6H??6Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8??m@??mH??mXb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8??l@??lH??lXbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8??l@??lH??lXb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
:cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_64x8x256x8x32}?R* 2Q8??l@??lH??lXb*model/bbn_features_transition3_conv/Conv2Dhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8ūl@ūlHūlXbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?%8??l@??!H??%XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*  2?8??k@??kH??kXbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??k@??kH??kXb?model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2Dhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?@8??k@??kH??kXbKgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??j@ß5H??5Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cudnn_convolve_sgemm_sm35_ldg_nn_64x16x64x16x16O?B*2Q8??i@??iH??iXb*model/bbn_features_transition3_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8đi@điHđiXb)model/ssd_res_block1_branch1b_conv/Conv2Dhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2 ?8??h@??hH??hXb(model/ssd_res_block1_branch2_conv/Conv2Dhu  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?%8??h@?? H??#Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?A
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??f@??fH??fXbbgradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??f@??fH??fXbbgradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??e@??eH??eXb)model/ssd_res_block3_branch1b_conv/Conv2Dhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??e@??eH??eXb<gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?<8??e@??eH??eXb=gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??e@??eH??eXbKgradient_tape/model/ssd_res_block4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??e@??eH??eXb)model/ssd_res_block4_branch1b_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*28??e@?? H??#bmodel/re_lu_113/Reluhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??e@??eH??eXbKgradient_tape/model/ssd_res_block5_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??e@??eH??eXb)model/ssd_res_block5_branch1b_conv/Conv2Dhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??e@??eH??eXb<gradient_tape/model/ssd_cls5conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8Ļe@ĻeHĻeXbKgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??e@??eH??eXb<gradient_tape/model/ssd_cls4conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??e@??eH??eXbKgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??e@??eH??eXb<gradient_tape/model/ssd_cls3conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?<8??e@??eH??eXb=gradient_tape/model/ssd_cls3conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?<8??d@??dH??dXb=gradient_tape/model/ssd_cls4conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*208??d@??!H??"Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?<8??d@??dH??dXb=gradient_tape/model/ssd_cls5conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??d@??dH??dXbJgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??c@0H??2Xb?model/bbn_features_denseblock1_denselayer1_branch2b_conv/Conv2Dhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??b@??bH??bXbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??b@??bH??bXbJgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2B8??b@??bH??bXb?model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2Dhu  ?A
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??`@??`H??`Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)P?2* 28??`@??`H??`XbHgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??_@??_H??_Xbagradient_tape/model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?%8??_@??H"XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??_@??_H??_Xbmodel/ssd_cls4conv2/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8ĩ_@ĩ_Hĩ_Xbmodel/ssd_cls3conv2/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??_@??_H??_Xbmodel/ssd_cls1conv2/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??_@??_H??_Xbmodel/ssd_cls5conv2/Conv2Dhu  ?B

/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??^@??^H??^Xb(model/ssd_res_block2_branch2_conv/Conv2Dhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2 B8??]@??]H??]Xbagradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8??\@??.H??.Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?%8Ú\@??H?? Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?A
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8ħ[@??H??Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?A
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??Z@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1b_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
?void DSE::regular_fft_pad<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??*2? 8??Z@??ZH??ZXbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??Z@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch2c_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
?void DSE::regular_fft_pad<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??*2? 8??W@??WH??WXbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2?8??W@??+H??+Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2
?8ĊW@ĊWHĊWXbagradient_tape/model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2
?8??V@??VH??VXbagradient_tape/model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??V@??VH??VXbbgradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??U@??UH??UXbRgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??U@??UH??UXbbgradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8??U@??*H??*XbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*208??T@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??S@??SH??SXbLgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)\?*28??S@??SH??SXbMgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2&8??R@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?a8??R@??RH??RXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?a8??R@??RH??RXbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_r2c_32x32<float, true, 0u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)? ??*?2 8ÑR@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?8??Q@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*208??Q@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?a8??Q@??QH??QXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?a8??Q@??QH??QXbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?8??Q@??QH??QXbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8??P@??'H??(Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2,8??O@??OH??OXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??O@??OH??OXb?model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??O@??OH??OXbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2	?8??N@??NH??NXbagradient_tape/model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2	?8??N@??NH??NXbagradient_tape/model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)P?2* 2?
8??M@??MH??MXbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8äM@??H??Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?A
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??L@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch1b_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??K@??KH??KXb?model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2Dhu  HB
?
?void fft2d_c2r_32x32<float, false, false, 0u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)8 ??*?28??K@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2 ?8ÆK@ÆKHÆKXbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??J@??JH??JXbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void DSE::regular_fft_clip<1, 2, 256, 16, 16, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??*2?8??J@??JH??JXb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?,8??J@??JH??JXbLgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2)8??J@??H??Xb)model/ssd_res_block1_branch1b_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??I@??H??Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  HB
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2?88??I@??IH??IXbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8??H@??HH??HXb?model/bbn_features_denseblock2_denselayer1_branch2c_conv/Conv2Dhu  ?B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2? 8??H@??HH??HXb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu ??B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??H@??HH??HXbJgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??H@??HH??HXb?model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??G@??GH??GXbbgradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB

/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??G@??GH??GXb(model/ssd_res_block1_branch2_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??G@??GH??GXbbgradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
q
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??G@??GH??GXbmodel/ssd_cls1conv2/Conv2Dhu  HB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 8, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)|?R* 28??G@??GH??GXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*28??F@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?8??F@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??E@??EH??EXbagradient_tape/model/bbn_features_denseblock4_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 28þE@??H??XbKgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2,?8??E@??EH??EXb)model/ssd_res_block2_branch1a_conv/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??E@??EH??EXbKgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??E@??EH??EXbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*208??D@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*2$8??D@??DH??DXbIgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 512, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float) ?*?2?8??D@??DH??Db4model/bbn_features_transition1_norm/FusedBatchNormV3hu  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??C@??H??XbHgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropInputh$u  ?B
?
hvoid fft1d_r2c_32<float, float, float2, true, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?y8??C@??CH??CXb*model/bbn_features_transition4_conv/Conv2Dhu  ?A
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8??C@??!H??!Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28ÓC@ÓCHÓCXbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??B@??BH??BXb<gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??B@??BH??BXb?model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2Dhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??B@??BH??Bbmodel/re_lu_114/Reluhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??B@??BH??Bbmodel/re_lu_115/Reluhu  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?y8??B@??BH??BXbLgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??B@??BH??BXbHgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??B@??BH??BXbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??B@??BH??BXbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??B@??BH??BXbHgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 512, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float) ?*?2 8??B@??BH??Bb8model/bbn_features_stemblock_stem1_norm/FusedBatchNormV3hu  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??B@??H??XbHgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropInputh$u  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?%8??A@??Hµ"XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8ÅA@ÅAHÅAXbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 6, 7, 3, 3, 5, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)M?2* 28??@@??@H??@Xb*model/bbn_features_transition4_conv/Conv2Dhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??@@?? H?? Xb?model/bbn_features_denseblock1_denselayer1_branch2a_conv/Conv2Dhu  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?%8??@@??H??"Xb*model/bbn_features_transition2_conv/Conv2Dhu  ?A
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2 ?8??@@??@H??@Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??@@??@H??@Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2E8??@@??@H??@Xbmodel/ssd_cls4conv2/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*28???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb?model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2Dhu  HB
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2?08???@???H???Xbbgradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 5, 6, 4, 3, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)X?A*2+8???@??H??XbHgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropInputhu ??B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??>@??>H??>Xb?model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2Dhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8??>@??>H??>Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*2	8ù>@ù>Hù>Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2? 8î>@î>Hî>Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu ??B
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)?*?28??>@??>H??>bKgradient_tape/model/bbn_features_stemblock_stem2a_norm/FusedBatchNormGradV3hu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??>@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
cvoid DSE::vector_fft<1, 2, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)5 ??*2?8??>@??>H??>Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??>@??>H??>Xb)model/ssd_res_block1_branch1c_conv/Conv2Dhu  HB
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 4, 6, 3, 2, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)O? *2,8??=@??=H??=XbJgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*28=@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??=@??=H??=XbLgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)M?2* 2$8??<@??H??XbIgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropFilterhu  ?B
j
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8½<@½<H½<bmodel/re_lu_114/Reluhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??<@??<H??<XbKgradient_tape/model/ssd_res_block4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??<@??<H??<XbKgradient_tape/model/ssd_res_block5_branch2_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28Ç<@Ç<HÇ<Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??;@??;H??;Xb)model/ssd_res_block5_branch1c_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??;@??;H??;Xb)model/ssd_res_block3_branch1c_conv/Conv2Dhu  HB
j
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??;@??;H??;bmodel/re_lu_115/Reluhu  HB
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??;@??;H??;Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??;@??;H??;XbKgradient_tape/model/ssd_res_block3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??;@??;H??;Xb)model/ssd_res_block4_branch1c_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8;@;H;Xb?model/bbn_features_denseblock4_denselayer1_branch2a_conv/Conv2Dhu  HB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?28??;@??;H??;b%gradient_tape/model/re_lu_19/ReluGradhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??;@??;H??;XbKgradient_tape/model/ssd_res_block5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?28??;@??;H??;b"gradient_tape/model/re_lu/ReluGradhu  ?B
?
?void fft2d_c2r_32x32<float, true, false, 0u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)8 ??*?2 8??;@??H??Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  ?B
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2?8;@??H??Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??:@??:H??:Xbagradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2??8??:@??:H??:Xb*model/bbn_features_transition3_conv/Conv2Dhu  H?
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2,8??:@??:H??:XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??:@??:H??:XbKgradient_tape/model/ssd_res_block2_branch1c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2,8??:@??HXbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::pooling_bw_kernel_max<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor) ? *?2,8??:@??:H??:b5gradient_tape/model/max_pooling2d/MaxPool/MaxPoolGradhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8ß:@ß:Hß:XbHgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropInputhu  HB
~
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??:@??:H??:Xb&model/ssd_feature_extend3_conv1/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??:@??:H??:XbHgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??9@??9H??9Xbbgradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??9@??9H??9XbKgradient_tape/model/ssd_res_block5_branch1c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??9@??9H??9Xb)model/ssd_res_block5_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??9@??9H??9XbKgradient_tape/model/ssd_res_block3_branch1c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??9@??9H??9Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??9@??9H??9Xb)model/ssd_res_block3_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??9@??9H??9Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??9@??9H??9XbKgradient_tape/model/ssd_res_block4_branch1c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??9@??9H??9Xb?model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2Dhu  ?B
?
cvoid DSE::vector_fft<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)8 ??*2?8??9@??9H??9XbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??9@??H??Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8??9@??9H??9Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??9@??9H??9Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
.cudnn_convolve_sgemm_sm35_ldg_nn_32x16x64x8x16S?2*2)8??9@??9H??9Xb*model/bbn_features_transition4_conv/Conv2Dhu ??B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8??9@??H??Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu  HB
?
cvoid DSE::vector_fft<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)8 ??*2?8??9@??9H??9XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??8@??8H??8Xbbgradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??8@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??8@??8H??8Xbbgradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*2[8??8@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??8@??8H??8Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?88@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch2a_conv/Conv2Dhu  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2? 8??7@??H??Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu  ?A
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??7@??7H??7XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??7@??7H??7Xbbgradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??7@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)H?*2?8??7@??7H??7Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??7@??7H??7Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??7@??7H??7Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2,8??7@??7H??7Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+87@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?A8??7@??7H??7XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?	
?	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??7@??7H??7bAdam/gradients/AddN_30hu  ?B
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?A8??7@??7H??7Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?A8??6@??6H??6Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??6@??6H??6Xbagradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?A8??6@??6H??6Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??6@??6H??6Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch2a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??6@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch2c_conv/Conv2Dhu  HB
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2? 8??6@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch2a_conv/Conv2Dhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?A8??6@??6H??6XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_r2c_32x32<float, false, 0u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?28??6@??H??Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??6@??6H??6bAdam/gradients/AddN_33hu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?A8»6@»6H»6Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?A8??6@??6H??6Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8??6@??H??Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2? 8??6@??6H??6XbLgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2? 8??6@??6H??6XbKgradient_tape/model/ssd_res_block3_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2? 8??6@??6H??6XbKgradient_tape/model/ssd_res_block4_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??6@??H??Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  HB
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2 ?8??6@??6H??6Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?8??5@??5H??5Xbbgradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??5@??5H??5Xb?model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??5@??5H??5Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch2b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??5@??5H??5Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??5@??5H??5Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??5@??5H??5Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch2b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2? 8??5@??5H??5XbKgradient_tape/model/ssd_res_block5_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8ü5@??H??Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?A
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??5@??5H??5Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch2b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??5@??5H??5Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8­5@­5H­5Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8ë5@ë5Hë5Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch2c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??5@??5H??5Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch2b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*2
8ç5@ç5Hç5Xbbgradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??5@??5H??5Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??5@??5H??5Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch2b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 28??5@??H??XbLgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??5@??5H??5Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch2b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??5@??5H??5Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??5@??5H??5Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch2c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??5@??5H??5Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
85@5H5Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch2c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2? 85@5H5XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu ??B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??4@??4H??4XbKgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2?(8??4@??4H??4Xbbgradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2 ?8??4@??4H??4Xb)model/ssd_res_block1_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??4@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*2[8??4@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch2b_conv/Conv2Dh$u  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*28??4@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)?*?2?8??4@??4H??4bFgradient_tape/model/bbn_features_transition2_norm/FusedBatchNormGradV3hu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*208á4@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2? 8??4@??4H??4Xb*model/bbn_features_transition2_conv/Conv2Dhu ??B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2? 8??4@??4H??4Xb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu ??B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??3@??3H??3Xbbgradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8??3@??H??Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?A
?
&cudnn_dgrad_sm35_ldg_nt_64x16x128x8x32{?`* 28??3@??H??XbKgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8??3@??3H??3Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  HB
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?%8??3@??H??Xb*model/bbn_features_transition2_conv/Conv2Dhu  ?A
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??3@??3H??3Xbbgradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
[void cudnn::ops::scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*?2??8??2@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?%8??2@??H??XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??2@??2H??2XbKgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??2@??2H??2Xb(model/ssd_res_block5_branch2_conv/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??2@??2H??2Xb(model/ssd_res_block4_branch2_conv/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??2@??2H??2XbJgradient_tape/model/ssd_res_block4_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??2@??2H??2Xb(model/ssd_res_block3_branch2_conv/Conv2Dhu  ?B
?
[void cudnn::ops::scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*?2??8??2@??H??XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??2@??2H??2XbJgradient_tape/model/ssd_res_block5_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??2@??2H??2XbJgradient_tape/model/ssd_res_block3_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8½2@½2H½2Xb?model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??2@??2H??2Xbagradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??2@??2H??2Xbbgradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28??2@??H??XbJgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28ª2@ª2Hª2Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?8??2@??2H??2Xbbgradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?82@??H??Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??1@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch1b_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8??1@??H??Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8??1@??H??Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu  ?A
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8??1@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?8??1@??H??XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*28??1@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::pooling_bw_kernel_avg<float, float, cudnn::averpooling_func<float, true>, 2, false>(cudnnTensorStruct, float const*, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)( ? *?2 81@1H1b1gradient_tape/model/average_pooling2d/AvgPoolGradhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 28??0@??H??XbKgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??0@??0H??0Xbbgradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?8??0@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?8??0@??0H??0Xbbgradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2)8??0@??H??Xb(model/ssd_res_block2_branch2_conv/Conv2Dhu  HB
q
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28??0@??H??Xb)model/ssd_res_block1_branch1b_conv/Conv2Dh$u  ?B
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??0@??0H??0Xb?model/bbn_features_denseblock1_denselayer1_branch2c_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??0@??0H??0Xb?model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?80@0H0Xb?model/bbn_features_denseblock1_denselayer2_branch1b_conv/Conv2Dhu  HB
q
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??0@??0H??0Xbmodel/ssd_cls3conv2/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??0@??0H??0Xb?model/bbn_features_denseblock1_denselayer3_branch2b_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??0@??0H??0Xb?model/bbn_features_denseblock1_denselayer2_branch2c_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?80@0H0Xb?model/bbn_features_denseblock1_denselayer3_branch2c_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?80@0H0Xb?model/bbn_features_denseblock1_denselayer1_branch1b_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??0@??0H??0Xb?model/bbn_features_denseblock1_denselayer2_branch2b_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 28??/@??H??Xb=gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)H?*2?8??/@??/H??/Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??/@??/H??/Xb)model/ssd_res_block2_branch1a_conv/Conv2Dhu  HB
?
?void tensorflow::functor::PadInputCustomKernelNCHW<float, 4>(int, float const*, tensorflow::functor::Dimension<4>, float*, tensorflow::functor::Dimension<4>, tensorflow::functor::Dimension<(4)-(2)>, float)*?28??/@??/H??/Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  ?B
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2?8¦/@¦/H¦/Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu ??B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??.@??H??XbLgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??.@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch1b_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
?void tensorflow::functor::PadInputCustomKernelNCHW<float, 4>(int, float const*, tensorflow::functor::Dimension<4>, float*, tensorflow::functor::Dimension<4>, tensorflow::functor::Dimension<(4)-(2)>, float)*?28??.@??.H??.XbRgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??.@??.H??.Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void DSE::regular_fft_pad<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??*2?8§.@§.H§.XbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void tensorflow::functor::PadInputCustomKernelNCHW<float, 4>(int, float const*, tensorflow::functor::Dimension<4>, float*, tensorflow::functor::Dimension<4>, tensorflow::functor::Dimension<(4)-(2)>, float)*?28??.@??.H??.XbQgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?	8??.@??H??	XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??.@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch2c_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 28??-@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??*2?8??-@??-H??-XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 4, 6, 3, 2, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)O? *2 8??-@??-H??-XbJgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2? 8??-@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??-@??-H??-Xb<gradient_tape/model/ssd_cls3conv2/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??-@??-H??-Xb?model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8??-@??-H??-Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch2a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??-@??-H??-Xbagradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
q
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28??-@??H??Xb)model/ssd_res_block3_branch1b_conv/Conv2Dh$u  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*208-@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
Z
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28??,@??H??bmodel/re_lu_114/Reluh$u  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*28??,@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_r2c_32x32<float, false, 0u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??,@??H??XbMgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
Z
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28??,@??H??bmodel/re_lu_115/Reluh$u  ?B
q
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28??,@??H??Xb)model/ssd_res_block5_branch1b_conv/Conv2Dh$u  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??,@??H??XbKgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??+@??+H??+Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*28??+@??H??bmodel/re_lu_114/Reluhu  HB
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??+@??H??XbKgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
q
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28??+@??H??Xb)model/ssd_res_block4_branch1b_conv/Conv2Dh$u  ?B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8??+@??+H??+XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 28??+@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2 ?8??+@??+H??+Xbagradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?8??+@??+H??+Xbbgradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8??+@??H??Xb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu  HB
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8??*@??*H??*Xb*model/bbn_features_transition2_conv/Conv2Dhu ??B
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2? 8??*@??*H??*Xbbgradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??*@??H??XbKgradient_tape/model/ssd_res_block4_branch1b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 4, 6, 3, 2, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)O? *28??*@??*H??*Xb<gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?	8??*@??H??Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?A
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??*@??H??XbKgradient_tape/model/ssd_res_block5_branch1b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?228??*@??*H??*Xb)model/ssd_res_block3_branch1a_conv/Conv2Dhu  ?B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2? 8??*@??*H??*Xb?model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2Dhu ??B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8??)@??H??Xb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu  ?A
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?
8??)@??)H??)Xb?model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?
8??)@??)H??)Xbagradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?28??)@??)H??)Xbmodel/ssd_box1conv2/Conv2Dhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*28×)@??H??Xb)model/ssd_res_block3_branch1b_conv/Conv2Dhu  HB
?
?void wgrad2d_grouped_direct_kernel<float, float, float, (cudnnTensorFormat_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnConvolutionStruct, cudnnFilterStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int)*?2?	8??(@??(H??(Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2?8??(@??(H??(Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2?8??(@??(H??(Xb?model/bbn_features_denseblock1_denselayer1_branch2b_conv/Conv2Dhu  HB
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2??8??(@??(H??(Xb(model/ssd_res_block2_branch2_conv/Conv2Dhu  H?
?
hvoid fft1d_r2c_32<float, float, float2, true, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?@8??(@??(H??(Xb*model/bbn_features_transition3_conv/Conv2Dhu  ?A
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8??(@??(H??(Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch2a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2	8??(@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)H?*2?8??(@??(H??(Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2
?8??(@??(H??(Xbbgradient_tape/model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2
?8??'@??'H??'Xbbgradient_tape/model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 28??'@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*?28??'@??H??bmodel/concatenate_3/concathu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, false, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)P?*28??'@??H??Xb)model/ssd_res_block5_branch1b_conv/Conv2Dhu  HB
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?
8??'@??'H??'Xbbgradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8??'@??H??Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu  HB
?
&cudnn_dgrad_sm35_ldg_nt_64x16x128x8x32{?`* 28??'@??'H??'XbHgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28'@'H'XbKgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?8'@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8??&@??&H??&Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*?28??&@??&H??&bmodel/re_lu_19/Reluhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*?28??&@??&H??&bmodel/re_lu/Reluhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??&@??H??XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?	8??&@??&H??&Xb?model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28??&@??H??XbJgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void wgrad2d_grouped_direct_kernel<float, float, float, (cudnnTensorFormat_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnConvolutionStruct, cudnnFilterStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int)*?2?8??%@??	H??	Xb=gradient_tape/model/ssd_box3conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
[void cudnn::ops::scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*?2և8??%@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8??%@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??%@??%H??%Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??%@??%H??%Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch2a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8??%@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::im2col4d_kernel<float, long>(cudnn::cnn::im2col4d_params, cudnnConvolutionStruct, cudnnTensor4dStruct, float const*, float*) *?2?8??%@??%H??%Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?	8??%@??%H??%Xbagradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*28??%@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8??%@??%H??%Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu ??B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28¿%@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 28??%@??H??XbLgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?8??%@??%H??%Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*28??%@??H??bmodel/re_lu_115/Reluhu  HB
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 5, 6, 4, 3, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)X?A*2+8??%@??%H??%XbKgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*28??%@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?	
?	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??$@??$H??$bAdam/gradients/AddN_31hu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??$@??$H??$Xb)model/ssd_res_block1_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??$@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch2a_conv/Conv2Dhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?@8¤$@¤$H¤$XbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2	?8??$@??$H??$Xbbgradient_tape/model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2)8??$@??H??Xbmodel/ssd_cls1conv2/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2	?8$@$H$Xbbgradient_tape/model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?	8??$@??$H??$Xbbgradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2Q8??#@??#H??#Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*28??#@??H??Xb)model/ssd_res_block4_branch1b_conv/Conv2Dhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??#@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void tensorflow::functor::PadInputCustomKernelNCHW<float, 4>(int, float const*, tensorflow::functor::Dimension<4>, float*, tensorflow::functor::Dimension<4>, tensorflow::functor::Dimension<(4)-(2)>, float)*?28??#@??#H??#Xb.model/bbn_features_stemblock_stem1_conv/Conv2Dhu  ?B
?
?void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 512, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float) ?*?28??#@??#H??#b9model/bbn_features_stemblock_stem2a_norm/FusedBatchNormV3hu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?8??#@??#H??#Xbbgradient_tape/model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8??#@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 5, 6, 4, 3, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)X?A*2+8??#@??H??XbKgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28??#@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??#@??#H??#Xb)model/ssd_res_block1_branch1b_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)\?*28??#@??#H??#XbKgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??"@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28??"@??H??Xb<gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)P?2* 2?
8??"@??"H??"Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??"@??"H??"Xbagradient_tape/model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 8, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)|?R* 28??"@??"H??"Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
9cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_32x16x32x8x8D?$*2?8??"@??"H??"Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  HB
?
?void tensorflow::functor::PadInputCustomKernelNCHW<float, 4>(int, float const*, tensorflow::functor::Dimension<4>, float*, tensorflow::functor::Dimension<4>, tensorflow::functor::Dimension<(4)-(2)>, float)*?28??"@??"H??"XbQgradient_tape/model/bbn_features_stemblock_stem1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??"@??"H??"XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8??"@??H??Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu  HB
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2?8´"@´"H´"Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8??"@??"H??"Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 5, 6, 4, 3, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)X?A*2+8??"@??H??XbKgradient_tape/model/ssd_res_block4_branch1b_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??"@??"H??"Xb?model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2Dhu  ?B
?
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28"@?rH??Xb?model/bbn_features_denseblock4_denselayer1_branch2b_conv/Conv2Dh$u  ?B
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2?8??"@??H??Xb?model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2Dhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8??!@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2Dhu  ?A
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??!@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch2b_conv/Conv2Dhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??!@??!H??!XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??!@??!H??!Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??!@??!H??!XbKgradient_tape/model/ssd_res_block5_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?8??!@??!H??!Xbbgradient_tape/model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??!@??!H??!Xbagradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??!@??!H??!Xb)model/ssd_res_block3_branch1b_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??!@??!H??!XbKgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??!@??!H??!Xb)model/ssd_res_block4_branch1b_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??!@??!H??!XbKgradient_tape/model/ssd_res_block4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??!@??H??Xb?model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2Dhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??!@??!H??!Xb)model/ssd_res_block5_branch1b_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??!@??!H??!XbKgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28??!@?qH??Xb?model/bbn_features_denseblock3_denselayer1_branch1b_conv/Conv2Dh$u  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8??!@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch2a_conv/Conv2Dhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2*@8??!@??!H??!Xb?model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2Dhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??!@??!H??!Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8??!@??!H??!Xbbgradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8??!@??!H??!Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch2a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2Q8??!@??!H??!Xbagradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??!@??!H??!Xbagradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??!@??!H??!Xbagradient_tape/model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?8?? @??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?? @?? H?? Xb?model/bbn_features_denseblock3_denselayer1_branch1b_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28?? @?? H?? Xbagradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
9cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_32x16x32x8x8D?$*2?8?? @??H??Xb?model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*208?? @?? H?? Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch2a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?? @?? H?? Xbbgradient_tape/model/bbn_features_denseblock4_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?8?? @?? H?? XbHgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28?? @?nH??XbLgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28?? @?? H?? b'gradient_tape/model/concatenate_3/Slicehu  ?B
?
9cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_32x16x32x8x8D?$*2?8?? @?? H?? Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8?? @?? H?? Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8?? @?? H?? Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu ??B
?
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28?? @?nH??Xb?model/bbn_features_denseblock2_denselayer1_branch1b_conv/Conv2Dh$u  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?8 @ H Xbbgradient_tape/model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28?? @??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2?8?? @??H??Xb?model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?? @?? H?? Xb)model/ssd_res_block1_branch1b_conv/Conv2Dhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2(@8??@??H??Xb?model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8??@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch2b_conv/Conv2Dhu  ?A
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?8??@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??@??H??Xb?model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2Dhu  HB
?
?void cub::DeviceSegmentedRadixSortKernel<cub::DeviceRadixSortPolicy<float, int, int>::Policy700, true, true, float, int, cub::TransformInputIterator<int, tensorflow::impl::SegmentOffsetCreator, cub::CountingInputIterator<int, long>, long>, int>(float const*, float*, int const*, int*, cub::TransformInputIterator<int, tensorflow::impl::SegmentOffsetCreator, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::impl::SegmentOffsetCreator, cub::CountingInputIterator<int, long>, long>, int, int, int)`?D*?28??@??H??b2compute_loss/cond/else/_1/compute_loss/cond/TopKV2hu  zB
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)H?*2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbKgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?	
?	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@??H??bAdam/gradients/AddN_26hu  ?B
?
9cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_32x16x32x8x8D?$*2)8??@??H??Xb)model/ssd_res_block2_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xb)model/ssd_res_block4_branch1b_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xb)model/ssd_res_block5_branch1b_conv/Conv2Dhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?$8??@??H??Xb*model/bbn_features_transition3_conv/Conv2Dhu  ?A
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbLgradient_tape/model/ssd_res_block4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbLgradient_tape/model/ssd_res_block2_branch1c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbLgradient_tape/model/ssd_res_block3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbIgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropFilterhu  HB
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 1024, 2, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?`*?2?8??@??H??bwgradient_tape/model/bbn_features_stemblock_stem1_conv/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2?8??@??H??Xb?model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2Dhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?$8??@??H??XbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbIgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2Q8??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbLgradient_tape/model/ssd_res_block5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbLgradient_tape/model/ssd_res_block5_branch1c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
9cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_32x16x32x8x8D?$*2?8??@??H??Xb?model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??@??H??Xb?model/bbn_features_denseblock4_denselayer1_branch2b_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbKgradient_tape/model/ssd_res_block4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbKgradient_tape/model/ssd_res_block5_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2??8??@??H??Xb(model/ssd_res_block1_branch2_conv/Conv2Dhu  H?
r
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xbmodel/ssd_box1conv2/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xb)model/ssd_res_block3_branch1b_conv/Conv2Dhu  HB
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 5, 6, 4, 3, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)X?A*2+8??@??H??XbHgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropInputhu ??B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2&@8??@??H??Xb?model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2Dhu  ?B
?
:cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_64x8x256x8x32}?R* 28??@??H??Xb(model/ssd_res_block1_branch2_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??@??H??Xb?model/bbn_features_denseblock3_denselayer1_branch1b_conv/Conv2Dhu  HB
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbKgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?8??@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8??@??H??Xb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)\?*28??@??H??XbKgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*?28??@??H??bmodel/concatenate_2/concathu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2)8??@??H??Xb(model/ssd_res_block2_branch2_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)\?*28??@??H??Xb=gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropFilterhu  HB
r
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xbmodel/ssd_box4conv2/Conv2Dhu  HB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2	8??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2Q8??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch2a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)G?*2$8??@??H??XbLgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8??@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2?8??@??H??Xb?model/bbn_features_denseblock1_denselayer3_branch2a_conv/Conv2Dhu  HB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?28??@??H??b$gradient_tape/model/re_lu_1/ReluGradhu  ?B
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)?*?2 8??@??H??bKgradient_tape/model/bbn_features_stemblock_stem2b_norm/FusedBatchNormGradV3hu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8??@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
r
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xbmodel/ssd_box5conv2/Conv2Dhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28@??H??XbKgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2,8??@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
r
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xbmodel/ssd_box3conv2/Conv2Dhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2$@8??@??H??Xb?model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
hvoid fft1d_r2c_32<float, float, float2, true, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?,8??@??H??Xb(model/ssd_res_block2_branch2_conv/Conv2Dhu  ?A
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)?*?2 8??@??H??bJgradient_tape/model/bbn_features_stemblock_stem3_norm/FusedBatchNormGradV3hu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*208??@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer3_branch2a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?!8??@??H??Xb?model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2Dhu  ?A
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?8??@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 28??@??	H??	XbLgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2Q8??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2"@8??@??H??Xb?model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??@??H??Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??@??H??Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch2b_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??@??H??XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8??@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 5, 6, 4, 3, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)X?A*28??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1b_conv/Conv2D/Conv2DBackpropInputhu ??B
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2??8??@??H??Xbmodel/ssd_cls5conv2/Conv2Dhu  H?
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?	8??@??H??
XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)( ? *?2,8??@??H??bmodel/max_pooling2d/MaxPoolhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*208??@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer1_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?28??@??H??b%gradient_tape/model/re_lu_40/ReluGradhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*208??@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch2b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??@??	H??
Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void pooling_fw_4d_kernel<float, float, cudnn::averpooling_func<float, true>, (cudnnPoolingMode_t)2, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)0 ?*?2 8??@??H??bmodel/average_pooling2d/AvgPoolhu?O?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??@?^H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??@??H??Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*208??@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer3_branch2b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?	8??@??H??	Xb?model/bbn_features_denseblock1_denselayer1_branch2a_conv/Conv2Dhu  ?A