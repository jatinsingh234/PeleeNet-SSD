
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?W8???"@???"H???"Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2??8???@???H???XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb*model/bbn_features_transition3_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???XbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@???H???XbHgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropInputhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2@B8???@???H???Xb*model/bbn_features_transition2_conv/Conv2Dhu  ?A
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2@B8???@???H???XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2 ?8???
@???
H???
XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2 ?8?ك
@?ك
H?ك
Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?A
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@??
H??Xb.model/bbn_features_stemblock_stem1_conv/Conv2Dhdu  HB
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2??8???@???H???XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2!?8???@???H???XbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2??8???@???H???XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2??8???@???H???Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8Ύ?@Ύ?HΎ?XbMgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8ή?@ή?Hή?XbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8???@???H???XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8???@???H???XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8???@???H???XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8̓?@̓?H̓?Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???XbLgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb*model/bbn_features_transition4_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputh$u  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?+8?Ֆ@?ՖH?ՖXb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2??8???@???H???XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu ??B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2??8?Ӑ@?ӐH?ӐXb*model/bbn_features_transition2_conv/Conv2Dhu ??B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2??8???@???H???Xb*model/bbn_features_transition3_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2??8???@???H???XbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2??8???@???H???XbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2??8ʠ?@ʠ?Hʠ?XbHgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2??8???@???H???XbMgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 6, 7, 5, 4, 5, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)V??* 28???@???H???XbQgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, true, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2d8???@???H???bmodel/re_lu_113/Reluhu  ?B
c
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28???@??H??Xbmodel/ssd_cls1conv2/Conv2Dh?u  HB
c
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28???@??H??Xbmodel/ssd_cls3conv2/Conv2Dh?u  HB
c
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28?ً@??H??Xbmodel/ssd_cls4conv2/Conv2Dh?u  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8要@要H要Xb?model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2Dhu  ?B
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
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 2?8???@???H?јXbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???XbMgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8璸@璸H璸Xb?model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2Dhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2??8???@???H???XbMgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8斗@斗H斗Xb?model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8Ƞ?@??H??Xb?model/bbn_features_denseblock3_denselayer1_branch1b_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8鍏@??H??Xb?model/bbn_features_denseblock3_denselayer1_branch2c_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb)model/ssd_res_block2_branch1a_conv/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2,?8???@???H???XbLgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2,?8???@???H???Xb*model/bbn_features_transition4_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2Dhu  ?B
m
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@???H???bmodel/re_lu_113/Reluhu  HB
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch2b_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock4_denselayer1_branch1b_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block1_branch1b_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock4_denselayer1_branch2c_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block4_branch1b_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block3_branch1b_conv/Conv2Dhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 6, 7, 3, 3, 5, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)M?2* 2?
8???@??xH??yXb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block5_branch1b_conv/Conv2Dhu  ?B
I
redzone_checker*?2?@8???@??H??bmodel/re_lu_114/Reluhu  ?B
I
redzone_checker*?2?@8???@??H??bmodel/re_lu_115/Reluhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch2c_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch1b_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@??H??Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 6, 7, 3, 3, 5, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)@?2* 2?8???@??sH??tXb*model/bbn_features_transition2_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8ŕ?@ŕ?Hŕ?Xb?model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
0cudnn_convolve_sgemm_sm35_ldg_nn_128x8x128x16x16??A*2?8???@???H???Xb.model/bbn_features_stemblock_stem1_conv/Conv2Dhu  B
_
redzone_checker*?2?@8???@??H??Xb(model/ssd_res_block1_branch2_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock4_denselayer1_branch1a_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block1_branch1c_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2Dhu  ?B
a
redzone_checker*?2?@8???@??H??Xb*model/bbn_features_transition3_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block3_branch1c_conv/Conv2Dhu  ?B
_
redzone_checker*?2?@8Ù?@??H??Xb(model/ssd_res_block4_branch2_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block4_branch1c_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??H??XbIgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropFilterhu  ?B
_
redzone_checker*?2?@8???@??H??Xb(model/ssd_res_block3_branch2_conv/Conv2Dhu  ?B
a
redzone_checker*?2?@8???@??H??Xb*model/bbn_features_transition4_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??H??XbIgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 208Ĭ?@??lH??mXbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
]
redzone_checker*?2?@8???@??H??Xb&model/ssd_feature_extend1_conv1/Conv2Dhu  ?B
?
?void wgrad2d_grouped_direct_kernel<float, float, float, (cudnnTensorFormat_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnConvolutionStruct, cudnnFilterStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int)*?2?8???@???H???Xbbgradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8?ݮ@?ݮH?ݮXb?model/bbn_features_denseblock1_denselayer1_branch2a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???XbJgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer6_branch2c_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8?٪@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer6_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?
8Ą?@Ą?HĄ?Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8嶩@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch2c_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8㛩@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch2a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?
8???@???H???Xb?model/bbn_features_denseblock1_denselayer1_branch2b_conv/Conv2Dhu  ?B
Q
redzone_checker*?2?@8?ŧ@??H??Xbmodel/ssd_box1conv2/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?B
Q
redzone_checker*?2?@8???@??H??Xbmodel/ssd_cls1conv2/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block1_branch1a_conv/Conv2Dhu  ?B
e
redzone_checker*?2?@8???@??H??Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8?Ȧ@??H??Xb?model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8ű?@??H??Xb?model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8ᢥ@??H??Xb?model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8Ĩ?@Ĩ?HĨ?Xb&model/ssd_feature_extend1_conv1/Conv2Dhu  HB
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
`
redzone_checker*?2?@8?֣@??H??Xb)model/ssd_res_block4_branch1a_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8?£@??H??XbKgradient_tape/model/ssd_res_block5_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
v
redzone_checker*?2?@8Ī?@??H??Xb?model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2Dhu  ?B
a
redzone_checker*?2?@8???@??H??Xb*model/bbn_features_transition2_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2Dhu  ?B
Q
redzone_checker*?2?@8扣@??H??Xbmodel/ssd_box4conv2/Conv2Dhu  ?B
?
redzone_checker*?2?@8剣@??H??XbKgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
Q
redzone_checker*?2?@8䂣@??H??Xbmodel/ssd_box5conv2/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu  ?B

redzone_checker*?2?@8???@??H??XbHgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropInputhu  ?B
_
redzone_checker*?2?@8???@??H??Xb(model/ssd_res_block5_branch2_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8?آ@??H??XbLgradient_tape/model/ssd_res_block4_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B

redzone_checker*?2?@8?ע@??H??XbHgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8?Т@??H??XbLgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8?Ģ@??H??XbKgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2Dhu  ?B
Q
redzone_checker*?2?@8㱢@??H??Xbmodel/ssd_cls4conv2/Conv2Dhu  ?B
Q
redzone_checker*?2?@8???@??H??Xbmodel/ssd_cls3conv2/Conv2Dhu  ?B
Q
redzone_checker*?2?@8姢@??H??Xbmodel/ssd_box3conv2/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
`
redzone_checker*?2?@8䡢@??H??Xb)model/ssd_res_block5_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8垢@??H??Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2Dhu  ?B
Q
redzone_checker*?2?@8???@??H??Xbmodel/ssd_cls5conv2/Conv2Dhu  ?B
`
redzone_checker*?2?@8ŏ?@??H??Xb)model/ssd_res_block3_branch1a_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block2_branch1a_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8傢@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 8, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)u?R* 28???@??_H??`XbMgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb)model/ssd_res_block1_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8?ߖ@?ߖH?ߖXb?model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2?8???@???H???XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2?8ĵ?@???H???XbQgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8?ِ@?ِH?ِXb?model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)]?*28Ļ?@Ļ?HĻ?XbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void wgrad2d_grouped_direct_kernel<float, float, float, (cudnnTensorFormat_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnConvolutionStruct, cudnnFilterStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int)*?2?8?͎@?͎H?͎Xbbgradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8?υ@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8ç?@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
f
redzone_checker*?2?@8???@??H??Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28???@???H???Xb*model/bbn_features_transition4_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28???@???H???XbLgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropInputhu  HB
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block5_branch1c_conv/Conv2Dhu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8???@???H???XbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
a
redzone_checker*?2?@8???@??H??Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8???@??RH??SXbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*2?8???@£RH??RXbQgradient_tape/model/bbn_features_stemblock_stem1_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2? 8ĸ?@??{H??{XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8???@??wH??yXb*model/bbn_features_transition2_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8???@??wH??xXbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8???@??tHuXbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28???@???H???XbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28Ä?@??rH??sXbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8???@???H???XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
cvoid DSE::vector_fft<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)8 ??*2? 8???@??rH??rXb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28???@???H???Xb*model/bbn_features_transition3_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xbagradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xbagradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8Ĕ?@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8???@???H???Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?A
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2 ?8???@??oH??pXbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2? 8???@???H???XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8õ?@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8²?@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8Ø?@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8ŗ?@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8Ñ?@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
f
redzone_checker*?2?@8???@??H??Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8Ø?@??H??XbMgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
e
redzone_checker*?2?@8???@??H??Xb.model/bbn_features_stemblock_stem1_conv/Conv2Dhu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2B8???@???H???Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8???@???H???XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8ã?@??H??XbMgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8Ë?@Ë?HË?Xb(model/ssd_res_block1_branch2_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???XbJgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu  ?B
u
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xbmodel/ssd_cls5conv2/Conv2Dhu  HB
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block4_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 8, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)|?R* 2X8???@??mH??mXbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
t
redzone_checker*?2?@8???@??H??Xb=gradient_tape/model/ssd_cls4conv2/Conv2D/Conv2DBackpropFilterhu  ?B
t
redzone_checker*?2?@8???@??H??Xb=gradient_tape/model/ssd_cls5conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8???@??mH??mXbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block5_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block2_branch1c_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
s
redzone_checker*?2?@8???@??H??Xb<gradient_tape/model/ssd_box4conv2/Conv2D/Conv2DBackpropInputhu  ?B
s
redzone_checker*?2?@8Ż?@??H??Xb<gradient_tape/model/ssd_box2conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8Ļ?@??H??XbKgradient_tape/model/ssd_res_block3_branch1c_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbJgradient_tape/model/ssd_res_block4_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block5_branch1c_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbJgradient_tape/model/ssd_res_block5_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block5_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbJgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
s
redzone_checker*?2?@8???@??H??Xb<gradient_tape/model/ssd_box5conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
s
redzone_checker*?2?@8???@??H??Xb<gradient_tape/model/ssd_box3conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8Â?@??H??XbLgradient_tape/model/ssd_res_block3_branch1c_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block3_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
s
redzone_checker*?2?@8???@??H??Xb<gradient_tape/model/ssd_cls5conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbJgradient_tape/model/ssd_res_block3_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block2_branch1c_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block4_branch1c_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block5_branch1c_conv/Conv2D/Conv2DBackpropFilterhu  ?B
t
redzone_checker*?2?@8???@??H??Xb=gradient_tape/model/ssd_cls3conv2/Conv2D/Conv2DBackpropFilterhu  ?B
t
redzone_checker*?2?@8???@??H??Xb=gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbJgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropInputhu  ?B
s
redzone_checker*?2?@8²?@??H??Xb<gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block4_branch1c_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8Ʃ?@??H??XbLgradient_tape/model/ssd_res_block5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
s
redzone_checker*?2?@8???@??H??Xb<gradient_tape/model/ssd_cls3conv2/Conv2D/Conv2DBackpropInputhu  ?B
s
redzone_checker*?2?@8???@??H??Xb<gradient_tape/model/ssd_cls4conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
?void sgemm_largek_lds64<false, true, 5, 5, 4, 4, 4, 32>(float*, float const*, float const*, int, int, int, int, int, int, float const*, float const*, float, float, int, int, int*, int*)*?!*28???@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
?void implicit_convolve_sgemm<float, float, 512, 6, 8, 3, 3, 5, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)u?R* 2Q8???@??iH??kXb*model/bbn_features_transition3_conv/Conv2Dhu  HB
?
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28???@??H??Xb=gradient_tape/model/ssd_cls4conv2/Conv2D/Conv2DBackpropFilterh?u  HB
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2? 8???@???H???Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28???@??H??Xb=gradient_tape/model/ssd_cls3conv2/Conv2D/Conv2DBackpropFilterh?u  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8???@???H???Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2? 8???@???H???XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_c2r_32x32<float, true, false, 0u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)8 ??*?2 8???@??H??Xb.model/bbn_features_stemblock_stem1_conv/Conv2Dhdu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2Q8???@???H???Xb?model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb<gradient_tape/model/ssd_cls4conv2/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb<gradient_tape/model/ssd_cls3conv2/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb<gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb<gradient_tape/model/ssd_cls5conv2/Conv2D/Conv2DBackpropInputhu  HB
m
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@???H???bmodel/re_lu_114/Reluhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 28???@??CH??CXbMgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropFilterhu  ?B
m
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@???H???bmodel/re_lu_115/Reluhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8È?@È?HÈ?Xb?model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2Q8???@???H???Xb?model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8õ?@õ?Hõ?Xbagradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xbagradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void sgemm_largek_lds64<false, true, 5, 5, 4, 4, 4, 32>(float*, float const*, float const*, int, int, int, int, int, int, float const*, float const*, float, float, int, int, int*, int*)*?!*28???@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28???@???H??@XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8???@???H???Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 5, 6, 4, 3, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)X?A*28???@???H???XbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?W8Ã?@??>H???Xb.model/bbn_features_stemblock_stem1_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?W8ø?@??>H???Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2Q8???@???H???Xb?model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@???H???XbHgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropInputhu  HB
?
?void sgemm_largek_lds64<false, true, 5, 5, 4, 4, 4, 32>(float*, float const*, float const*, int, int, int, int, int, int, float const*, float const*, float, float, int, int, int*, int*)*?!*28?ݺ@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8ㇺ@ㇺHㇺXbHgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropInputhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2B8㧹@㧹H㧹Xbbgradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8???@???H???XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, true, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?28Ë?@Ë?HË?bmodel/re_lu_115/Reluhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28㽷@??<H??=Xb*model/bbn_features_transition1_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8Þ?@Þ?HÞ?Xb?model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2Q8???@???H???Xb?model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterh
u  ?B
t
redzone_checker*?2?@8???@??H??Xb=gradient_tape/model/ssd_box5conv2/Conv2D/Conv2DBackpropFilterh
u  ?B
?
redzone_checker*?2?@8???@??H??XbIgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropFilterh
u  ?B
t
redzone_checker*?2?@8???@??H??Xb=gradient_tape/model/ssd_box2conv2/Conv2D/Conv2DBackpropFilterh
u  ?B
t
redzone_checker*?2?@8???@??H??Xb=gradient_tape/model/ssd_box4conv2/Conv2D/Conv2DBackpropFilterh
u  ?B
t
redzone_checker*?2?@8º?@??H??Xb=gradient_tape/model/ssd_box3conv2/Conv2D/Conv2DBackpropFilterh
u  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterh
u  ?B
I
redzone_checker*?2?@8㉵@??H??bmodel/re_lu_113/Reluh
u  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8???@??XH??ZXb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??*2? 8???@??WH??ZXb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)P?2* 28???@???H???XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8???@???H???Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb?model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2Dhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 28B8???@???H???Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???XbKgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?a8⣤@??QH??RXb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)P?2* 28???@???H???XbMgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2B8???@???H???Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu  ?A
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8㷡@㷡H㷡XbKgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?ؠ@?ؠH?ؠXbagradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8?Š@??5H??5Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?
8ⷠ@ⷠHⷠXb?model/bbn_features_denseblock1_denselayer1_branch2a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xbmodel/ssd_cls3conv2/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xbagradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8?ǟ@?ǟH?ǟXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8?@?H?Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2,8?@??OH??OXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, true, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2$8???@???H???bmodel/re_lu_114/Reluhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?@?H?Xb?model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2Dhu  HB
?
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28?ޛ@??H??XbKgradient_tape/model/ssd_res_block3_branch2_conv/Conv2D/Conv2DBackpropFilterh?u  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8㮛@??3H??3Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28⃛@??H??XbKgradient_tape/model/ssd_res_block4_branch2_conv/Conv2D/Conv2DBackpropFilterh?u  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8???@??3H??3XbRgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8ý?@??1H??3XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2B8???@???H???Xbbgradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
&cudnn_dgrad_sm35_ldg_nt_64x16x128x8x32{?`* 2$8???@??KH??LXbHgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?@?H?Xb)model/ssd_res_block2_branch1a_conv/Conv2Dhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8???@???H???Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?A
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)P?2* 2?
8?@?H?XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8?ؓ@??H??XbQgradient_tape/model/bbn_features_stemblock_stem1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8¾?@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?X8???@???H???XbKgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbRgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropFilterhu  ?B

redzone_checker*?2?@8?@??H??XbHgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
.cudnn_convolve_sgemm_sm35_ldg_nn_32x16x64x8x16S?2*2?8???@???H???Xb*model/bbn_features_transition2_conv/Conv2Dhu ??B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8⌑@⌑H⌑Xb?model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2Dhu  HB
?
?void DSE::regular_fft_clip<1, 2, 256, 16, 16, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??*2? 8???@???H???XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2B8???@???H???Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu  ?A
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 20B8???@???H???Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8?Ê@?ÊH?ÊXb?model/bbn_features_denseblock2_denselayer1_branch1b_conv/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2,?8???@???H???Xb&model/ssd_feature_extend1_conv1/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8???@???H???XbJgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)P?2* 2?
8???@???H???Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2@B8?Ç@??CH??CXbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 6, 7, 3, 3, 5, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)M?2* 28?Յ@??BH??CXb*model/bbn_features_transition4_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2? 8¸?@¸?H¸?bmodel/re_lu_113/Reluhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb?model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)?*?2?8?@?H?bFgradient_tape/model/bbn_features_transition1_norm/FusedBatchNormGradV3hu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28???@??AH??AXbLgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void DSE::regular_fft_clip<1, 2, 256, 16, 16, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??*2? 8???@???H???XbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xbagradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputh	u  HB
?
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28??@?jH??XbLgradient_tape/model/ssd_res_block3_branch1c_conv/Conv2D/Conv2DBackpropFilterh?u  HB
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 4, 6, 3, 2, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)O? *2,8??@??H??XbLgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2? 8??~@??~H??~XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28??~@?jH??XbLgradient_tape/model/ssd_res_block4_branch1c_conv/Conv2D/Conv2DBackpropFilterh?u  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8è~@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch2b_conv/Conv2Dh	u  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??}@??>H???Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*2X8??}@??}H??}XbQgradient_tape/model/bbn_features_stemblock_stem1_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8??|@??|H??|Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
cvoid DSE::vector_fft<1, 2, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)5 ??*2? 8??|@??|H??|XbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
cvoid DSE::vector_fft<1, 2, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)5 ??*2? 8??|@??|H??|XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??|@??|H??|Xb=gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropFilterhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2? 8??|@??|H??|XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)P?2* 2?8??{@??{H??{XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2? 8??{@??{H??{Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??{@??{H??{Xb=gradient_tape/model/ssd_cls5conv2/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*2X8??z@??zH??zXbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)P?2* 2?
8??y@??<H??=XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)H?*2?W8ºy@ºyHºyXbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??x@??xH??xXb?model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??x@??xH??xXbKgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_r2c_32x32<float, false, 0u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?28??x@??H??Xb.model/bbn_features_stemblock_stem1_conv/Conv2Dhdu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2? 8¿x@¿xH¿xXb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2B8??x@??xH??xXbbgradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??v@??vH??vXbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??u@??uH??uXbKgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2B8??t@??tH??tXb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu  ?A
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)?*?2 8??s@??sH??sbJgradient_tape/model/bbn_features_stemblock_stem1_norm/FusedBatchNormGradV3hu  ?B
?
cvoid DSE::vector_fft<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)8 ??*2? 8??s@??sH??sXbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
:cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_64x8x256x8x32}?R* 2Q8??s@??sH??sXb*model/bbn_features_transition3_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2&8??r@??H??Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  HB
?
cvoid DSE::vector_fft<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)8 ??*2? 8??r@??rH??rXbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2(B8¡r@¡rH¡rXbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8r@rHrXb(model/ssd_res_block5_branch2_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??q@??%H??&Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??q@??qH??qXb(model/ssd_res_block3_branch2_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8¨q@¨qH¨qXb)model/ssd_res_block1_branch1a_conv/Conv2Dhu  HB
?
0cudnn_convolve_sgemm_sm35_ldg_nn_128x8x128x16x16??A*2?8??q@??qH??qXb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?@8q@qHqXbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??p@??pH??pXbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??p@??pH??pXbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)]?*28??p@??pH??pXbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??p@??8H??8Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??p@??pH??pXb(model/ssd_res_block4_branch2_conv/Conv2Dhu  HB
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2 ?8??p@??pH??pXb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??p@??pH??pXbJgradient_tape/model/ssd_res_block3_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??o@??oH??oXbJgradient_tape/model/ssd_res_block5_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2,8??o@??7H??7Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2??8??o@??oH??oXb*model/bbn_features_transition4_conv/Conv2Dhu  H?
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??n@??nH??nXbJgradient_tape/model/ssd_res_block4_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*  2?8??n@??nH??nXbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8??n@??nH??nXbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8n@nHnXbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8??m@??mH??mXb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8??m@??mH??mXb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8??m@??mH??mXbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?A8??m@??6H??6Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8??l@??lH??lXbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??l@??lH??lXb?model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2Dhu  HB
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8??l@??lH??lXb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8??k@??kH??kXbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?@8??k@??kH??kXbKgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cudnn_convolve_sgemm_sm35_ldg_nn_64x16x64x16x16O?B*2Q8??k@??kH??kXb*model/bbn_features_transition3_conv/Conv2Dhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??j@??jH??jXbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?%8??i@?? H??$XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??g@??gH??gXbbgradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??g@??gH??gXb)model/ssd_res_block1_branch1b_conv/Conv2Dhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2 ?8??g@??gH??gXb(model/ssd_res_block1_branch2_conv/Conv2Dhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??g@??3H??3Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??f@??fH??fXbbgradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?%8??f@?? H??#Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?A
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?<8??e@??eH??eXb=gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??e@??eH??eXbKgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??e@??eH??eXb)model/ssd_res_block4_branch1b_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??e@??eH??eXb)model/ssd_res_block5_branch1b_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??e@??eH??eXbKgradient_tape/model/ssd_res_block5_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?<8??e@??eH??eXb=gradient_tape/model/ssd_cls3conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??e@??eH??eXbKgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*28??e@??!H??"bmodel/re_lu_113/Reluhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??e@??eH??eXb<gradient_tape/model/ssd_cls3conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??e@??eH??eXb<gradient_tape/model/ssd_cls4conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??e@??eH??eXb<gradient_tape/model/ssd_cls5conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??e@??eH??eXb<gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??e@??eH??eXb)model/ssd_res_block3_branch1b_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??d@??dH??dXbKgradient_tape/model/ssd_res_block4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?<8??d@??dH??dXb=gradient_tape/model/ssd_cls5conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?<8??d@??dH??dXb=gradient_tape/model/ssd_cls4conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??d@??dH??dXbJgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??c@??1H??2Xb?model/bbn_features_denseblock1_denselayer1_branch2b_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28b@bHbXbJgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??a@??aH??aXbmodel/ssd_cls1conv2/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??a@??aH??aXbagradient_tape/model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2B8??`@??`H??`Xb?model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2Dhu  ?A
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??_@??_H??_Xbmodel/ssd_cls4conv2/Conv2Dhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??_@??_H??_Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??_@??_H??_Xbmodel/ssd_cls5conv2/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??_@??_H??_Xbmodel/ssd_cls3conv2/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*208??_@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)P?2* 28??_@??_H??_XbHgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropInputhu  ?B
}
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??^@??^H??^Xb&model/ssd_feature_extend1_conv1/Conv2Dhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2 B8??]@??]H??]Xbagradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??]@??]H??]Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?%8??]@??H?? XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?%8??\@??H??!Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?A
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??\@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch2c_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8??\@??-H??.Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28[@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1b_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8??Z@??H??Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?A
?
?void DSE::regular_fft_pad<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??*2? 8??Y@??YH??YXbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??Y@??YH??YXbbgradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??X@??XH??XXbbgradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2
?8??X@??XH??XXbagradient_tape/model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2
?8??X@??XH??XXbagradient_tape/model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8??V@??*H??+XbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
?void DSE::regular_fft_pad<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??*2? 8®V@®VH®VXbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2?8??U@??*H??+Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??T@??TH??TXbRgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_r2c_32x32<float, true, 0u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)? ??*?2 8??T@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??S@??SH??SXbLgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?8??R@??RH??RXbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?a8??R@??RH??RXbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?a8??R@??RH??RXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?a8??Q@??QH??QXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)\?*28??Q@??QH??QXbMgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?8??Q@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?a8??Q@??QH??QXbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2&8??Q@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8??P@??(H??(Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*208??P@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*208??P@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??P@??PH??PXbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2	?8??P@??PH??PXbagradient_tape/model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2,8??O@??OH??OXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??O@??OH??OXb?model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2Dhu  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8??M@??H??Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?A
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2	?8??M@??MH??MXbagradient_tape/model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_c2r_32x32<float, false, false, 0u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)8 ??*?28??L@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)P?2* 2?
8??L@??LH??LXbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??K@??KH??KXb?model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??K@??KH??KXbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??K@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch1b_conv/Conv2Dhu  HB
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2 ?8??K@??KH??KXbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?,8??J@??JH??JXbLgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2)8??I@??H??Xb)model/ssd_res_block1_branch1b_conv/Conv2Dhu  HB
?
?void DSE::regular_fft_clip<1, 2, 256, 16, 16, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??*2?8??I@??IH??IXb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?B
q
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??I@??IH??IXbmodel/ssd_cls1conv2/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??I@??H??Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??I@??IH??IXbbgradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2?88??I@??IH??IXbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8??H@??HH??HXb?model/bbn_features_denseblock2_denselayer1_branch2c_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??H@??HH??HXbbgradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2? 8??H@??HH??HXb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu ??B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??H@??HH??HXbJgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??G@??GH??GXb?model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2Dhu  HB

/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??G@??GH??GXb(model/ssd_res_block1_branch2_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*28??G@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 8, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)|?R* 28??F@??FH??FXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?8??E@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 28??E@??H??XbKgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??E@??EH??EXbKgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2,?8??E@??EH??EXb)model/ssd_res_block2_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??D@??DH??DXbagradient_tape/model/bbn_features_denseblock4_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 512, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float) ?*?2?8??D@??DH??Db4model/bbn_features_transition1_norm/FusedBatchNormV3hu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??D@??DH??DXb?model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2Dhu  HB
?
:cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_64x8x256x8x32}?R* 28??C@??CH??CXb*model/bbn_features_transition4_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*2$8??C@??CH??CXbIgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*208??C@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
hvoid fft1d_r2c_32<float, float, float2, true, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?y8??C@??CH??CXb*model/bbn_features_transition4_conv/Conv2Dhu  ?A
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??B@??BH??BXb<gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropInputhu  HB
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??B@??H??XbHgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropInputh$u  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??B@??BH??BXbHgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??B@??BH??Bbmodel/re_lu_114/Reluhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??B@??BH??Bbmodel/re_lu_115/Reluhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??B@??BH??BXbHgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?y8??B@??BH??BXbLgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??B@??BH??BXbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8??B@?? H??!Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 512, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float) ?*?2 8??A@??AH??Ab8model/bbn_features_stemblock_stem1_norm/FusedBatchNormV3hu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??A@??AH??AXbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??@@??H??XbHgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropInputh$u  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*28??@@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@@??@H??@Xb?model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2Dhu  HB
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2 ?8??@@??@H??@Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu ??B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?%8??@@??H??!XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?%8??@@??H??"Xb*model/bbn_features_transition2_conv/Conv2Dhu  ?A
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??@@??@H??@Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??@@??@H??@XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??@@??H?? Xb?model/bbn_features_denseblock1_denselayer1_branch2a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2E8??@@??@H??@Xbmodel/ssd_cls4conv2/Conv2Dhu  ?B
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2?08???@???H???Xbbgradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8???@???H???Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 5, 6, 4, 3, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)X?A*2+8???@??H??XbHgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropInputhu ??B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??>@??>H??>Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2? 8??>@??>H??>Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu ??B
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)?*?28??>@??>H??>bKgradient_tape/model/bbn_features_stemblock_stem2a_norm/FusedBatchNormGradV3hu  ?B
?
cvoid DSE::vector_fft<1, 2, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)5 ??*2?8??>@??>H??>Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*28??>@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8??>@??>H??>Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??=@??=H??=Xb?model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)M?2* 2$8??=@??H??XbIgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 4, 6, 3, 2, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)O? *2,8??=@??=H??=XbJgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??=@??=H??=Xb)model/ssd_res_block1_branch1c_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??=@??=H??=XbLgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??<@??<H??<Xb)model/ssd_res_block3_branch1c_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??<@??<H??<XbKgradient_tape/model/ssd_res_block5_branch2_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8 <@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
j
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??<@??<H??<bmodel/re_lu_115/Reluhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??;@??;H??;Xbagradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
j
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??;@??;H??;bmodel/re_lu_114/Reluhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??;@??;H??;Xbbgradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??;@??;H??;XbKgradient_tape/model/ssd_res_block5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??;@??;H??;Xb)model/ssd_res_block4_branch1c_conv/Conv2Dhu  HB
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2??8??;@??;H??;Xb*model/bbn_features_transition3_conv/Conv2Dhu  H?
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??;@??;H??;XbKgradient_tape/model/ssd_res_block3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?28??;@??;H??;b"gradient_tape/model/re_lu/ReluGradhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??;@??;H??;Xb?model/bbn_features_denseblock4_denselayer1_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??;@??;H??;Xb)model/ssd_res_block5_branch1c_conv/Conv2Dhu  HB
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2?8??;@??H??Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??;@??;H??;XbKgradient_tape/model/ssd_res_block4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??;@??;H??;Xb?model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2Dhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?28??:@??:H??:b%gradient_tape/model/re_lu_19/ReluGradhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??:@??:H??:Xbbgradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2,8??:@??:H??:XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??:@??:H??:Xb)model/ssd_res_block4_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??:@??:H??:XbHgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropInputhu  HB
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??:@??:H??:Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??:@??:H??:XbKgradient_tape/model/ssd_res_block4_branch1c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??:@??:H??:Xb)model/ssd_res_block3_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??:@??:H??:XbKgradient_tape/model/ssd_res_block3_branch1c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2,8??:@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??9@??9H??9Xb)model/ssd_res_block5_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??9@??9H??9XbHgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??9@??9H??9XbKgradient_tape/model/ssd_res_block2_branch1c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??9@??9H??9Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
.cudnn_convolve_sgemm_sm35_ldg_nn_32x16x64x8x16S?2*2)8??9@??9H??9Xb*model/bbn_features_transition4_conv/Conv2Dhu ??B
?
cvoid DSE::vector_fft<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)8 ??*2?8??9@??9H??9XbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??9@??9H??9Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8??9@??H??Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu  HB
?
?void fft2d_c2r_32x32<float, true, false, 0u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)8 ??*?2 8??9@??H??Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??9@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
cvoid DSE::vector_fft<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)8 ??*2?8??9@??9H??9XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::pooling_bw_kernel_max<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor) ? *?2,8??9@??9H??9b5gradient_tape/model/max_pooling2d/MaxPool/MaxPoolGradhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??9@??9H??9XbKgradient_tape/model/ssd_res_block5_branch1c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??9@??9H??9Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??8@??H??Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??8@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??8@??H??Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8??8@??8H??8Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??8@??8H??8Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??8@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch2a_conv/Conv2Dhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??7@??7H??7Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2? 8??7@??H??Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu  ?A
?	
?	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??7@??7H??7bAdam/gradients/AddN_30hu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??7@??7H??7Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??7@??7H??7XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*2[8??7@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??7@??7H??7Xbagradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2,8??7@??7H??7Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??7@??7H??7Xbbgradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?A8??7@??7H??7Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)H?*2?8??7@??7H??7Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?A8??7@??7H??7XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?A8??7@??7H??7Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??7@??7H??7Xb?model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*2	8??7@??7H??7Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??6@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch2c_conv/Conv2Dhu  HB
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?8??6@??6H??6Xbbgradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*2[8??6@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch2b_conv/Conv2Dh$u  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?A8??6@??6H??6Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_r2c_32x32<float, false, 0u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?28??6@??H??Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2? 8??6@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch2a_conv/Conv2Dhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8??6@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?A8??6@??6H??6XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?A8??6@??6H??6Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??6@??6H??6bAdam/gradients/AddN_33hu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*2
8??6@??6H??6Xbbgradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2? 8??6@??6H??6XbLgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?A8??6@??6H??6Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*28??6@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??6@??6H??6Xbbgradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2? 8??6@??6H??6XbKgradient_tape/model/ssd_res_block3_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??5@??5H??5Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8??5@??H??Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2? 8??5@??5H??5XbKgradient_tape/model/ssd_res_block4_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??5@??5H??5Xbbgradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2? 8??5@??5H??5XbKgradient_tape/model/ssd_res_block5_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*208??5@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??5@??5H??5XbKgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8??5@??H??Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?A
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??5@??5H??5Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch2a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8??4@??H??Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?A
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)?*?2?8??4@??4H??4bFgradient_tape/model/bbn_features_transition2_norm/FusedBatchNormGradV3hu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??4@??4H??4Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch2b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2 ?8??4@??4H??4Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2? 8??4@??4H??4XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu ??B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2? 8??4@??4H??4Xb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu ??B
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2? 8??4@??4H??4Xb*model/bbn_features_transition2_conv/Conv2Dhu ??B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2 ?8??4@??4H??4Xb)model/ssd_res_block1_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??4@??4H??4Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch2b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??4@??4H??4Xbagradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 28??3@??H??XbLgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??3@??3H??3Xbbgradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??3@??3H??3Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??3@??3H??3Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
&cudnn_dgrad_sm35_ldg_nt_64x16x128x8x32{?`* 28??3@??H??XbKgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??3@??3H??3Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch2b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??3@??3H??3Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch2b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??3@??3H??3Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch2b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??3@??3H??3Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??3@??3H??3Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch2b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??3@??3H??3Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??3@??3H??3Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??3@??3H??3Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2?(8??3@??3H??3Xbbgradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??3@??3H??3Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch2c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??3@??3H??3Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch2c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??3@??3H??3Xb?model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2Dhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8??3@??H??Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??3@??3H??3Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??3@??3H??3Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??3@??3H??3Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch2c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??2@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?%8??2@??H??XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
[void cudnn::ops::scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*?2??8??2@??H??XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??2@??2H??2Xb(model/ssd_res_block4_branch2_conv/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??2@??2H??2Xb(model/ssd_res_block5_branch2_conv/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??2@??2H??2XbJgradient_tape/model/ssd_res_block3_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??2@??2H??2XbJgradient_tape/model/ssd_res_block5_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??2@??2H??2Xb(model/ssd_res_block3_branch2_conv/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??2@??2H??2XbKgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??2@??2H??2XbJgradient_tape/model/ssd_res_block4_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?8??2@??2H??2Xbbgradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?%8??2@??H??Xb*model/bbn_features_transition2_conv/Conv2Dhu  ?A
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8??2@??2H??2Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  HB
?
[void cudnn::ops::scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*?2??8??2@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8??2@??H??Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8??2@??H??Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu  ?A
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28??2@??H??XbJgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?8??2@??2H??2Xbbgradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??2@??2H??2Xbbgradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8??2@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??1@??1H??1Xb?model/bbn_features_denseblock1_denselayer1_branch1b_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??1@??1H??1Xb?model/bbn_features_denseblock1_denselayer1_branch2c_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*28??1@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 28??1@??H??XbKgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??1@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch1b_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??0@??0H??0Xbbgradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2)8??0@??H??Xb&model/ssd_feature_extend1_conv1/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??0@??0H??0Xb?model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??0@??0H??0Xb?model/bbn_features_denseblock1_denselayer2_branch1b_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??0@??0H??0Xb?model/bbn_features_denseblock1_denselayer3_branch2c_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??0@??0H??0Xb?model/bbn_features_denseblock1_denselayer3_branch2b_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??0@??0H??0Xb?model/bbn_features_denseblock1_denselayer2_branch2b_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??0@??0H??0Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??0@??0H??0Xb?model/bbn_features_denseblock1_denselayer2_branch2c_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 28??0@??H??Xb=gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropFilterhu  ?B
q
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??0@??0H??0Xbmodel/ssd_cls3conv2/Conv2Dhu  HB
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??/@??H??XbLgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??/@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch1b_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??/@??/H??/Xb<gradient_tape/model/ssd_cls3conv2/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*208??/@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?8??/@??H??XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??/@??/H??/Xb)model/ssd_res_block2_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::pooling_bw_kernel_avg<float, float, cudnn::averpooling_func<float, true>, 2, false>(cudnnTensorStruct, float const*, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)( ? *?2 8??/@??/H??/b1gradient_tape/model/average_pooling2d/AvgPoolGradhu  ?B
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2?8??/@??/H??/Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)H?*2?8??/@??/H??/Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??/@??/H??/Xbagradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
q
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28??/@??H??Xb)model/ssd_res_block1_branch1b_conv/Conv2Dh$u  ?B
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?8??/@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void tensorflow::functor::PadInputCustomKernelNCHW<float, 4>(int, float const*, tensorflow::functor::Dimension<4>, float*, tensorflow::functor::Dimension<4>, tensorflow::functor::Dimension<(4)-(2)>, float)*?28??/@??/H??/Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??.@??.H??.Xb?model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2Dhu  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??.@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch2c_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8??.@??.H??.Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch2a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 28??.@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?	8??-@??H??	XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 4, 6, 3, 2, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)O? *2 8??-@??-H??-XbJgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2? 8??-@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void tensorflow::functor::PadInputCustomKernelNCHW<float, 4>(int, float const*, tensorflow::functor::Dimension<4>, float*, tensorflow::functor::Dimension<4>, tensorflow::functor::Dimension<(4)-(2)>, float)*?28??-@??-H??-XbRgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??*2?8??-@??-H??-XbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void tensorflow::functor::PadInputCustomKernelNCHW<float, 4>(int, float const*, tensorflow::functor::Dimension<4>, float*, tensorflow::functor::Dimension<4>, tensorflow::functor::Dimension<(4)-(2)>, float)*?28??-@??-H??-XbQgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputhu  ?B
Z
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28??-@??H??bmodel/re_lu_115/Reluh$u  ?B
?
?void DSE::regular_fft_pad<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??*2?8??-@??-H??-XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
q
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28??-@??H??Xb)model/ssd_res_block3_branch1b_conv/Conv2Dh$u  ?B
Z
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28??-@??H??bmodel/re_lu_114/Reluh$u  ?B
?
?void fft2d_r2c_32x32<float, false, 0u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??,@??H??XbMgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??,@??,H??,Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*28??,@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
q
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28??,@??H??Xb)model/ssd_res_block4_branch1b_conv/Conv2Dh$u  ?B
q
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28??+@??H??Xb)model/ssd_res_block5_branch1b_conv/Conv2Dh$u  ?B
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?8??+@??+H??+Xbbgradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 28??+@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??+@??H??XbKgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??+@??H??XbKgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*28??+@??H??bmodel/re_lu_114/Reluhu  HB
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8??+@??+H??+XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu ??B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8??*@??*H??*Xb*model/bbn_features_transition2_conv/Conv2Dhu ??B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??*@??H??XbKgradient_tape/model/ssd_res_block5_branch1b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 4, 6, 3, 2, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)O? *28??*@??*H??*Xb<gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?
8??*@??*H??*Xb?model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8??*@??H??Xb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?
8??*@??*H??*Xbagradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??*@??H??XbKgradient_tape/model/ssd_res_block4_branch1b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?	8??*@??H??Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?A
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2 ?8??*@??*H??*Xbagradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2? 8??)@??)H??)Xbbgradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?228??)@??)H??)Xb)model/ssd_res_block3_branch1a_conv/Conv2Dhu  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8??)@??H??Xb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu  ?A
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2? 8??)@??)H??)Xb?model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2Dhu ??B
?
hvoid fft1d_r2c_32<float, float, float2, true, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?@8??)@??)H??)Xb*model/bbn_features_transition3_conv/Conv2Dhu  ?A
?
?void wgrad2d_grouped_direct_kernel<float, float, float, (cudnnTensorFormat_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnConvolutionStruct, cudnnFilterStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int)*?2?	8??)@??)H??)Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2	8??)@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*28??)@??H??Xb)model/ssd_res_block3_branch1b_conv/Conv2Dhu  HB
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2?8??(@??(H??(Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu ??B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2
?8??(@??(H??(Xbbgradient_tape/model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?28??(@??(H??(Xbmodel/ssd_box1conv2/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2
?8??(@??(H??(Xbbgradient_tape/model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 28??(@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2??8??(@??(H??(Xb&model/ssd_feature_extend1_conv1/Conv2Dhu  H?
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?
8??(@??(H??(Xbbgradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2?8??(@??(H??(Xb?model/bbn_features_denseblock1_denselayer1_branch2b_conv/Conv2Dhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8??'@??'H??'Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch2a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)H?*2?8??'@??'H??'Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, false, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)P?*28??'@??H??Xb)model/ssd_res_block5_branch1b_conv/Conv2Dhu  HB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*?28??'@??H??bmodel/concatenate_3/concathu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8??'@??H??Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu  HB
?
&cudnn_dgrad_sm35_ldg_nt_64x16x128x8x32{?`* 28??'@??'H??'XbHgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropInputhu  HB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*?28??'@??'H??'bmodel/re_lu/Reluhu  ?B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8??'@??'H??'Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*?28??&@??&H??&bmodel/re_lu_19/Reluhu  ?B
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?8??&@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??&@??&H??&Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch2a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??&@??&H??&XbKgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28??&@??H??XbJgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?	8??&@??&H??&Xb?model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??&@??&H??&Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?	8??&@??&H??&Xbagradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void wgrad2d_grouped_direct_kernel<float, float, float, (cudnnTensorFormat_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnConvolutionStruct, cudnnFilterStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int)*?2?8??&@??	H??	Xb=gradient_tape/model/ssd_box3conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
9cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_32x16x32x8x8D?$*2?8??&@??H??Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8??&@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*28??%@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28??%@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?8??%@??%H??%Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
[void cudnn::ops::scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*?2և8??%@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8??%@??%H??%Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu ??B
?
?void cudnn::cnn::im2col4d_kernel<float, long>(cudnn::cnn::im2col4d_params, cudnnConvolutionStruct, cudnnTensor4dStruct, float const*, float*) *?2?8??%@??%H??%Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*28??%@??H??bmodel/re_lu_115/Reluhu  HB
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 5, 6, 4, 3, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)X?A*2+8??%@??%H??%XbKgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropInputhu ??B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??%@??H??XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 28??%@??H??XbLgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*28??$@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?	
?	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??$@??$H??$bAdam/gradients/AddN_31hu  ?B
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?	8??$@??$H??$Xbbgradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2	?8??$@??$H??$Xbbgradient_tape/model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?@8??$@??$H??$XbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8??$@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2)8??$@??H??Xbmodel/ssd_cls1conv2/Conv2Dhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?8??$@??$H??$Xbbgradient_tape/model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2Q8??$@??$H??$Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??#@??#H??#Xb)model/ssd_res_block1_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28??#@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??#@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch2a_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*28??#@??H??Xb)model/ssd_res_block4_branch1b_conv/Conv2Dhu  HB
?
?void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 512, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float) ?*?28??#@??#H??#b9model/bbn_features_stemblock_stem2a_norm/FusedBatchNormV3hu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2	?8??#@??#H??#Xbbgradient_tape/model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??#@??#H??#Xbagradient_tape/model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 5, 6, 4, 3, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)X?A*2+8??#@??H??XbKgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 8, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)|?R* 28??#@??#H??#Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)\?*28??#@??#H??#XbKgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
9cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_32x16x32x8x8D?$*2?8??"@??H??Xb?model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8??"@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2Dhu  HB
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2?8??"@??H??Xb?model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28??"@??H??Xb<gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??"@??"H??"Xb)model/ssd_res_block1_branch1b_conv/Conv2Dhu  ?B
?
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28??"@?nH??Xb?model/bbn_features_denseblock2_denselayer1_branch1b_conv/Conv2Dh$u  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?8??"@??"H??"Xbbgradient_tape/model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8??"@??"H??"Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 5, 6, 4, 3, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)X?A*2+8??"@??H??XbKgradient_tape/model/ssd_res_block4_branch1b_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??"@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void tensorflow::functor::PadInputCustomKernelNCHW<float, 4>(int, float const*, tensorflow::functor::Dimension<4>, float*, tensorflow::functor::Dimension<4>, tensorflow::functor::Dimension<(4)-(2)>, float)*?28??"@??"H??"Xb.model/bbn_features_stemblock_stem1_conv/Conv2Dhu  ?B
?
?void tensorflow::functor::PadInputCustomKernelNCHW<float, 4>(int, float const*, tensorflow::functor::Dimension<4>, float*, tensorflow::functor::Dimension<4>, tensorflow::functor::Dimension<(4)-(2)>, float)*?28??"@??"H??"XbQgradient_tape/model/bbn_features_stemblock_stem1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)P?2* 2?
8??"@??"H??"Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??"@??"H??"Xb?model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2Dhu  ?B
?
9cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_32x16x32x8x8D?$*2?8??"@??"H??"Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??"@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??!@??!H??!Xbagradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??!@??!H??!Xbagradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??!@??!H??!Xb)model/ssd_res_block4_branch1b_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??!@??!H??!Xbagradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??!@??!H??!XbKgradient_tape/model/ssd_res_block4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??!@??!H??!Xbagradient_tape/model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??!@??!H??!XbKgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??!@??!H??!Xb)model/ssd_res_block5_branch1b_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??!@??!H??!Xb)model/ssd_res_block3_branch1b_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??!@??!H??!XbKgradient_tape/model/ssd_res_block5_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cub::DeviceSegmentedRadixSortKernel<cub::DeviceRadixSortPolicy<float, int, int>::Policy700, true, true, float, int, cub::TransformInputIterator<int, tensorflow::impl::SegmentOffsetCreator, cub::CountingInputIterator<int, long>, long>, int>(float const*, float*, int const*, int*, cub::TransformInputIterator<int, tensorflow::impl::SegmentOffsetCreator, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::impl::SegmentOffsetCreator, cub::CountingInputIterator<int, long>, long>, int, int, int)`?D*?28??!@??H??b2compute_loss/cond/else/_1/compute_loss/cond/TopKV2hu  zB
?
9cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_32x16x32x8x8D?$*2?8??!@??!H??!Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8??!@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2Dhu  ?A
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??!@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch2b_conv/Conv2Dhu  ?B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8??!@??!H??!Xbbgradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2Q8??!@??!H??!Xbagradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??!@??!H??!XbKgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2*@8??!@??!H??!Xb?model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2Dhu  ?B
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2?8??!@??!H??!Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8??!@??!H??!Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch2a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??!@??!H??!Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8??!@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch2a_conv/Conv2Dhu  HB
?
9cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_32x16x32x8x8D?$*2?8??!@??H??Xb?model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2Dhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??!@??!H??!XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28??!@?qH??Xb?model/bbn_features_denseblock3_denselayer1_branch1b_conv/Conv2Dh$u  ?B
?
9cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_32x16x32x8x8D?$*2?8??!@??H??Xb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu  HB
?
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28??!@?qH??Xb?model/bbn_features_denseblock4_denselayer1_branch1b_conv/Conv2Dh$u  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28?? @?nH??XbLgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?8?? @??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8?? @?? H?? XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2?8?? @??H??Xb?model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*208?? @?? H?? Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch2a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?8?? @?? H?? XbHgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?8?? @?? H?? Xbbgradient_tape/model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?? @?? H?? Xb?model/bbn_features_denseblock3_denselayer1_branch1b_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28?? @?? H?? Xb?model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28?? @??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8?? @?? H?? Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu ??B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28?? @?? H?? Xbagradient_tape/model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28?? @?? H?? b'gradient_tape/model/concatenate_3/Slicehu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8?? @?? H?? Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2(@8?? @?? H?? Xb?model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8?? @?? H?? Xbagradient_tape/model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?$8??@??H??Xb*model/bbn_features_transition3_conv/Conv2Dhu  ?A
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbKgradient_tape/model/ssd_res_block5_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)H?*2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?8??@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbKgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2?8??@??H??Xb?model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbKgradient_tape/model/ssd_res_block4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbIgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropFilterhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?$8??@??H??XbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8??@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch2b_conv/Conv2Dhu  ?A
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbLgradient_tape/model/ssd_res_block3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?	
?	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??@??H??bAdam/gradients/AddN_26hu  ?B
?
9cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_32x16x32x8x8D?$*2)8??@??H??Xb)model/ssd_res_block2_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbLgradient_tape/model/ssd_res_block2_branch1c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xb)model/ssd_res_block1_branch1b_conv/Conv2Dhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2Q8??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xb)model/ssd_res_block4_branch1b_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbKgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbIgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbLgradient_tape/model/ssd_res_block5_branch1c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbLgradient_tape/model/ssd_res_block5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2Q8??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 1024, 2, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?`*?2?8??@??H??bwgradient_tape/model/bbn_features_stemblock_stem1_conv/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbLgradient_tape/model/ssd_res_block4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xb)model/ssd_res_block5_branch1b_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xb)model/ssd_res_block3_branch1b_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)\?*28??@??H??Xb=gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??@??H??Xb?model/bbn_features_denseblock3_denselayer1_branch1b_conv/Conv2Dhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2&@8??@??H??Xb?model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 5, 6, 4, 3, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)X?A*2+8??@??H??XbHgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropInputhu ??B
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
:cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_64x8x256x8x32}?R* 28??@??H??Xb(model/ssd_res_block1_branch2_conv/Conv2Dhu  HB
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2??8??@??H??Xb(model/ssd_res_block1_branch2_conv/Conv2Dhu  H?
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2)8??@??H??Xb&model/ssd_feature_extend1_conv1/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??@??H??Xb?model/bbn_features_denseblock4_denselayer1_branch1b_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??@??H??Xb?model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)\?*28??@??H??XbKgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2	8??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
r
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xbmodel/ssd_box1conv2/Conv2Dhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?8??@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
r
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xbmodel/ssd_box4conv2/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8??@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8??@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??@??H??Xb?model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2Dhu  HB
r
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xbmodel/ssd_box3conv2/Conv2Dhu  HB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*?28??@??H??bmodel/concatenate_2/concathu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28??@??H??XbKgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch2b_conv/Conv2Dhu  ?B
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2?8??@??H??Xb?model/bbn_features_denseblock1_denselayer3_branch2a_conv/Conv2Dhu  HB
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch2a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?28??@??H??b$gradient_tape/model/re_lu_1/ReluGradhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 5, 6, 4, 3, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)X?A*28??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1b_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??@??H??Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2$@8??@??H??Xb?model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2,8??@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?!8??@??H??Xb?model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2Dhu  ?A
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??@??H??Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
r
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xbmodel/ssd_box5conv2/Conv2Dhu  HB
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)?*?2 8??@??H??bKgradient_tape/model/bbn_features_stemblock_stem2b_norm/FusedBatchNormGradV3hu  ?B
?
hvoid fft1d_r2c_32<float, float, float2, true, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?,8??@??H??Xb&model/ssd_feature_extend1_conv1/Conv2Dhu  ?A
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2Q8??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2??8??@??H??Xbmodel/ssd_cls1conv2/Conv2Dhu  H?
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??@?`H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?28??@??H??b%gradient_tape/model/re_lu_40/ReluGradhu  ?B
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)?*?2 8??@??H??bJgradient_tape/model/bbn_features_stemblock_stem3_norm/FusedBatchNormGradV3hu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??@??H??Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?8??@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??@??H??XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 28??@??	H??	XbLgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)( ? *?2,8??@??H??bmodel/max_pooling2d/MaxPoolhu  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??@?_H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2"@8??@??H??Xb?model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
?void pooling_fw_4d_kernel<float, float, cudnn::averpooling_func<float, true>, (cudnnPoolingMode_t)2, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)0 ?*?2 8??@??H??bmodel/average_pooling2d/AvgPoolhu?O?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2??8??@??H??Xbmodel/ssd_cls5conv2/Conv2Dhu  H?
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2??8??@??H??Xbmodel/ssd_cls3conv2/Conv2Dhu  H?
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2? 8??@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2??8??@??H??Xbmodel/ssd_cls4conv2/Conv2Dhu  H?
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2? 8??@??H??Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*208??@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch2b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2? 8??@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B