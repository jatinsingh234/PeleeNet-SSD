
?
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28???=@??H??XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputh?u  HB
s
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28???<@??H??Xb*model/bbn_features_transition2_conv/Conv2Dh?u  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?R8??? @??? H??? Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8???@???H???XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8???@???H???Xb*model/bbn_features_transition1_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???XbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2??8???@???H???XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?Ŭ@?ŬH?ŬXb*model/bbn_features_transition3_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@???H???XbHgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropInputhu  HB
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?B8???	@???	H???	Xb*model/bbn_features_transition2_conv/Conv2Dhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?B8???	@???	H???	XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2??8???	@???	H???	Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@??
H??Xb.model/bbn_features_stemblock_stem1_conv/Conv2Dhdu  HB
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2??8???@???H???XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2??8???@???H???XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2??8߸?@߸?H߸?Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2!?8?ν@?νH?νXbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2??8?ɻ@?ɻH?ɻXbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???XbMgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2??8???@???H???Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???XbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8???@???H???XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8???@???H???Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8???@???H???XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8̱?@̱?H̱?XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2??8???@???H???XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8?ɛ@?ɛH?ɛXb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???XbLgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?̒@?̒H?̒Xb*model/bbn_features_transition4_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputh$u  HB
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2??8剠@剠H剠XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu ??B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2??8???@???H???Xb*model/bbn_features_transition2_conv/Conv2Dhu ??B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2??8ǉ?@ǉ?Hǉ?XbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8?Ќ@?ЌH?ЌXb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  HB
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2??8???@???H???XbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?)8???@???H???Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2??8???@???H???Xb*model/bbn_features_transition3_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2??8???@???H???XbHgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 6, 7, 5, 4, 5, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)V??* 28???@???H???XbQgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2??8???@???H???XbMgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, true, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2^8???@???H???bmodel/re_lu_113/Reluhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
c
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28???@??H??Xbmodel/ssd_cls1conv2/Conv2Dh?u  HB
c
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28?@??H߳Xbmodel/ssd_cls3conv2/Conv2Dh?u  HB
c
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28???@??H??Xbmodel/ssd_cls4conv2/Conv2Dh?u  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8???@???H???Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8???@???H???Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8߀?@߀?H߀?XbMgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xbmodel/ssd_cls1conv2/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8???@???H???Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 2?8???@???H???XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8弱@弱H弱Xb?model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2Dhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2??8???@???H???XbMgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock3_denselayer1_branch1b_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8?ٌ@??HޏXb?model/bbn_features_denseblock3_denselayer1_branch2c_conv/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2,?8̼?@̼?H̼?XbLgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2,?8???@???H???Xb*model/bbn_features_transition4_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch1b_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2Dhu  ?B
m
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@???H???bmodel/re_lu_113/Reluhu  HB
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch2c_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@ޑH??Xb?model/bbn_features_denseblock2_denselayer1_branch1b_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8Ϊ?@??H??Xb)model/ssd_res_block1_branch1b_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu  HB
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock4_denselayer1_branch2c_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block3_branch1b_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block5_branch1b_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block4_branch1b_conv/Conv2Dhu  ?B
I
redzone_checker*?2?@8???@ݗH??bmodel/re_lu_115/Reluhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock4_denselayer1_branch1b_conv/Conv2Dhu  ?B
I
redzone_checker*?2?@8???@??H??bmodel/re_lu_114/Reluhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@??H??Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 128, 6, 7, 3, 3, 5, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)M?2* 2?
8???@??qH??sXb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb)model/ssd_res_block2_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8???@???H???XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  HB
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block3_branch1c_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
?void implicit_convolve_sgemm<float, float, 1024, 6, 7, 3, 3, 5, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)@?2* 2?8???@??iH??pXb*model/bbn_features_transition2_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block1_branch1c_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block4_branch1c_conv/Conv2Dhu  ?B
_
redzone_checker*?2?@8???@??HݪXb(model/ssd_res_block4_branch2_conv/Conv2Dhu  ?B
a
redzone_checker*?2?@8???@??H??Xb*model/bbn_features_transition4_conv/Conv2Dhu  ?B
_
redzone_checker*?2?@8???@??H??Xb(model/ssd_res_block3_branch2_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??HޞXbIgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropFilterhu  ?B
_
redzone_checker*?2?@8???@??H??Xb(model/ssd_res_block2_branch2_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??HފXbIgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropFilterhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock4_denselayer1_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2Dhu  ?B
a
redzone_checker*?2?@8և?@??H??Xb*model/bbn_features_transition3_conv/Conv2Dhu  ?B
_
redzone_checker*?2?@8???@??H??Xb(model/ssd_res_block1_branch2_conv/Conv2Dhu  ?B
?
0cudnn_convolve_sgemm_sm35_ldg_nn_128x8x128x16x16??A*2?8???@???H???Xb.model/bbn_features_stemblock_stem1_conv/Conv2Dhu  B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 2-8?ź@??hH??iXbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8???@???H???Xb?model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?ѭ@?ѭH?ѭXbJgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
redzone_checker*?2?@8???@??HݒXbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8Ղ?@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch2c_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?׫@?׫H?׫Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
redzone_checker*?2?@8???@??HݑXbagradient_tape/model/bbn_features_denseblock4_denselayer6_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8?Ԫ@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer6_branch2c_conv/Conv2D/Conv2DBackpropInputhu  ?B
e
redzone_checker*?2?@8?˪@??H??Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8?ª@??H??Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void wgrad2d_grouped_direct_kernel<float, float, float, (cudnnTensorFormat_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnConvolutionStruct, cudnnFilterStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int)*?2?8؀?@؀?H؀?Xbbgradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
v
redzone_checker*?2?@8?٨@??H??Xb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8?Ѩ@޻H??Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu  ?B
Q
redzone_checker*?2?@8???@??H??Xbmodel/ssd_cls1conv2/Conv2Dhu  ?B
Q
redzone_checker*?2?@8???@ޗH??Xbmodel/ssd_cls3conv2/Conv2Dhu  ?B
Q
redzone_checker*?2?@8???@??H??Xbmodel/ssd_box1conv2/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?ĥ@?ĥH?ĥXb(model/ssd_res_block2_branch2_conv/Conv2Dhu  HB
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2Dhu  ?B
Q
redzone_checker*?2?@8???@??H??Xbmodel/ssd_box4conv2/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8ې?@ޠHޱXb?model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2Dhu  ?B
a
redzone_checker*?2?@8???@ހH??Xb*model/bbn_features_transition2_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block4_branch1a_conv/Conv2Dhu  ?B
Q
redzone_checker*?2?@8ء?@??H??Xbmodel/ssd_box3conv2/Conv2Dhu  ?B
_
redzone_checker*?2?@8ؠ?@??HީXb(model/ssd_res_block5_branch2_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block5_branch1a_conv/Conv2Dhu  ?B
Q
redzone_checker*?2?@8???@??H??Xbmodel/ssd_cls4conv2/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??HޱXb?model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2Dhu  ?B
Q
redzone_checker*?2?@8?آ@??H޹Xbmodel/ssd_cls5conv2/Conv2Dhu  ?B
?
redzone_checker*?2?@8?ޡ@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8?ԡ@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8?ġ@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
Q
redzone_checker*?2?@8٫?@??H??Xbmodel/ssd_box5conv2/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block2_branch1a_conv/Conv2Dhu  ?B
`
redzone_checker*?2?@8ڍ?@??H??Xb)model/ssd_res_block3_branch1a_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??HݑXbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??HޕXbKgradient_tape/model/ssd_res_block5_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B

redzone_checker*?2?@8?ڠ@??HޓXbHgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropInputhu  ?B

redzone_checker*?2?@8?ՠ@??H??XbHgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8?Ӡ@??H??XbLgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8?Ҡ@??H??XbLgradient_tape/model/ssd_res_block4_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
`
redzone_checker*?2?@8?Ƞ@??HޠXb)model/ssd_res_block1_branch1a_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8?Ġ@??HޑXbKgradient_tape/model/ssd_res_block4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8?à@??H??XbKgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
v
redzone_checker*?2?@8???@??HݍXb?model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8٥?@??H??Xb?model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8َ?@??H??Xb?model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2Dhu  ?B
v
redzone_checker*?2?@8???@??H??Xb?model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?
8???@???H???Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?
8???@???H???Xb?model/bbn_features_denseblock1_denselayer1_branch1b_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 8, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)u?R* 28???@??\H??]XbMgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8ܼ?@ܼ?Hܼ?Xb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28???@???H???XbLgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2Dhu  ?B
?
?void wgrad2d_grouped_direct_kernel<float, float, float, (cudnnTensorFormat_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnConvolutionStruct, cudnnFilterStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int)*?2?8?È@?ÈH?ÈXbbgradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2?8???@???H?ǃXbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)]?*28???@???H???XbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2?8???@???HΛ?XbQgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
f
redzone_checker*?2?@8???@??H??Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28???@???H???Xb*model/bbn_features_transition4_conv/Conv2Dhu  HB
`
redzone_checker*?2?@8???@??H??Xb)model/ssd_res_block5_branch1c_conv/Conv2Dhu  ?B
a
redzone_checker*?2?@8???@??HݟXb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb)model/ssd_res_block1_branch1a_conv/Conv2Dhu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8???@???H???XbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8???@??NH??OXbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xbagradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xbagradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28???@???H???XbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8???@???H???XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*2?8???@??LH??NXbQgradient_tape/model/bbn_features_stemblock_stem1_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8???@???H???Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28???@???H???Xb*model/bbn_features_transition3_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8¨?@¨?H¨?Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2	8???@??qH??qXbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8???@???H???Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void sgemm_largek_lds64<false, true, 5, 5, 4, 4, 4, 32>(float*, float const*, float const*, int, int, int, int, int, int, float const*, float const*, float, float, int, int, int*, int*)*?!*28???@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8???@???H???Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8???@јqH??qXbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8???@??pH??qXbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
redzone_checker*?2?@8Ô?@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8߀?@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2? 8???@???H???XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
f
redzone_checker*?2?@8???@??H??Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  ?B
e
redzone_checker*?2?@8???@??H??Xb.model/bbn_features_stemblock_stem1_conv/Conv2Dhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbMgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbMgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@ݺH??Xbagradient_tape/model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28???@??nH??oXbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu  HB
?
redzone_checker*?2?@8???@??H??XbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8­?@­?H­?XbJgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
u
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xbmodel/ssd_cls5conv2/Conv2Dhu  HB
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??HެXbagradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8?@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb(model/ssd_res_block1_branch2_conv/Conv2Dhu  HB
t
redzone_checker*?2?@8???@??H??Xb=gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??HݺXbbgradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block3_branch1c_conv/Conv2D/Conv2DBackpropFilterhu  ?B
s
redzone_checker*?2?@8???@??H??Xb<gradient_tape/model/ssd_cls3conv2/Conv2D/Conv2DBackpropInputhu  ?B
t
redzone_checker*?2?@8???@??HݿXb=gradient_tape/model/ssd_cls4conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block2_branch1c_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbJgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block4_branch1c_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??HޏXbLgradient_tape/model/ssd_res_block2_branch1c_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8þ?@??H??XbLgradient_tape/model/ssd_res_block5_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8»?@??H??XbLgradient_tape/model/ssd_res_block5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb?model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2Dhu  HB
?
redzone_checker*?2?@8???@??HޖXbKgradient_tape/model/ssd_res_block4_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??HݎXbLgradient_tape/model/ssd_res_block5_branch1c_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
s
redzone_checker*?2?@8???@??H??Xb<gradient_tape/model/ssd_box2conv2/Conv2D/Conv2DBackpropInputhu  ?B
s
redzone_checker*?2?@8???@??H??Xb<gradient_tape/model/ssd_box3conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8ï?@??H??XbKgradient_tape/model/ssd_res_block5_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??HސXbLgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8ê?@??H??XbKgradient_tape/model/ssd_res_block3_branch1c_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8ê?@??H??XbJgradient_tape/model/ssd_res_block4_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
t
redzone_checker*?2?@8???@??H??Xb=gradient_tape/model/ssd_cls3conv2/Conv2D/Conv2DBackpropFilterhu  ?B
s
redzone_checker*?2?@8é?@??H??Xb<gradient_tape/model/ssd_cls4conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbLgradient_tape/model/ssd_res_block4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8¦?@??H݉XbKgradient_tape/model/ssd_res_block5_branch1c_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8ä?@??iH??nXb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?B
s
redzone_checker*?2?@8???@??H??Xb<gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block4_branch1c_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbJgradient_tape/model/ssd_res_block3_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
s
redzone_checker*?2?@8???@??H??Xb<gradient_tape/model/ssd_cls5conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block3_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbKgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8???@??H??XbJgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8?@??H??XbJgradient_tape/model/ssd_res_block5_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
s
redzone_checker*?2?@8???@??H??Xb<gradient_tape/model/ssd_box4conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
s
redzone_checker*?2?@8???@??HݖXb<gradient_tape/model/ssd_box5conv2/Conv2D/Conv2DBackpropInputhu  ?B
t
redzone_checker*?2?@8???@??H??Xb=gradient_tape/model/ssd_cls5conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2	8???@҆kH??kXb*model/bbn_features_transition2_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?A8???@???H???Xb?model/bbn_features_denseblock1_denselayer1_branch1b_conv/Conv2Dhu  HB
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8???@??jH??jXbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8???@???H???XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28???@??H??Xb=gradient_tape/model/ssd_cls3conv2/Conv2D/Conv2DBackpropFilterh?u  HB
?
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28???@??H??Xb=gradient_tape/model/ssd_cls4conv2/Conv2D/Conv2DBackpropFilterh?u  HB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 8, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)|?R* 2X8???@ҰgH??gXbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb<gradient_tape/model/ssd_cls4conv2/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb<gradient_tape/model/ssd_cls3conv2/Conv2D/Conv2DBackpropInputhu  HB
m
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@???H???bmodel/re_lu_114/Reluhu  HB
m
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@???H???bmodel/re_lu_115/Reluhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb<gradient_tape/model/ssd_cls5conv2/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb<gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropInputhu  HB
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?y8Ė?@??eH??fXbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_c2r_32x32<float, true, false, 0u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)8 ??*?28???@??H??Xb.model/bbn_features_stemblock_stem1_conv/Conv2Dhdu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xbagradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xbagradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2B8???@???H???Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8???@???H???Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8Ǹ?@Ǹ?HǸ?Xb?model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 28???@??>H???XbMgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, true, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?28?׻@?׻H?׻bmodel/re_lu_115/Reluhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8ƃ?@ƃ?Hƃ?XbHgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@???H???XbHgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28???@??=H??>XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void sgemm_largek_lds64<false, true, 5, 5, 4, 4, 4, 32>(float*, float const*, float const*, int, int, int, int, int, int, float const*, float const*, float, float, int, int, int*, int*)*?!*28Ȍ?@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
?void sgemm_largek_lds64<false, true, 5, 5, 4, 4, 4, 32>(float*, float const*, float const*, int, int, int, int, int, int, float const*, float const*, float, float, int, int, int*, int*)*?!*28ǌ?@??H߶Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 5, 6, 4, 3, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)X?A*28???@???H???XbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu ??B
?
redzone_checker*?2?@8?շ@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterh
u  ?B
?
redzone_checker*?2?@8???@ݧH??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterh
u  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?R8???@??<H???Xb.model/bbn_features_stemblock_stem1_conv/Conv2Dhu  HB
?
:cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_64x8x256x8x32}?R* 2L8???@??ZH??\Xb*model/bbn_features_transition3_conv/Conv2Dhu  HB
I
redzone_checker*?2?@8?̴@??H??bmodel/re_lu_113/Reluh
u  ?B
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?R8???@??;H??<Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  HB
?
redzone_checker*?2?@8箴@??H??XbIgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropFilterh
u  ?B
t
redzone_checker*?2?@8???@??H??Xb=gradient_tape/model/ssd_box5conv2/Conv2D/Conv2DBackpropFilterh
u  ?B
t
redzone_checker*?2?@8祴@??H??Xb=gradient_tape/model/ssd_box3conv2/Conv2D/Conv2DBackpropFilterh
u  ?B
t
redzone_checker*?2?@8???@??H??Xb=gradient_tape/model/ssd_box2conv2/Conv2D/Conv2DBackpropFilterh
u  ?B
t
redzone_checker*?2?@8艴@??H??Xb=gradient_tape/model/ssd_box4conv2/Conv2D/Conv2DBackpropFilterh
u  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8苳@苳H苳Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb?model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28???@??;H??;Xb*model/bbn_features_transition1_conv/Conv2Dhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2B8???@???H???Xbbgradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8ɪ?@ɪ?Hɪ?Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8???@???H???Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2L8ʼ?@ʼ?Hʼ?Xb?model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8곦@곦H곦Xb?model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8ꞥ@ꞥHꞥXbKgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)P?2* 28???@???H???XbMgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8낡@낡H낡XbKgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8???@??5H??5Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xbagradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?Ԡ@?ԠH?ԠXbagradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, true, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2"8둟@둟H둟bmodel/re_lu_114/Reluhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2L8???@???H???Xb?model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8ʑ?@??3H??4Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8뙛@뙛H뙛Xb?model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2Dhu  HB
?
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28???@??H??XbKgradient_tape/model/ssd_res_block3_branch2_conv/Conv2D/Conv2DBackpropFilterh?u  HB
?
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28???@??H??XbKgradient_tape/model/ssd_res_block4_branch2_conv/Conv2D/Conv2DBackpropFilterh?u  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?
8???@???H???Xb?model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?Ė@?ĖH?ĖXbagradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2L8???@???H???Xb?model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)P?2* 28???@???H???XbMgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2,8???@??JH??JXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
redzone_checker*?2?@8???@??H??XbQgradient_tape/model/bbn_features_stemblock_stem1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
redzone_checker*?2?@8젔@??H??XbRgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xb)model/ssd_res_block2_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8???@??0H??1XbRgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8?ړ@ٶ0H??1XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?X8̹?@̹?H̹?XbKgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xbmodel/ssd_cls3conv2/Conv2Dhu  ?B

redzone_checker*?2?@8???@??H??XbHgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2B8?ӎ@?ӎH?ӎXbbgradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?@?H?Xb?model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2L8???@???H???Xb?model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
&cudnn_dgrad_sm35_ldg_nt_64x16x128x8x32{?`* 2$8???@??FH??GXbHgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropInputhu  HB
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?<8???@???H???Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?<8???@???H???XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
.cudnn_convolve_sgemm_sm35_ldg_nn_32x16x64x8x16S?2*2?8???@???H???Xb*model/bbn_features_transition2_conv/Conv2Dhu ??B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8???@???H???XbJgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2,?8???@???H???Xb(model/ssd_res_block2_branch2_conv/Conv2Dhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 256, 16, 16, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??*2?8???@???H???XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)P?2* 2?
8톉@톉H톉XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@??HޥXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputh	u  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28???@??BH??CXbLgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8???@???H???Xb?model/bbn_features_denseblock2_denselayer1_branch1b_conv/Conv2Dhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 256, 16, 16, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??*2? 8???@???H???XbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2? 8@Hbmodel/re_lu_113/Reluhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2? 8???@???H???XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?ԃ@?ԃH?ԃXb?model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8???@??H޵Xb?model/bbn_features_denseblock1_denselayer1_branch1b_conv/Conv2Dh	u  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8???@???H???Xbagradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)P?2* 2?
8?߀@?߀H?߀Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2<B8Ζ@???H???XbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2pB8??}@??}H??}Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2? 8ϫ}@ϫ}Hϫ}Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2pB8??|@??|H??|Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??|@??|H??|Xb=gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropFilterhu  HB
?
cvoid DSE::vector_fft<1, 2, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)5 ??*2? 8??|@??|H??|XbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??{@??{H??{Xb=gradient_tape/model/ssd_cls5conv2/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)]?*28??{@??{H??{Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??{@??{H??{Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28??z@?iH??XbLgradient_tape/model/ssd_res_block3_branch1c_conv/Conv2D/Conv2DBackpropFilterh?u  HB
?
?void fft2d_r2c_32x32<float, false, 0u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?28??z@??H??Xb.model/bbn_features_stemblock_stem1_conv/Conv2Dhdu  ?B
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 4, 6, 3, 2, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)O? *2,8еy@еyHеyXbLgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropInputhu  HB
?
cgemm_sm35_ldg_tn_64x8x64x16x16?A*28??y@?iH??XbLgradient_tape/model/ssd_res_block4_branch1c_conv/Conv2D/Conv2DBackpropFilterh?u  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??x@??xH??xXbKgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8??x@??xH??xXbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??x@??xH??xXbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*2S8??x@??xH??xXbQgradient_tape/model/bbn_features_stemblock_stem1_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*2S8??w@??wH??wXbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??w@??wH??wXbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)P?2* 2?8??w@??wH??wXbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??v@??vH??vXb?model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??u@??uH??uXbKgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
cvoid DSE::vector_fft<1, 2, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)5 ??*2?8??t@??tH??tXbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
cvoid DSE::vector_fft<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)8 ??*2? 8їs@їsHїsXbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??s@??sH??sXb(model/ssd_res_block4_branch2_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??r@??rH??rXb(model/ssd_res_block5_branch2_conv/Conv2Dhu  HB
?
cvoid DSE::vector_fft<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)8 ??*2? 8??r@??rH??rXb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??q@??qH??qXb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)?*?2?8??q@??qH??qbFgradient_tape/model/bbn_features_transition1_norm/FusedBatchNormGradV3hu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)H?*2?R8??q@??qH??qXbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??q@??qH??qXbJgradient_tape/model/ssd_res_block4_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2B8??q@??qH??qXbbgradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
0cudnn_convolve_sgemm_sm35_ldg_nn_128x8x128x16x16??A*2?8??q@??qH??qXb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?@8??p@??pH??pXbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??p@??pH??pXbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??p@??pH??pXbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??p@??pH??pXbJgradient_tape/model/ssd_res_block3_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??p@??pH??pXb(model/ssd_res_block3_branch2_conv/Conv2Dhu  HB
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??p@??pH??pXbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2$8їp@??H??Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)P?2* 2?
8??o@??7H??8XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??o@??oH??oXbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2 ?8??o@??oH??oXb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2??8??o@??oH??oXb*model/bbn_features_transition4_conv/Conv2Dhu  H?
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2 ?8??n@??nH??nXbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)?*?2 8??n@??nH??nbJgradient_tape/model/bbn_features_stemblock_stem1_norm/FusedBatchNormGradV3hu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??n@??nH??nXbJgradient_tape/model/ssd_res_block5_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
:cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_64x8x256x8x32}?R* 28??n@??6H؀8Xb*model/bbn_features_transition4_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8єn@єnHєnXb)model/ssd_res_block1_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8ғn@ܠ$H??%Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8??m@??mH??mXbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8ѣm@ѣmHѣmXb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2??8??m@??mH??mXbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??l@??lH??lXbbgradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??k@??kH??kXbbgradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?@8??k@??kH??kXbKgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
cvoid DSE::vector_fft<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)8 ??*2?8??k@??kH??kXbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
cvoid DSE::vector_fft<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)8 ??*2?8??k@??kH??kXb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??j@??jH??jXb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??j@??jH??jXbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2`B8??i@??iH??iXbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??i@??iH??iXb?model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2Dhu  HB
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2`B8ѳi@ѳiHѳiXb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2,8??h@??4H??4Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*  2?8үh@үhHүhXbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??g@??gH??gXb)model/ssd_res_block1_branch1b_conv/Conv2Dhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??g@??3Hِ4Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??g@??gH??gXb)model/ssd_res_block3_branch1b_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8һg@һgHһgXb)model/ssd_res_block4_branch1b_conv/Conv2Dhu  HB
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?y8??g@??gH??gXbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?y8??g@??gH??gXb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??f@??fH??fXb)model/ssd_res_block5_branch1b_conv/Conv2Dhu  HB
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?y8??f@??fH??fXb*model/bbn_features_transition1_conv/Conv2Dhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?<8??f@??fH??fXb=gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?y8??e@??eH??eXbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??e@??eH??eXb<gradient_tape/model/ssd_cls5conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??e@??eH??eXb<gradient_tape/model/ssd_cls4conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??e@??eH??eXb<gradient_tape/model/ssd_cls3conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??e@??eH??eXbKgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??e@??eH??eXbKgradient_tape/model/ssd_res_block4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??e@??eH??eXb<gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?<8??e@??eH??eXb=gradient_tape/model/ssd_cls3conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8Һe@ҺeHҺeXbKgradient_tape/model/ssd_res_block5_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??e@??eH??eXbKgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cudnn_convolve_sgemm_sm35_ldg_nn_64x16x64x16x16O?B*2L8??e@??eH??eXb*model/bbn_features_transition3_conv/Conv2Dhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?<8҂e@҂eH҂eXb=gradient_tape/model/ssd_cls4conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?<8??d@??dH??dXb=gradient_tape/model/ssd_cls5conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*28??d@ܚ!H??"bmodel/re_lu_113/Reluhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2 ?8??d@??dH??dXb(model/ssd_res_block1_branch2_conv/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8Ҿd@ҾdHҾdXbJgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??d@??dH??dXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void implicit_convolve_sgemm<float, float, 512, 6, 8, 3, 3, 5, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)u?R* 2L8??a@??aH??aXb*model/bbn_features_transition3_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??a@??aH??aXbJgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??a@??aH??aXbagradient_tape/model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??a@??aH??aXbmodel/ssd_cls1conv2/Conv2Dhu  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?#8??a@??H??!XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??`@??`H??`Xbmodel/ssd_cls3conv2/Conv2Dhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??_@??_H??_Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??_@??_H??_Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??_@??/H??/Xb?model/bbn_features_denseblock1_denselayer1_branch1b_conv/Conv2Dhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??_@??_H??_Xbmodel/ssd_cls5conv2/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??_@??_H??_Xbmodel/ssd_cls4conv2/Conv2Dhu  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?#8??_@??H?? Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?A

/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??^@??^H??^Xb(model/ssd_res_block2_branch2_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??\@??\H??\Xbbgradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??\@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch2c_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??\@??\H??\Xbbgradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??Z@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1b_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
?void DSE::regular_fft_pad<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??*2? 8??Y@??YH??YXb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2
?8??X@??XH??XXbagradient_tape/model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2
?8??X@??XH??XXbagradient_tape/model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void DSE::regular_fft_pad<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??*2? 8??X@??XH??XXbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?#8??W@??H??XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2PB8լW@լWHլWXbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2PB8??W@??WH??WXb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??*2?8??V@??VH??VXb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?-8??U@??UH??UXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?#8??U@??H??Xb*model/bbn_features_transition1_conv/Conv2Dhu  ?A
?
?void DSE::regular_fft_pad<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??*2?8??U@??UH??UXbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8??U@??*H??*Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??U@??UH??UXb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*2-8??U@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??U@??UH??UXb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8??T@??HޣXb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?A
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??T@??TH??TXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)\?*28??S@??SH??SXbMgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??S@??SH??SXbLgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?8ՙS@ՙSHՙSXbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??R@??RH??RXbRgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2?8??R@۩)Hڰ)Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  HB
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?a8եR@եRHեRXbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_r2c_32x32<float, true, 0u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)? ??*?28??R@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?a8??R@??RH??RXb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?a8??Q@??QH??QXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8??Q@??(H??(XbMgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??P@??PH??PXb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??P@??PH??PXb?model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)P?2* 28??P@??PH??PXbHgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2$8??P@??HݭXbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??P@??PH??PXbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??O@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch1b_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2	?8֥O@֥OH֥OXbagradient_tape/model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2	?8??N@??NH??NXbagradient_tape/model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*2-8ոN@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?[8??M@??MH??MXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??M@??MH??MXb?model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2Dhu  HB
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?8??L@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?[8??L@??LH??LXb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?[8??L@??LH??LXbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??K@??KH??KXb?model/bbn_features_denseblock2_denselayer1_branch1b_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??K@??KH??KXbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_c2r_32x32<float, false, false, 0u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)8 ??*?28??K@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*2-8??J@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8??J@??%H??%Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2,8??J@??JH??JXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2? 8֨J@֨JH֨JXb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu ??B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?,8??J@??JH??JXbLgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
q
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??I@??IH??IXbmodel/ssd_cls1conv2/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??I@??IH??IXbbgradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8??I@??H݋Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?A
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??H@??HH??HXbbgradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??H@??HH??HXb?model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2Dhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?y8??H@??HH??HXbLgradient_tape/model/bbn_features_transition4_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??H@??HH??HXbJgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2 ?8ֱG@ֱGHֱGXbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void DSE::regular_fft_clip<1, 2, 256, 16, 16, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??*2?8??F@??FH??FXb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2?8??F@??FH??FXb?model/bbn_features_denseblock2_denselayer1_branch2c_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??F@ݶH??Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??F@??FH??FXbagradient_tape/model/bbn_features_denseblock4_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??E@??EH??EXb?model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2Dhu  HB
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2?88??E@??EH??EXbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2,?8??E@??EH??EXb)model/ssd_res_block2_branch1a_conv/Conv2Dhu  ?B

/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??E@??EH??EXb(model/ssd_res_block1_branch2_conv/Conv2Dhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8׫E@׫EH׫EXbKgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?8??E@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??D@??DH??DXbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??D@??DH??DXbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)P?2* 2?
8??D@??DH??DXbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2&8׬D@??H??Xb)model/ssd_res_block1_branch1b_conv/Conv2Dhu  HB
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??D@??DH??DXbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 8, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)|?R* 28??D@??DH??DXbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??C@??CH??Cbmodel/re_lu_114/Reluhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2@B8??C@??CH??CXbagradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2@B8??C@??CH??CXb?model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2Dhu  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??C@??H߰XbHgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropInputh$u  ?B
?
hvoid fft1d_r2c_32<float, float, float2, true, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?y8??C@??CH??CXb*model/bbn_features_transition4_conv/Conv2Dhu  ?A
?
?void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??C@??CH??Cbmodel/re_lu_115/Reluhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??C@??CH??CXbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8׀C@׀CH׀CXbHgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??C@??CH??CXbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 6, 7, 3, 3, 5, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)M?2* 28??B@??BH??BXb*model/bbn_features_transition4_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8׿B@׿BH׿BXbHgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??B@??BH??BXb<gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropInputhu  HB
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28ׂB@??H??XbHgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropInputh$u  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*28??A@ޗH??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??A@??AH??AXb?model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 28??@@??H??XbKgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*2-8??@@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 512, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float) ?*?2?8??@@??@H??@b4model/bbn_features_transition1_norm/FusedBatchNormV3hu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2A8??@@??@H??@Xbmodel/ssd_cls4conv2/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*2$8??@@??@H??@XbIgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropFilterhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8???@???H???Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2? 8???@???H???Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu ??B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2<B8???@???H???XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8???@???H???XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2<B8???@???H???Xb*model/bbn_features_transition2_conv/Conv2Dhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8???@???H???Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8??>@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??>@??>H??>Xbagradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*28??>@ݾH??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??=@??=H??=Xb?model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2Dhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?"8??=@??H?? XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 512, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float) ?*?2 8??=@??=H??=b8model/bbn_features_stemblock_stem1_norm/FusedBatchNormV3hu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??=@??=H??=Xb)model/ssd_res_block3_branch1c_conv/Conv2Dhu  HB
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2 ?8؟=@؟=H؟=Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu ??B
j
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??=@??=H??=bmodel/re_lu_115/Reluhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??=@??=H??=Xb)model/ssd_res_block1_branch1c_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8؁=@؁=H؁=XbLgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?"8??<@??Hۍ Xb*model/bbn_features_transition2_conv/Conv2Dhu  ?A
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2!x8??<@??<H??<Xb*model/bbn_features_transition2_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??<@??<H??<XbKgradient_tape/model/ssd_res_block3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??<@??<H??<Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??<@??<H??<Xb)model/ssd_res_block4_branch1c_conv/Conv2Dhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??<@??<H??<Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*28??<@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??<@??<H??<Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2!x8ػ<@ػ<Hػ<XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??<@??<H??<XbKgradient_tape/model/ssd_res_block5_branch2_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8ت<@ت<Hت<Xb)model/ssd_res_block5_branch1c_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??;@??;H??;Xb)model/ssd_res_block5_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??;@??;H??;Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2?08??;@??;H??;Xbbgradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu ??B
j
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??;@??;H??;bmodel/re_lu_114/Reluhu  HB
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)?*?28??;@??;H??;bKgradient_tape/model/bbn_features_stemblock_stem2a_norm/FusedBatchNormGradV3hu  ?B
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 5, 6, 4, 3, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)X?A*2+8??;@??H??XbHgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropInputhu ??B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??;@??;H??;Xbbgradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??;@??;H??;XbKgradient_tape/model/ssd_res_block4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8ة;@ة;Hة;XbKgradient_tape/model/ssd_res_block5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)~* 2?8??;@??;H??;Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??;@??;H??;Xbbgradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??:@??:H??:XbKgradient_tape/model/ssd_res_block4_branch1c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??:@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 4, 6, 3, 2, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)O? *2,8??:@??:H??:XbJgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2??8??:@??:H??:Xb*model/bbn_features_transition3_conv/Conv2Dhu  H?
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??:@??:H??:Xb)model/ssd_res_block4_branch1a_conv/Conv2Dhu  HB
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*2[8??:@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
cvoid DSE::vector_fft<1, 2, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)5 ??*2?8??:@??:H??:Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??:@??:H??:XbKgradient_tape/model/ssd_res_block2_branch1c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??:@??:H??:XbHgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??:@??:H??:XbKgradient_tape/model/ssd_res_block3_branch1c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??:@??:H??:Xb?model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??:@??:H??:Xb)model/ssd_res_block3_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??:@??:H??:XbHgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??9@??9H??9Xbagradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??9@??9H??9Xb?model/bbn_features_denseblock4_denselayer1_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??9@??9H??9XbKgradient_tape/model/ssd_res_block5_branch1c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_c2r_32x32<float, true, false, 0u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)8 ??*?28??9@??H??Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??9@??9H??9Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??9@??9H??9Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2?8??8@??H??Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::pooling_bw_kernel_max<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor) ? *?2,8??8@??8H??8b5gradient_tape/model/max_pooling2d/MaxPool/MaxPoolGradhu  ?B
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?8??8@??8H??8Xbbgradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??8@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch2c_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*2	8??8@??8H??8Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)M?2* 2$8??8@??H??XbIgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*2[8??7@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch1b_conv/Conv2Dh$u  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??7@??7H??7Xbbgradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??7@??7H??7Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??7@??7H??7Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??7@??7H??7Xbbgradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?28??7@??7H??7b"gradient_tape/model/re_lu/ReluGradhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??7@??7H??7Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?28ٌ7@ٌ7Hٌ7b%gradient_tape/model/re_lu_19/ReluGradhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?25B8??6@??6H??6Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu  ?B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??6@??6H??6Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?A8??6@??6H??6Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??6@??6H??6Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?A8??6@??6H??6Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?A8??6@??6H??6Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??6@??H܅Xb?model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??6@ވH??Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  HB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2,8??6@??6H??6XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2,8??6@ݟHܣXbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28پ6@??H??Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2? 8??6@??6H??6XbLgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
.cudnn_convolve_sgemm_sm35_ldg_nn_32x16x64x8x16S?2*2&8??6@??6H??6Xb*model/bbn_features_transition4_conv/Conv2Dhu ??B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2? 8ُ6@ُ6Hُ6XbKgradient_tape/model/ssd_res_block4_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2? 8ً6@ً6Hً6XbKgradient_tape/model/ssd_res_block3_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
cvoid DSE::vector_fft<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)8 ??*2?8??5@??5H??5XbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??5@??5H??5Xbagradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??5@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8??5@??H??Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu  ?A
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??5@??5H??5Xb?model/bbn_features_denseblock2_denselayer1_branch2c_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??5@??5H??5Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch2a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??5@??5H??5Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??5@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??5@??5H??5Xb?model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??5@??5H??5Xbbgradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
cvoid DSE::vector_fft<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float2*, int, int3, int3)8 ??*2?8??5@??5H??5XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2? 8??5@??5H??5XbKgradient_tape/model/ssd_res_block5_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_r2c_32x32<float, false, 0u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?28??5@??H??Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  ?B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2? 8??5@??5H??5Xb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu ??B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??5@??5H??5XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28ٓ5@ٓ5Hٓ5XbKgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??5@??5H??5Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??5@??5H??5Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??5@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??4@??4H??4Xbbgradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2?8??4@??4H??4XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??4@??4H??4Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2	8ک4@??H??Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??4@??4H??4Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 2,8??4@??4H??4Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??4@??4H??4Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch2b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??4@??4H??4Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch2b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??3@??3H??3Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch2b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*2
8??3@??3H??3Xbbgradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?8??3@??3H??3Xbbgradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??3@??3H??3Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch2b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??3@??3H??3Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropInputhu  HB
?	
?	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??3@??3H??3bAdam/gradients/AddN_30hu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??3@??3H??3Xbbgradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??3@??3H??3Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?<8??3@??3H??3Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??3@??3H??3Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8??3@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??3@??3H??3Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??3@??3H??3Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?<8??3@??3H??3XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??3@??3H??3Xb(model/ssd_res_block5_branch2_conv/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??3@??3H??3Xb(model/ssd_res_block4_branch2_conv/Conv2Dhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??3@??3H??3bAdam/gradients/AddN_33hu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??3@??3H??3Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??3@??3H??3Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch2b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)H?*2?8٫3@٫3H٫3Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??3@??3H??3Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??3@??3H??3Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch2c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??3@??3H??3Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2 ?8ٙ3@ٙ3Hٙ3Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8??3@??H??Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??3@??3H??3Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch2c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8ً3@ً3Hً3Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch2b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?<8??3@??3H??3Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?<8??2@??2H??2XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?<8??2@??2H??2Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2?
8??2@??2H??2Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch2c_conv/Conv2D/Conv2DBackpropInputhu  HB
?
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)  ? *2?<8??2@??2H??2Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??2@??2H??2XbKgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?8??2@??2H??2Xbbgradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??2@??2H??2XbJgradient_tape/model/ssd_res_block5_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
q
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??2@??2H??2Xbmodel/ssd_cls3conv2/Conv2Dhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??2@??2H??2Xb(model/ssd_res_block3_branch2_conv/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??2@??2H??2XbJgradient_tape/model/ssd_res_block3_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2 ?8??2@??2H??2Xb)model/ssd_res_block1_branch1a_conv/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??2@??2H??2XbJgradient_tape/model/ssd_res_block4_branch2_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??2@??2H??2Xb?model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2Dhu  ?B
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2?(8٨2@٨2H٨2Xbbgradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2?8??2@??2H??2XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*2-8??2@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8??1@??H??Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8??1@??H??Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  ?A
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8??1@??HމXb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  ?A
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 28٭1@??H??XbLgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??1@??1H??1Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch2c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??1@??1H??1Xbagradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8??0@??0H??0Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  HB
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2?8??0@??0H??0Xb*model/bbn_features_transition2_conv/Conv2Dhu ??B
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??/@??/H??/Xb?model/bbn_features_denseblock1_denselayer2_branch1b_conv/Conv2Dhu  HB
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?"8??/@??HܰXbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
&cudnn_dgrad_sm35_ldg_nt_64x16x128x8x32{?`* 28??/@??H??XbKgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*28??/@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??/@??/H??/Xb?model/bbn_features_denseblock1_denselayer1_branch2c_conv/Conv2Dhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8??/@??HܠXb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu  ?A
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28??/@??H??XbJgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??/@??/H??/Xb?model/bbn_features_denseblock1_denselayer2_branch2c_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??/@??/H??/Xb?model/bbn_features_denseblock1_denselayer1_branch2b_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??/@??/H??/Xb?model/bbn_features_denseblock1_denselayer2_branch2b_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28??/@??/H??/Xbbgradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
[void cudnn::ops::scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*?2??8??/@??H??XbLgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2!i8??/@??/H??/Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?"8??/@??H??Xb*model/bbn_features_transition2_conv/Conv2Dhu  ?A
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??/@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch1b_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
?void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 512, true, 1>(float, float, float, float, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float*, float*, float const*, float const*, float)?*?2?8??/@??/H??/bFgradient_tape/model/bbn_features_transition2_norm/FusedBatchNormGradV3hu  ?B
?
[void cudnn::ops::scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*?2??8??/@??H??XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??/@??/H??/Xb)model/ssd_res_block2_branch1a_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??/@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch1b_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 28??.@??H??XbKgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?8??.@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
q
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28??.@??H??Xb)model/ssd_res_block3_branch1b_conv/Conv2Dh$u  ?B
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?8??.@??H??XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2&8ڇ.@??H??Xb(model/ssd_res_block2_branch2_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2	8??.@??H??Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu  HB
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28ڂ.@ߟH??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch2c_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??.@??H??XbLgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2-B8??-@??-H??-Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu  ?B
q
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28??-@??H??Xb)model/ssd_res_block1_branch1b_conv/Conv2Dh$u  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??-@??-H??-Xb<gradient_tape/model/ssd_cls3conv2/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 28??-@??H??Xb=gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??-@??-H??-Xb?model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2Dhu  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8??-@??H??	XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?A
Z
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28ۜ-@??H??bmodel/re_lu_115/Reluh$u  ?B
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??-@??-H??-Xb?model/bbn_features_denseblock1_denselayer3_branch2b_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??-@??-H??-Xb?model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2Dhu  HB
?
?void tensorflow::functor::PadInputCustomKernelNCHW<float, 4>(int, float const*, tensorflow::functor::Dimension<4>, float*, tensorflow::functor::Dimension<4>, tensorflow::functor::Dimension<(4)-(2)>, float)*?28??-@??-H??-Xb/model/bbn_features_stemblock_stem2b_conv/Conv2Dhu  ?B
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*2?8??,@??,H??,Xb?model/bbn_features_denseblock1_denselayer3_branch2c_conv/Conv2Dhu  HB
q
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28??,@ߙH??Xb)model/ssd_res_block5_branch1b_conv/Conv2Dh$u  ?B
q
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28??,@??H??Xb)model/ssd_res_block4_branch1b_conv/Conv2Dh$u  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*28??,@??HޏXbbgradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
Z
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28??,@??H??bmodel/re_lu_114/Reluh$u  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??,@??,H??,Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)H?*2?8??,@??,H??,Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?8??,@??,H??,Xbbgradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 28ڡ,@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2!Z8ڠ,@ڠ,Hڠ,Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??,@??H??XbKgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*28??,@??H??bmodel/re_lu_114/Reluhu  HB
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 4, 6, 3, 2, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)O? *2 8??+@??+H??+XbJgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2	8??+@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void tensorflow::functor::PadInputCustomKernelNCHW<float, 4>(int, float const*, tensorflow::functor::Dimension<4>, float*, tensorflow::functor::Dimension<4>, tensorflow::functor::Dimension<(4)-(2)>, float)*?28??+@??+H??+XbRgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??+@??H??XbKgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
?void cudnn::pooling_bw_kernel_avg<float, float, cudnn::averpooling_func<float, true>, 2, false>(cudnnTensorStruct, float const*, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)( ? *?2 8??+@??+H??+b1gradient_tape/model/average_pooling2d/AvgPoolGradhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??*2?8??+@??+H??+XbRgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void tensorflow::functor::PadInputCustomKernelNCHW<float, 4>(int, float const*, tensorflow::functor::Dimension<4>, float*, tensorflow::functor::Dimension<4>, tensorflow::functor::Dimension<(4)-(2)>, float)*?28??+@??+H??+XbQgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 256, 16, 16, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??*2?8??+@??+H??+XbQgradient_tape/model/bbn_features_stemblock_stem2a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_r2c_32x32<float, false, 0u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??+@??H??XbMgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void wgrad2d_grouped_direct_kernel<float, float, float, (cudnnTensorFormat_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnConvolutionStruct, cudnnFilterStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int)*?2?8??+@??
H??
Xb=gradient_tape/model/ssd_box3conv2/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*2-8??+@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8??*@??*H??*Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch2a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?
8??*@??*H??*Xbagradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??*@??H??XbKgradient_tape/model/ssd_res_block5_branch1b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2?8??*@??*H??*Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?8۹*@??H݇Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??*@??H??XbKgradient_tape/model/ssd_res_block4_branch1b_conv/Conv2D/Conv2DBackpropInputh$u  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*28??)@??H??Xb)model/ssd_res_block3_branch1b_conv/Conv2Dhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?
8??)@??)H??)Xb?model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2? 8??)@??)H??)Xb?model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2Dhu ??B
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 4, 6, 3, 2, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)O? *28??)@??)H??)Xb<gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?2/8??)@??)H??)Xb)model/ssd_res_block3_branch1a_conv/Conv2Dhu  ?B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2 ?8??)@??)H??)Xbagradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu ??B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8??)@??)H??)Xb*model/bbn_features_transition2_conv/Conv2Dhu ??B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8ې)@ې)Hې)XbLgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 28??)@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::conv2d_grouped_direct_kernel<float, float, float, float, float, float, true, false, 0, 0, 0>(cudnnTensorStruct, float const*, cudnnFilterStruct, float const*, cudnnConvolutionStruct, cudnnTensorStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, float const*, float const*, cudnnActivationStruct) *?28??(@??(H??(Xbmodel/ssd_box1conv2/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2
?8??(@??(H??(Xbbgradient_tape/model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
hvoid fft1d_r2c_32<float, float, float2, true, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?@8??(@??(H??(Xb*model/bbn_features_transition3_conv/Conv2Dhu  ?A
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2
?8??(@??(H??(Xbbgradient_tape/model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2? 8??(@??(H??(Xbbgradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??(@??(H??(Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?8??(@??H??Xb/model/bbn_features_stemblock_stem2a_conv/Conv2Dhu  ?A
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2??8??(@??(H??(Xb(model/ssd_res_block2_branch2_conv/Conv2Dhu  H?
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, false, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)P?*28??(@??H??Xb)model/ssd_res_block5_branch1b_conv/Conv2Dhu  HB
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?
8??'@??'H??'Xbbgradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8??'@??H??Xb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu  ?A
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 7, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)P?2* 28??'@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2	8??'@??H??Xb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??'@??'H??'XbKgradient_tape/model/ssd_res_block1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::im2col4d_kernel<float, long>(cudnn::cnn::im2col4d_params, cudnnConvolutionStruct, cudnnTensor4dStruct, float const*, float*) *?2?8۟'@۟'H۟'Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
?void wgrad2d_grouped_direct_kernel<float, float, float, (cudnnTensorFormat_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnConvolutionStruct, cudnnFilterStruct, float*, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int)*?2?	8??'@??'H??'Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2	8??'@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??&@??H??XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?	8??&@??&H??&Xbagradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2?8??&@??&H??&Xb?model/bbn_features_denseblock1_denselayer1_branch1b_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)\?*28??&@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?8??&@??&H??&XbHgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?8??&@??&H??&Xbbgradient_tape/model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*28??&@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer8_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?@8??%@??%H??%XbLgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?	8??%@??%H??%Xb?model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8??%@??H??Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)>?*28??%@??H??bmodel/re_lu_115/Reluhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8??%@??%H??%Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch2a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
&cudnn_dgrad_sm35_ldg_nt_64x16x128x8x32{?`* 28??%@??%H??%XbHgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropInputhu  HB
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2&B8??%@??%H??%Xb?model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)H?*2?8??%@??%H??%Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*?28??%@??H??bmodel/concatenate_3/concathu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28??%@ݴH??XbJgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*28??$@??H??Xb)model/ssd_res_block4_branch1b_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2?8??$@??$H??$Xbbgradient_tape/model/bbn_features_denseblock1_denselayer2_branch2a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28??$@ݯH??Xbagradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2?8??$@??$H??$Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu ??B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2	?8??$@??$H??$Xbbgradient_tape/model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 28??$@??HޮXbLgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*?28??$@??$H??$bmodel/re_lu_19/Reluhu  ?B
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?	8??$@??$H??$Xbbgradient_tape/model/bbn_features_denseblock3_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*?28ۧ$@ۧ$Hۧ$bmodel/re_lu/Reluhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?8??$@??$H??$Xbbgradient_tape/model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2	?8??$@??$H??$Xbbgradient_tape/model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8??$@??$H??$Xb?model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2Dhu ??B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??#@??#H??#Xbagradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??#@??#H??#Xbagradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*28??#@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??#@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2Dhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??#@??#H??#Xbagradient_tape/model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
[void cudnn::ops::scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*?2??8??#@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??#@??#H??#Xb)model/ssd_res_block1_branch1a_conv/Conv2Dhu  HB
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?8??#@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 5, 6, 4, 3, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)X?A*2+8??#@??#H??#XbKgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??"@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8??"@??"H??"Xbbgradient_tape/model/bbn_features_denseblock2_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu ??B
?	
?	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28??"@??"H??"bAdam/gradients/AddN_31hu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?8??"@??"H??"Xbbgradient_tape/model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??"@??"H??"Xb)model/ssd_res_block4_branch1b_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??"@??"H??"Xb)model/ssd_res_block1_branch1b_conv/Conv2Dhu  ?B
?
?void cudnn::bn_fw_tr_1C11_kernel_NCHW<float, float, 512, true, 1>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, float const*, float const*, float, float, float*, float*, float*, float*, float, float) ?*?28??"@??"H??"b9model/bbn_features_stemblock_stem2a_norm/FusedBatchNormV3hu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??"@??"H??"Xb)model/ssd_res_block5_branch1b_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, true>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8ۧ"@ۧ"Hۧ"Xb)model/ssd_res_block3_branch1b_conv/Conv2Dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2*@8??"@??"H??"Xb?model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??"@??"H??"Xbagradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??"@??"H??"Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28??"@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??!@??!H??!Xbagradient_tape/model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8??!@??!H??!Xb?model/bbn_features_denseblock1_denselayer3_branch1a_conv/Conv2Dhu  HB
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2&8??!@??H??Xbmodel/ssd_cls1conv2/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2	8??!@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??!@??!H??!XbKgradient_tape/model/ssd_res_block4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??!@??!H??!XbKgradient_tape/model/ssd_res_block5_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??!@??!H??!XbKgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??!@??!H??!Xb?model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2Dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??!@??!H??!XbKgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void tensorflow::functor::PadInputCustomKernelNCHW<float, 4>(int, float const*, tensorflow::functor::Dimension<4>, float*, tensorflow::functor::Dimension<4>, tensorflow::functor::Dimension<(4)-(2)>, float)*?28??!@??!H??!Xb.model/bbn_features_stemblock_stem1_conv/Conv2Dhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)P?2* 2?
8??!@??!H??!Xbagradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2L8??!@??!H??!Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2!K8ܹ!@ܹ!Hܹ!Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 5, 6, 4, 3, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)X?A*2+8??!@??H??XbKgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropInputhu ??B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2	8??!@ݾH??Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8??!@??H??Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu  HB
?
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28??!@?oH??Xb?model/bbn_features_denseblock4_denselayer1_branch1b_conv/Conv2Dh$u  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28?? @?? H?? Xbagradient_tape/model/bbn_features_denseblock4_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8?? @?? H?? XbQgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28?? @?oH߬Xb?model/bbn_features_denseblock2_denselayer1_branch1b_conv/Conv2Dh$u  ?B
?
9cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_32x16x32x8x8D?$*2?8?? @?? H?? Xb.model/bbn_features_stemblock_stem3_conv/Conv2Dhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2(@8?? @?? H?? Xb?model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2Dhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8?? @?? H?? Xbbgradient_tape/model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8?? @?? H?? XbPgradient_tape/model/bbn_features_stemblock_stem3_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8?? @?? H?? Xbbgradient_tape/model/bbn_features_denseblock4_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 5, 6, 4, 3, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)X?A*2+8?? @??H??XbKgradient_tape/model/ssd_res_block4_branch1b_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?8ܦ @ܦ Hܦ Xbbgradient_tape/model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8?? @?? H?? Xbagradient_tape/model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8?? @?? H?? Xb?model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2Dhu  ?B
?
9cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_32x16x32x8x8D?$*2?8?? @?? H?? Xb?model/bbn_features_denseblock1_denselayer2_branch1a_conv/Conv2Dhu  HB
?
?void tensorflow::functor::PadInputCustomKernelNCHW<float, 4>(int, float const*, tensorflow::functor::Dimension<4>, float*, tensorflow::functor::Dimension<4>, tensorflow::functor::Dimension<(4)-(2)>, float)*?28?? @?? H?? XbQgradient_tape/model/bbn_features_stemblock_stem1_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
 sgemm_sm35_ldg_nn_64x16x64x16x16G?B*28܏ @?mH??Xb?model/bbn_features_denseblock3_denselayer1_branch1b_conv/Conv2Dh$u  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*28܍ @??H??XbKgradient_tape/model/ssd_res_block2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)\?*28??@??H??XbKgradient_tape/model/ssd_res_block2_branch2_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xb)model/ssd_res_block1_branch1b_conv/Conv2Dhu  HB
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2?8??@??H??Xb?model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xb?model/bbn_features_denseblock3_denselayer1_branch1b_conv/Conv2Dhu  HB
?
9cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_32x16x32x8x8D?$*2?8??@??H??Xb?model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xb)model/ssd_res_block4_branch1b_conv/Conv2Dhu  HB
?
?void cudnn::detail::dgrad_engine<float, 128, 6, 8, 3, 3, 5, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)|?R* 28??@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
 sgemm_sm35_ldg_nt_64x16x64x16x16G?@*28??@?mH??XbLgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropFilterh$u  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??@??H??Xb?model/bbn_features_denseblock3_denselayer8_branch1a_conv/Conv2Dhu  HB
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, true, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)O?*2?8??@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28??@??H??Xb<gradient_tape/model/ssd_cls2conv2/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)3 ??*?2?8??@??H??Xbbgradient_tape/model/bbn_features_denseblock3_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xb)model/ssd_res_block3_branch1b_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cub::DeviceSegmentedRadixSortKernel<cub::DeviceRadixSortPolicy<float, int, int>::Policy700, true, true, float, int, cub::TransformInputIterator<int, tensorflow::impl::SegmentOffsetCreator, cub::CountingInputIterator<int, long>, long>, int>(float const*, float*, int const*, int*, cub::TransformInputIterator<int, tensorflow::impl::SegmentOffsetCreator, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::impl::SegmentOffsetCreator, cub::CountingInputIterator<int, long>, long>, int, int, int)`?D*?28??@??H߳b2compute_loss/cond/else/_1/compute_loss/cond/TopKV2hu  zB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2L8??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbLgradient_tape/model/ssd_res_block3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbLgradient_tape/model/ssd_res_block4_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2?8܌@޵H??Xb?model/bbn_features_denseblock3_denselayer7_branch1a_conv/Conv2Dhu  HB
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8??@??H??Xb?model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2Dhu ??B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2&@8??@??H??Xb?model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbIgradient_tape/model/ssd_feature_extend2_conv2/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8ۄ@ۄHۄXbLgradient_tape/model/ssd_res_block2_branch1c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2	8??@??H??Xb?model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2Dhu  HB
?
?void DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)8 ??* 2?8??@??H??Xb?model/bbn_features_denseblock1_denselayer1_branch1a_conv/Conv2Dhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??@??H??Xb?model/bbn_features_denseblock4_denselayer1_branch1b_conv/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*28??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbLgradient_tape/model/ssd_res_block5_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbIgradient_tape/model/ssd_feature_extend3_conv2/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbKgradient_tape/model/ssd_res_block4_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xb)model/ssd_res_block5_branch1b_conv/Conv2Dhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)J?"* 2?8??@??HޱXb?model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2Dhu  ?A
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*2+8??@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch2a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)\?*28??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu ??B
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbLgradient_tape/model/ssd_res_block5_branch1c_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbKgradient_tape/model/ssd_res_block5_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbKgradient_tape/model/ssd_res_block2_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??XbKgradient_tape/model/ssd_res_block3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)  ?D*?2?8??@??H??Xbbgradient_tape/model/bbn_features_denseblock4_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 4> const, Eigen::DSizes<int, 4> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28??@??H??b'gradient_tape/model/concatenate_3/Slicehu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8??@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer5_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
r
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xbmodel/ssd_box1conv2/Conv2Dhu  HB
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2
8??@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer4_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
r
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xbmodel/ssd_box3conv2/Conv2Dhu  HB
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)8?"* 2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer2_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?A
r
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8??@??H??Xbmodel/ssd_box4conv2/Conv2Dhu  HB
?
?void cudnn::detail::dgrad2d_alg1_1<float, 0, 5, 6, 4, 3, 4, false, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int)X?A*28??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer8_branch1b_conv/Conv2D/Conv2DBackpropInputhu ??B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*).?A*?2!<8??@??H??Xbagradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)@ ?B*?2?8??@??H??Xbbgradient_tape/model/bbn_features_denseblock2_denselayer3_branch1a_conv/Conv2D/Conv2DBackpropFilterhu ??B
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??@??H??Xbbgradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropFilterhu  ?B
?
/cgemm_strided_batched_sm35_ldg_nt_64x8x64x16x16?@*2?8??@??H??Xb?model/bbn_features_denseblock3_denselayer1_branch1b_conv/Conv2Dhu  HB
?
9cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_32x16x32x8x8D?$*2?8??@??H??Xb?model/bbn_features_denseblock3_denselayer4_branch1a_conv/Conv2Dhu  HB
?
?void DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)@ ??* 2?8??@??H??Xbagradient_tape/model/bbn_features_denseblock1_denselayer3_branch1b_conv/Conv2D/Conv2DBackpropInputhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)( ?L*?2$@8??@??H??Xb?model/bbn_features_denseblock4_denselayer3_branch1a_conv/Conv2Dhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)\?*2L8??@??H??Xbagradient_tape/model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)H?*2?8ܻ@ܻHܻXbagradient_tape/model/bbn_features_denseblock2_denselayer1_branch1a_conv/Conv2D/Conv2DBackpropInputhu  HB
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 1024, 2, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) ?`*?2?8??@??H??bwgradient_tape/model/bbn_features_stemblock_stem1_conv/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void precomputed_convolve_sgemm<float, 128, 5, 5, 3, 3, 3, 1, false>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, int*)P?*2?8ܯ@??H??Xb?model/bbn_features_denseblock3_denselayer6_branch1a_conv/Conv2Dhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)\?*28??@??H??XbKgradient_tape/model/ssd_res_block1_branch2_conv/Conv2D/Conv2DBackpropFilterhu  HB
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2??8ܦ@ܦHܦXb(model/ssd_res_block1_branch2_conv/Conv2Dhu  H?
r
/cgemm_strided_batched_sm35_ldg_tn_64x8x64x16x16?A*2?8ܐ@ܐHܐXbmodel/ssd_box5conv2/Conv2Dhu  HB
?
:cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_64x8x256x8x32}?R* 28??@ޭH??Xb(model/ssd_res_block1_branch2_conv/Conv2Dhu  HB