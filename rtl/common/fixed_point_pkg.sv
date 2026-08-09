package fixed_point_pkg;
    parameter int DATA_W = 16;
    parameter int ACC_W  = 48;

    // ------------------------------------------------------------------
    // Activation formats. A stored integer N represents N / 2^F.
    // ------------------------------------------------------------------
    parameter int EEG_INPUT_F   = 12;

    parameter int CONV1_OUT_F   = 13;
    parameter int BN1_OUT_F     = 11;
    parameter int RELU1_OUT_F   = 11;

    parameter int CONV2_OUT_F   = 10;
    parameter int BN2_OUT_F     = 11;
    parameter int RELU2_OUT_F   = 12;
    parameter int POOL1_OUT_F   = 12;

    parameter int CONV3_OUT_F   = 10;
    parameter int BN3_OUT_F     = 12;
    parameter int RELU3_OUT_F   = 12;
    parameter int POOL2_OUT_F   = 13;

    parameter int RESHAPE_OUT_F = POOL2_OUT_F;
    parameter int GRU_INPUT_F   = RESHAPE_OUT_F;
    parameter int GRU_OUT_F     = 15;
    parameter int FLATTEN_OUT_F = GRU_OUT_F;

    parameter int FC1_OUT_F     = 13;
    parameter int FC_RELU_OUT_F = 13;
    parameter int FC_BN_OUT_F   = 12;
    parameter int FC_OUT_F      = 10;

    // ------------------------------------------------------------------
    // Convolution weights and biases.
    // ------------------------------------------------------------------
    parameter int CONV1_W_F = 15;
    parameter int CONV1_B_F = 15;
    parameter int CONV2_W_F = 14;
    parameter int CONV2_B_F = 15;
    parameter int CONV3_W_F = 15;
    parameter int CONV3_B_F = 15;

    // ------------------------------------------------------------------
    // Batch-normalization affine parameters: y = A*x + B.
    // ------------------------------------------------------------------
    parameter int BN1_A_F = 11;
    parameter int BN1_B_F = 13;
    parameter int BN2_A_F = 15;
    parameter int BN2_B_F = 14;
    parameter int BN3_A_F = 15;
    parameter int BN3_B_F = 14;

    parameter int FC_BN_A_F = 13;
    parameter int FC_BN_B_F = 12;

    // ------------------------------------------------------------------
    // GRU and fully-connected parameters.
    // ------------------------------------------------------------------
    parameter int GRU_WR_F = 15;
    parameter int GRU_WZ_F = 14;
    parameter int GRU_WH_F = 15;
    parameter int GRU_UR_F = 15;
    parameter int GRU_UZ_F = 15;
    parameter int GRU_UH_F = 14;
    parameter int GRU_BR_F = 15;
    parameter int GRU_BZ_F = 15;
    parameter int GRU_BH_F = 15;

    parameter int FC1_W_F = 15;
    parameter int FC1_B_F = 14;
    parameter int FC_OUT_W_F = 15;
    parameter int FC_OUT_B_F = 15;

    // ------------------------------------------------------------------
    // Derived shifts used by the current verified RTL.
    // ------------------------------------------------------------------
    parameter int CONV1_BIAS_SHIFT =
        EEG_INPUT_F + CONV1_W_F - CONV1_B_F;       // 12
    parameter int CONV1_OUTPUT_SHIFT =
        EEG_INPUT_F + CONV1_W_F - CONV1_OUT_F;     // 14

    parameter int CONV2_BIAS_SHIFT =
        RELU1_OUT_F + CONV2_W_F - CONV2_B_F;       // 10
    parameter int CONV2_OUTPUT_SHIFT =
        RELU1_OUT_F + CONV2_W_F - CONV2_OUT_F;     // 15

    parameter int CONV3_BIAS_SHIFT =
        POOL1_OUT_F + CONV3_W_F - CONV3_B_F;       // 12
    parameter int CONV3_OUTPUT_SHIFT =
        POOL1_OUT_F + CONV3_W_F - CONV3_OUT_F;     // 17

    parameter int BN1_BIAS_SHIFT =
        CONV1_OUT_F + BN1_A_F - BN1_B_F;           // 11
    parameter int BN1_OUTPUT_SHIFT =
        CONV1_OUT_F + BN1_A_F - BN1_OUT_F;         // 13

    parameter int BN2_BIAS_SHIFT =
        CONV2_OUT_F + BN2_A_F - BN2_B_F;           // 11
    parameter int BN2_OUTPUT_SHIFT =
        CONV2_OUT_F + BN2_A_F - BN2_OUT_F;         // 14

    parameter int BN3_BIAS_SHIFT =
        CONV3_OUT_F + BN3_A_F - BN3_B_F;           // 11
    parameter int BN3_OUTPUT_SHIFT =
        CONV3_OUT_F + BN3_A_F - BN3_OUT_F;         // 13

    parameter int RELU1_LEFT_SHIFT = RELU1_OUT_F - BN1_OUT_F; // 0
    parameter int RELU2_LEFT_SHIFT = RELU2_OUT_F - BN2_OUT_F; // 1
    parameter int RELU3_LEFT_SHIFT = RELU3_OUT_F - BN3_OUT_F; // 0

    parameter int POOL1_FORMAT_SHIFT = POOL1_OUT_F - RELU2_OUT_F; // 0
    parameter int POOL2_FORMAT_SHIFT = POOL2_OUT_F - RELU3_OUT_F; // 1
endpackage
