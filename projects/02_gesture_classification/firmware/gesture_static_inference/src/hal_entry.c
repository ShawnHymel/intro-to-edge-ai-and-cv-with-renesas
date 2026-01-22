/* Custom libraries */
#include "common_utils.h"
#include "hal_data.h"
#include "inference/model.h"
#include "test_sample.h"

#define NUM_CLASSES 5
const char* class_names[] = {
    "_idle",
    "_other",
    "alpha",
    "beta",
    "gamma"
};

void hal_entry(void)
{
    fsp_err_t err;
    float* input_ptr;
    float* output_ptr;

    /* Initialize UART */
    TERM_INIT();
    APP_PRINT("\r\nGesture Static Inference Test\r\n");

    /* Workaround: Copy config with correct secure/privilege settings */
    rm_ethosu_cfg_t ethosu_cfg = g_rm_ethosu0_cfg;
    ethosu_cfg.secure_enable = 1;
    ethosu_cfg.privilege_enable = 1;

    /* Initialize Ethos core with modified config */
    err = RM_ETHOSU_Open(&g_rm_ethosu0_ctrl, &ethosu_cfg);
    if (FSP_SUCCESS != err) {
        APP_PRINT("  ERROR: RM_ETHOSU_Open failed with code: %d\r\n", err);
        while(1);
    }
    APP_PRINT("Ethos-U NPU successfully initialized!\r\n");

    /* Get pointers to model input/output buffers */
    input_ptr = GetModelInputPtr_input();
    output_ptr = GetModelOutputPtr_output_70010();

    /* Copy test data */
    memcpy(input_ptr, test_input, TEST_INPUT_SIZE * sizeof(float));

    /* Run inference */
    RunModel(true);

    /* Find predicted class */
    int predicted_class = 0;
    float max_logit = output_ptr[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (output_ptr[i] > max_logit) {
            max_logit = output_ptr[i];
            predicted_class = i;
        }
    }

    /* Print results */
    APP_PRINT("\r\nResults:\r\n");
    APP_PRINT("  Predicted: %s (class %d)\r\n", class_names[predicted_class], predicted_class);
    APP_PRINT("\r\n  Logits:\r\n");
    for (int i = 0; i < NUM_CLASSES; i++) {
        APP_PRINT("    %-15s %12.6f\r\n", class_names[i], output_ptr[i]);
    }
    APP_PRINT("\r\nTest Complete\r\n");

    while(1) {
        R_BSP_SoftwareDelay(1000, BSP_DELAY_UNITS_MILLISECONDS);
    }
}
